# ===========================================
# SISTEMA DE CORREÇÃO DE CARTÃO RESPOSTA
# ===========================================
# 
# TECNOLOGIAS UTILIZADAS:
# - OCR (Tesseract): Para extrair TEXTOS (escola, nome do aluno)
# - OMR/OpenCV: Para detectar ALTERNATIVAS marcadas (bolhas pintadas)
# - GEMINI Vision: Para análise inteligente e validação de detecções
# - PDF2IMAGE: Para converter PDFs em imagens processáveis
#
# ESTRUTURA:
# 1. PDF → Conversão para imagem (se necessário)
# 2. OCR → Cabeçalho (escola, nome, nascimento, turma)  
# 3. OMR → 52 questões organizadas em 4 colunas (A, B, C, D)
# 4. GEMINI → Validação e correção das detecções
# ===========================================

from PIL import Image
import pytesseract
import cv2
import numpy as np
import re
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from datetime import datetime
import os
import base64
import io
import tempfile
import shutil
import argparse
from typing import List, Dict, Optional
from sklearn.cluster import KMeans

# Importação do processador de PDF
try:
    from pdf_processor_simple import process_pdf_file, is_pdf_file, setup_pdf_support
    PDF_PROCESSOR_AVAILABLE = True
except ImportError:
    PDF_PROCESSOR_AVAILABLE = False

# Importação condicional do Gemini
try:
    import google.generativeai as genai
    GEMINI_DISPONIVEL = True
    print("✅ Gemini disponível")
except ImportError:
    GEMINI_DISPONIVEL = True
    print("⚠️ Gemini não disponível (google-generativeai não instalado)")
    genai = None

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

EXTENSOES_SUPORTADAS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf', '.webp')
DRIVE_MIME_TO_EXT = {
    'application/pdf': '.pdf',
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/jpg': '.jpg',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff',
    'image/webp': '.webp'
}

# ===========================================
# SEÇÃO 0: PREPROCESSAMENTO DE ARQUIVOS (PDF/IMAGEM)
# ===========================================

def preprocessar_arquivo(file_path: str, tipo: str = "aluno") -> str:
    """
    Preprocessa arquivo (PDF ou imagem) para garantir que temos uma imagem processável
    
    Args:
        file_path: Caminho para o arquivo (PDF ou imagem)
        tipo: Tipo do arquivo ("aluno", "gabarito") para logs
        
    Returns:
        Caminho para a imagem processável
    """
    print(f"\n🔄 PREPROCESSANDO ARQUIVO {tipo.upper()}: {os.path.basename(file_path)}")
    
    # Verificar se arquivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    # Se for PDF, converter para imagem
    if is_pdf_file(file_path) and PDF_PROCESSOR_AVAILABLE:
        print(f"Arquivo PDF detectado - convertendo para imagem...")
        try:
            best_image, temp_files = process_pdf_file(file_path, keep_temp_files=False)
            print(f" Imagem gerada: {os.path.basename(best_image)}")
            return best_image
        except Exception as e:
            print(f"❌ Erro ao converter PDF: {e}")
            raise e
    
    # Se for imagem, verificar se é válida
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            # Tentar carregar a imagem para validar
            img = Image.open(file_path)
            img.verify()  # Verificar se a imagem é válida
            return file_path
        except Exception as e:
            raise Exception(f"Arquivo de imagem inválido: {e}")
    
    # Tipo de arquivo não suportado
    else:
        if is_pdf_file(file_path) and not PDF_PROCESSOR_AVAILABLE:
            raise Exception(
                "Arquivo PDF detectado, mas processador de PDF não está disponível.\n"
                "Instale com: pip install pdf2image"
            )
        else:
            raise Exception(
                f"Tipo de arquivo não suportado: {file_path}\n"
                "Formatos suportados: PDF, PNG, JPG, JPEG, BMP, TIFF"
            )

def listar_arquivos_suportados(diretorio: str = ".") -> dict:
    """
    Lista todos os arquivos suportados no diretório
    
    Returns:
        Dicionário com listas de arquivos por tipo
    """
    arquivos_suportados = {
        'imagens': [],
        'pdfs': [],
        'todos': []
    }
    
    extensoes_imagem = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for arquivo in os.listdir(diretorio):
        caminho_completo = os.path.join(diretorio, arquivo)
        if os.path.isfile(caminho_completo):
            if arquivo.lower().endswith('.pdf'):
                arquivos_suportados['pdfs'].append(arquivo)
                arquivos_suportados['todos'].append(arquivo)
            elif arquivo.lower().endswith(extensoes_imagem):
                arquivos_suportados['imagens'].append(arquivo)
                arquivos_suportados['todos'].append(arquivo)
    
    return arquivos_suportados

# ===========================================
# SEÇÃO 1: OCR - EXTRAÇÃO DE TEXTOS DO CABEÇALHO
# ===========================================

def recortar_cabecalho(image_path):
    """OCR: Recorta região do cabeçalho para extração de textos"""
    img = Image.open(image_path)
    width, height = img.size
    
    # Crop mais amplo para capturar todo o cabeçalho
    crop_box = (int(width*0.05), int(height*0.02), int(width*0.61), int(height*0.40))
    cabecalho = img.crop(crop_box)
    
    # Salvar imagem de debug
    cabecalho.save("debug_cabecalho.png")

    return cabecalho

def extrair_campos_cabecalho(texto):
    """OCR: Extrai dados do aluno usando Tesseract"""
    escola = aluno = nascimento = turma = "N/A"
    linhas = [l.strip() for l in texto.split("\n") if len(l.strip()) > 2]
    
    print(f"🔍 DEBUG EXTRAÇÃO DE CAMPOS:")
    print(f"Total de linhas válidas: {len(linhas)}")
    for i, linha in enumerate(linhas):
        print(f"  Linha {i}: '{linha}'")
    
    # === EXTRAÇÃO FLEXÍVEL POR POSICIONAMENTO ESTRUTURAL ===
    
    # Escola - Abordagem flexível baseada em posição e contexto
    for i, l in enumerate(linhas):
        print(f"🏫 Analisando linha {i} para escola: '{l}'")
        
        # ESTRATÉGIA 1: Detectar qualquer variação de "Nome escola" (muito flexível)
        if re.search(r'Nome[a-z]*\s*(d[aeo]|de)?\s*(Escol[aeo]|escola)', l, re.I):
            print(f"✅ Encontrou padrão flexível de 'Nome escola' na linha {i}")
            
            # Extrair tudo depois do padrão até encontrar palavras de parada
            resto_linha = re.sub(r'^.*?Nome[a-z]*\s*(d[aeo]|de)?\s*(Escol[aeo]|escola)[:\s]*', '', l, flags=re.I)
            
            if resto_linha and len(resto_linha.strip()) > 2:
                # Limpar palavras de parada comuns
                escola = re.sub(r'\s*(RESULTADO\s+FINAL|TURMA|DATA|NASCIMENTO).*$', '', resto_linha, flags=re.I).strip()
                escola = re.sub(r'[\|\:\!]+\s*$', '', escola).strip()  # Remove caracteres finais
                
                if len(escola) > 2:
                    print(f"✅ Escola extraída da mesma linha: '{escola}'")
                    break
            
            # Se não achou na mesma linha, procurar na próxima
            if i+1 < len(linhas):
                candidata = linhas[i+1].strip()
                if len(candidata) > 2 and not re.search(r'nome|completo|data|nascimento|turma', candidata, re.I):
                    escola = candidata
                    print(f"✅ Escola extraída da linha seguinte: '{escola}'")
                    break
    
    # Nome completo - Abordagem flexível baseada em posição e contexto
    for i, l in enumerate(linhas):
        print(f"👤 Analisando linha {i} para nome: '{l}'")
        
        # ESTRATÉGIA 1: Detectar qualquer variação de "Nome" (SEM escola)
        if re.search(r'Nome[a-z]*\s*(completo)?[:\s]*', l, re.I) and not re.search(r'escola', l, re.I):
            print(f"✅ Encontrou padrão flexível de 'Nome' na linha {i}")
            
            # Extrair conteúdo depois do padrão - CORRIGIDO para remover "completo" também
            resto_linha = re.sub(r'^.*?Nome[a-z]*\s*(completo\s*)?[:\s]*', '', l, flags=re.I)
            
            if resto_linha and len(resto_linha.strip()) > 2:
                # Limpar palavras de parada
                nome = re.sub(r'\s*(DATA|NASCIMENTO|TURMA|RESULTADO).*$', '', resto_linha, flags=re.I).strip()
                nome = re.sub(r'[\|\:\!]+\s*$', '', nome).strip()
                
                # Verificar se parece ser um nome (pelo menos 2 palavras ou 1 palavra com mais de 3 chars)
                # E NÃO contenha palavras relacionadas a escola
                if (len(nome) > 2 and 
                    not re.search(r'escola|municipal|estadual|particular|escol|fundamental|médio', nome, re.I) and
                    (len(nome.split()) >= 2 or len(nome) > 3)):
                    aluno = nome
                    print(f"✅ Aluno extraído da mesma linha: '{aluno}'")
                    break
            
            # Se não achou na mesma linha, procurar nas próximas 2 linhas
            for next_i in range(i+1, min(i+3, len(linhas))):
                candidato_bruto = linhas[next_i].strip()
                
                # APLICAR O MESMO REGEX na linha seguinte também
                candidato = re.sub(r'^.*?Nome[a-z]*\s*(completo\s*)?[:\s]*', '', candidato_bruto, flags=re.I)
                
                # Se depois do regex ainda sobrou algo válido
                if candidato and len(candidato.strip()) > 2:
                    candidato = re.sub(r'\s*(DATA|NASCIMENTO|TURMA|RESULTADO).*$', '', candidato, flags=re.I).strip()
                    candidato = re.sub(r'[\|\:\!]+\s*$', '', candidato).strip()
                    
                    # Verificar se é um nome válido
                    if (len(candidato) > 2 and 
                        not re.search(r'data|nascimento|turma|avaliação|cartão|escola|municipal|resultado|fundamental', candidato, re.I) and
                        not candidato.upper() == candidato and  # Não é tudo maiúsculo (evita títulos)
                        (len(candidato.split()) >= 2 or len(candidato) > 3)):  # 2+ palavras OU 1 palavra longa
                        
                        aluno = candidato
                        print(f"✅ Aluno extraído da linha {next_i} (com regex): '{aluno}'")
                        break
                else:
                    # Se não há padrão "Nome", pegar linha direta (fallback)
                    if (len(candidato_bruto) > 2 and 
                        not re.search(r'data|nascimento|turma|avaliação|cartão|escola|municipal|resultado|fundamental', candidato_bruto, re.I) and
                        not candidato_bruto.upper() == candidato_bruto and  # Não é tudo maiúsculo
                        (len(candidato_bruto.split()) >= 2 or len(candidato_bruto) > 3)):
                        
                        aluno = candidato_bruto
                        print(f"✅ Aluno extraído da linha {next_i} (direto): '{aluno}'")
                        break
            
            if aluno:  # Se encontrou, sair do loop principal
                break
        
        # ESTRATÉGIA 2: Detectar linha que parece ser um nome (heurística inteligente)
        elif (i > 5 and  # Não pegar títulos no início
              len(l.split()) >= 2 and  # Pelo menos 2 palavras
              len(l) > 5 and len(l) < 50 and  # Tamanho razoável
              l[0].isupper() and  # Primeira letra maiúscula
              not l.upper() == l and  # Não é tudo maiúsculo
              not re.search(r'avaliação|diagnóstica|cartão|resposta|escola|municipal|turma|resultado|ensino|fundamental|instruções|julho|data|nascimento|preencha', l, re.I)):
            
            aluno = l.strip()
            print(f"✅ Aluno detectado por heurística de nome (linha {i}): '{aluno}'")
            break
    
    # Data de nascimento
    for l in linhas:
        nasc_match = re.search(r'(\d{2})[\/\-](\d{1,2})[\/\-](\d{2,4})', l)
        if nasc_match:
            nascimento = f"{nasc_match.group(1)}/{nasc_match.group(2)}/{nasc_match.group(3)}"
            break
    
    # Turma
    for i, l in enumerate(linhas):
        if "turma" in l.lower():
            turma_match = re.search(r'Turma[: ]*([A-Za-z0-9ºª\-_/]{1,8})', l, re.I)
            if turma_match and turma_match.group(1).strip():
                turma = turma_match.group(1).strip()
            elif i+1 < len(linhas):
                candidato = linhas[i+1].strip()
                if 1 < len(candidato) <= 8:
                    turma = candidato
            break
    
    return {
        "Escola": escola,
        "Aluno": aluno,
        "Nascimento": nascimento,
        "Turma": turma
    }

# ===========================================
# SEÇÃO 2: OMR - DETECÇÃO DE ALTERNATIVAS MARCADAS
# ===========================================

def detectar_respostas_pdf(image_path, debug=False):
    """
    Detecta as respostas marcadas no cartão resposta convertido de PDF.
    Otimizado para imagens de alta resolução com parâmetros específicos para PDFs.
    Retorna uma lista com as respostas ['A', 'B', 'C', 'D', '?'] onde '?' significa não detectado.
    """
    
    # CARREGAR E PREPROCESSAR IMAGEM
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem {image_path}")
        return ['?'] * 52
    
    # Verificar se é uma imagem de alta resolução (provavelmente de PDF)
    height, width = image.shape[:2]
    is_high_res = width > 3000 or height > 2000
    
    print(f"📐 Imagem PDF detectada: {width}x{height} pixels")
    
    # CROP ESPECÍFICO para alta resolução - área das questões
    # Para PDFs, usar proporções similares mas ajustadas
    crop = image[int(height*0.55):int(height*0.98), int(width*0.02):int(width*0.98)]
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Parâmetros otimizados para PDF de alta resolução
    if is_high_res:
        # Para alta resolução, usar parâmetros mais refinados
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        # Threshold OTSU automático para melhor adaptação
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Kernel maior para operações morfológicas em alta resolução
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Faixa de área adaptada para alta resolução
        area_min = 600   # Área mínima para alta resolução
        area_max = 6000  # Área máxima para alta resolução
        circularity_min = 0.12  # Menos rigoroso para marcações irregulares
        intensity_max = 60      # Intensidade máxima ajustada
        
    else:
        # Parâmetros padrão para resolução normal
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 30, 155, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        area_min = 150
        area_max = 800
        circularity_min = 0.25
        intensity_max = 35
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bolhas_pintadas = []
    
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        
        if area_min < area < area_max:
            # Verificar circularidade
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > circularity_min:
                    # Verificar aspect ratio mais flexível para PDFs
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h
                    
                    if 0.2 <= aspect_ratio <= 5.0:  # Mais flexível para PDFs
                        # Calcular centro
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Verificar se está na região válida
                            crop_height, crop_width = crop.shape[:2]
                            margem_segura = max(30, min(crop_width, crop_height) // 40)
                            
                            if (margem_segura < cx < crop_width - margem_segura and 
                                margem_segura < cy < crop_height - margem_segura):
                                
                                # Calcular intensidade e preenchimento
                                mask = np.zeros(gray.shape, dtype=np.uint8)
                                cv2.drawContours(mask, [cnt], -1, 255, -1)
                                intensidade_media = cv2.mean(gray, mask=mask)[0]
                                
                                pixels_escuros = cv2.countNonZero(cv2.bitwise_and(thresh, mask))
                                percentual_preenchimento = pixels_escuros / area
                                
                                # Critérios de aceitação mais flexíveis para PDFs
                                aceita_marcacao = False
                                
                                # Marcação escura bem preenchida
                                if intensidade_media < intensity_max and percentual_preenchimento > 0.25:
                                    aceita_marcacao = True
                                
                                # Marcação circular mesmo que pouco preenchida
                                elif circularity > 0.3 and 0.1 <= percentual_preenchimento <= 0.9 and intensidade_media < intensity_max + 20:
                                    aceita_marcacao = True
                                
                                # Marcação grande com baixa intensidade (marca grossa)
                                elif area > area_min * 2 and intensidade_media < intensity_max + 30 and percentual_preenchimento > 0.15:
                                    aceita_marcacao = True
                                
                                if aceita_marcacao:
                                    bolhas_pintadas.append((cx, cy, cnt, intensidade_media, area, circularity, percentual_preenchimento))
    
    if debug:
        print(f"=== DEBUG PDF - ALTA RESOLUÇÃO ===")
        print(f"Área do crop: {crop.shape[1]}x{crop.shape[0]} pixels")
        print(f"Parâmetros usados - Área: {area_min}-{area_max}, Circ: {circularity_min:.2f}, Int: {intensity_max}")
        for i, (cx, cy, _, intensidade, area, circ, preenchimento) in enumerate(bolhas_pintadas):
            continue

    # Verificar se temos bolhas suficientes
    if len(bolhas_pintadas) < 6:  # Mínimo mais baixo para PDFs
        print(f"⚠️ Poucas bolhas detectadas em PDF ({len(bolhas_pintadas)}). Tentando processamento simplificado.")
        if len(bolhas_pintadas) < 2:
            return ['?'] * 52
    
    # Organização em colunas usando clustering adaptativo
    xs = np.array([b[0] for b in bolhas_pintadas], dtype=np.float32).reshape(-1, 1)
    
    # Determinar número de colunas baseado no número de bolhas
    num_colunas = min(4, max(1, len(bolhas_pintadas) // 6))  # Mais flexível para PDFs
    
    if num_colunas < 4:
        print(f"⚠️ Detectadas apenas {num_colunas} colunas possíveis em PDF. Processamento adaptativo.")
    
    from sklearn.cluster import KMeans
    
    try:
        kmeans = KMeans(n_clusters=num_colunas, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(xs)
        centroids = kmeans.cluster_centers_
        
        # Ordenar colunas da esquerda para direita
        order = np.argsort(centroids.flatten())
        col_mapping = {old: new for new, old in enumerate(order)}
        
        # Agrupar bolhas por coluna
        colunas = [[] for _ in range(num_colunas)]
        for i, (cx, cy, cnt, intensidade, area, circ, preenchimento) in enumerate(bolhas_pintadas):
            col_id = col_mapping[cluster_labels[i]]
            colunas[col_id].append((cx, cy, cnt, intensidade, area, circ, preenchimento))
        
        # Ordenar bolhas em cada coluna por posição Y (de cima para baixo)
        for col in colunas:
            col.sort(key=lambda x: x[1])  # Ordenar por cy (coordenada Y)
        
        # Mapear questões para respostas usando distribuição equilibrada
        respostas = ['?'] * 52
        questoes_por_coluna = 52 // num_colunas
        extra_questoes = 52 % num_colunas
        
        questao = 1
        
        for col_idx, coluna in enumerate(colunas):
            # Calcular quantas questões esta coluna deve ter
            questoes_nesta_coluna = questoes_por_coluna + (1 if col_idx < extra_questoes else 0)
            
            for linha_idx, (cx, cy, cnt, intensidade, area, circ, preenchimento) in enumerate(coluna):
                if linha_idx < questoes_nesta_coluna and questao <= 52:
                    # Determinar a resposta baseada na posição X relativa
                    # Para PDFs, usar algoritmo mais sofisticado
                    
                    # Coletar todas as posições X únicas na mesma linha Y (aproximadamente)
                    tolerancia_y = max(50, crop.shape[0] // 30)  # Tolerância proporcional
                    mesma_linha = []
                    
                    for outras_bolhas in colunas:
                        for ocx, ocy, _, _, _, _, _ in outras_bolhas:
                            if abs(ocy - cy) <= tolerancia_y:  # Mesma linha
                                mesma_linha.append(ocx)
                    
                    if len(mesma_linha) >= 2:
                        mesma_linha = sorted(set(mesma_linha))
                        if cx in mesma_linha:
                            pos_x = mesma_linha.index(cx)
                            if pos_x < 4:
                                resposta = ['A', 'B', 'C', 'D'][pos_x]
                            else:
                                resposta = 'D'
                        else:
                            resposta = 'A'
                    else:
                        # Fallback: usar posição relativa na coluna
                        resposta = ['A', 'B', 'C', 'D'][col_idx % 4]
                    
                    respostas[questao - 1] = resposta
                    if debug:
                        continue
                    questao += 1
        
        return respostas
        
    except Exception as e:
        print(f"Erro no clustering para PDF: {e}")
        return ['?'] * 52


def detectar_respostas(image_path, debug=False):
    """
    OMR: Detecta APENAS alternativas pintadas usando OpenCV
    Versão aprimorada para lidar com PDFs convertidos
    """
    img_cv = cv2.imread(image_path)
    height, width = img_cv.shape[:2]
    
    # Crop ESPECÍFICO para APENAS as 4 colunas de questões (área retangular completa)
    # Expandido para capturar todas as 4 colunas de questões 1-52 + numeração
    crop = img_cv[int(height*0.60):int(height*0.98), int(width*0.02):int(width*0.98)]
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro suave
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # FOCO: Threshold MUITO restritivo para detectar APENAS marcações PRETAS
    _, thresh = cv2.threshold(blur, 30, 155, cv2.THRESH_BINARY_INV) 
    
    # Operações morfológicas para preencher bolhas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bolhas_pintadas = []
    
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        
        # FOCO: Área mais específica para bolinhas das questões (não texto ou elementos gráficos)
        if 150 < area < 800:  # Área mais ampla para capturar diferentes tamanhos de marcação
            # Verificar se tem formato aproximadamente circular/oval (mais flexível)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # FOCO: Circularidade bem mais flexível para aceitar marcações irregulares
                if circularity > 0.25:  # Muito menos rigoroso para aceitar contornos e formas irregulares
                    # Verificar aspect ratio mais flexível
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h
                    
                    if 0.3 <= aspect_ratio <= 3.0:  # Aceita formas bem mais alongadas ou irregulares
                        # Calcular centro
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Verificar se está na região das questões
                            crop_height, crop_width = crop.shape[:2]
                            if (20 < cx < crop_width - 20 and 20 < cy < crop_height - 20):
                                
                                # MELHORIA: Verificar densidade de pixels escuros na bolha
                                mask = np.zeros(gray.shape, dtype=np.uint8)
                                cv2.drawContours(mask, [cnt], -1, 255, -1)
                                intensidade_media = cv2.mean(gray, mask=mask)[0]
                                
                                # Calcular percentual de pixels escuros na bolha
                                pixels_escuros = cv2.countNonZero(cv2.bitwise_and(thresh, mask))
                                percentual_preenchimento = pixels_escuros / area
                                
                                # FOCO: ACEITAR MARCAÇÕES VARIADAS (muito mais flexível)
                                # Aceita marcações pretas, riscadas, contornadas, ou parcialmente preenchidas
                                if intensidade_media < 35 and percentual_preenchimento > 0.5:
                                    bolhas_pintadas.append((cx, cy, cnt, intensidade_media, area, circularity, percentual_preenchimento))
                                
                                # DETECÇÃO ADICIONAL: Contornos circulares (como questões 20, 21, 26)
                                # Para marcações que são apenas contornos com pouco preenchimento
                                elif circularity > 0.4 and 0.2 <= percentual_preenchimento <= 0.8 and intensidade_media < 45:
                                    bolhas_pintadas.append((cx, cy, cnt, intensidade_media, area, circularity, percentual_preenchimento))
    
    if debug:
        print(f"Bolhas pintadas detectadas: {len(bolhas_pintadas)}")
        print(f"Área do crop: {crop.shape[1]}x{crop.shape[0]} pixels")
        for i, (cx, cy, _, intensidade, area, circ, preenchimento) in enumerate(bolhas_pintadas):
            continue
        salvar_debug_deteccao(image_path, bolhas_pintadas, crop)
    
    # Verificar se temos bolhas suficientes para processamento
    if len(bolhas_pintadas) < 4:
        print(f"⚠️ Poucas bolhas detectadas ({len(bolhas_pintadas)}). Retornando lista vazia.")
        return ['?'] * 52
    
    # MELHORIA: Organização mais precisa usando KMeans para detectar as 4 colunas
    
    # 1) Após montar bolhas_pintadas, separe só os 'cx' (centros X)
    xs = np.array([b[0] for b in bolhas_pintadas], dtype=np.float32).reshape(-1, 1)

    # 2) Determinar número de colunas baseado no número de bolhas
    num_colunas = min(4, max(1, len(bolhas_pintadas) // 3))  # Pelo menos 3 bolhas por coluna
    
    if num_colunas < 4:
        print(f"⚠️ Detectadas apenas {num_colunas} colunas possíveis. Processamento simplificado.")
    
    # 3) Descubra as BANDAS VERTICAIS (colunas de questões) via KMeans
    k_cols = KMeans(n_clusters=num_colunas, n_init=10, random_state=0).fit(xs)
    col_idx_por_bolha = k_cols.predict(xs)
    centros_cols = sorted(k_cols.cluster_centers_.flatten())  # esquerda→direita

    # Mapeie cada bolha para a coluna correta usando a ordem dos centros
    # (reindexar para 0..num_colunas-1 na ordem esquerda→direita)
    ordem_cols = np.argsort(k_cols.cluster_centers_.flatten())
    remap = {int(c): i for i, c in enumerate(ordem_cols)}

    bolhas_por_coluna = [[] for _ in range(num_colunas)]
    for bolha, c_orig in zip(bolhas_pintadas, col_idx_por_bolha):
        if remap[int(c_orig)] < num_colunas:
            bolhas_por_coluna[remap[int(c_orig)]].append(bolha)

    # 4) Para CADA coluna, processar as questões
    letras = ['A', 'B', 'C', 'D']
    respostas_finais = ['?'] * 52

    for col_idx, bolhas_coluna in enumerate(bolhas_por_coluna):
        if not bolhas_coluna:
            continue

        # Se há bolhas suficientes na coluna, tentar detectar alternativas A-D
        if len(bolhas_coluna) >= 4:
            xs_col = np.array([b[0] for b in bolhas_coluna], dtype=np.float32).reshape(-1, 1)
            num_alternativas = min(4, len(bolhas_coluna))
            k_opts = KMeans(n_clusters=num_alternativas, n_init=10, random_state=0).fit(xs_col)
            centros_opts = k_opts.cluster_centers_.flatten()
            ordem_opts = np.argsort(centros_opts)  # esquerda→direita ⇒ A,B,C,D
        else:
            # Processamento simplificado se há poucas bolhas
            ordem_opts = list(range(len(bolhas_coluna)))
            centros_opts = [b[0] for b in bolhas_coluna]

        # Agrupe por LINHAS usando tolerância mais flexível
        ys = sorted([b[1] for b in bolhas_coluna])
        dy = np.median(np.diff(ys)) if len(ys) > 5 else 25  # Espaçamento base maior
        tolerance_y = max(18, int(dy * 0.7))  # 70% do espaçamento (mais tolerante)

        linhas = []
        for bolha in sorted(bolhas_coluna, key=lambda b: b[1]):  # por Y
            cy = bolha[1]
            if not linhas or abs(cy - linhas[-1][0][1]) > tolerance_y:
                linhas.append([bolha])
            else:
                linhas[-1].append(bolha)

        # Ordene as linhas por Y
        linhas.sort(key=lambda linha: linha[0][1])
        
        # Rastrear linhas usadas com conjunto de índices
        linhas_usadas = set()

        # Cada coluna tem 13 questões - MAPEAMENTO CORRETO
        offset_questao = col_idx * 13
        
        # Calcular as posições Y esperadas das 13 questões na coluna
        if linhas:
            y_min = min(linha[0][1] for linha in linhas)
            y_max = max(linha[0][1] for linha in linhas)
            altura_total = y_max - y_min
            espacamento_questao = altura_total / 12 if len(linhas) > 1 else 25  # 12 intervalos para 13 questões
        else:
            continue
            
        # Para cada questão (0-12) na coluna
        num_questoes = min(13, 52 - offset_questao)  # Não exceder 52 questões total
        
        for questao_idx in range(num_questoes):
            q = offset_questao + questao_idx
            if q >= 52:
                break
                
            # Calcular posição Y esperada desta questão
            y_esperado = y_min + (questao_idx * espacamento_questao)
            
            # Encontrar a linha mais próxima desta posição Y
            linha_mais_proxima = None
            linha_mais_proxima_idx = -1
            menor_distancia = float('inf')
            
            for idx, linha in enumerate(linhas):
                if idx in linhas_usadas:  # Pular linhas já usadas
                    continue
                    
                y_linha = linha[0][1]  # Y da primeira bolha da linha
                distancia = abs(y_linha - y_esperado)
                
                # Tolerância: aceitar linha se estiver dentro de uma janela mais ampla
                if distancia < espacamento_questao * 1.5 and distancia < menor_distancia:
                    menor_distancia = distancia
                    linha_mais_proxima = linha
                    linha_mais_proxima_idx = idx
            
            if linha_mais_proxima is not None:
                # Marcar linha como usada
                linhas_usadas.add(linha_mais_proxima_idx)
                
                # Escolha a bolha mais "preta" da linha
                bolha_marcada = min(linha_mais_proxima, key=lambda b: b[3] - (b[6] * 40))

                # Descobrir a ALTERNATIVA pela posição X
                cx = bolha_marcada[0]
                if len(centros_opts) >= 4:
                    idx_opt = np.argmin([abs(cx - centros_opts[j]) for j in range(len(centros_opts))])
                    if idx_opt < len(ordem_opts):
                        letra_idx = int(np.where(ordem_opts == idx_opt)[0][0])
                        if letra_idx < len(letras):
                            letra = letras[letra_idx]
                        else:
                            letra = '?'
                    else:
                        letra = '?'
                else:
                    # Processamento simplificado
                    letra = letras[0] if len(letras) > 0 else '?'

                respostas_finais[q] = letra
                
                if debug:
                    intensidade = bolha_marcada[3]
                    preenchimento = bolha_marcada[6]
                    
            
            # Se não encontrou nenhuma linha, deixar como '?' (já é o padrão)
            if respostas_finais[q] == '?' and debug:
                print(f"Questão {q + 1}: ? (col {col_idx + 1}, linha {questao_idx + 1}, sem marcação detectada)")
    
    if debug:
        questoes_detectadas = sum(1 for r in respostas_finais if r != '?')
        print(f"Total de questões detectadas: {questoes_detectadas}")
        print(f"Respostas: {respostas_finais}")
    
    return respostas_finais

def salvar_debug_deteccao(image_path, bolhas_pintadas, crop):
    """Salva imagem de debug com as bolhas detectadas marcadas"""
    debug_img = crop.copy()
    
    for cx, cy, cnt, intensidade, area, circ, preenchimento in bolhas_pintadas:
        # Marcar com círculo verde
        cv2.circle(debug_img, (cx, cy), 8, (0, 255, 0), 2)
        # Adicionar texto com intensidade
        cv2.putText(debug_img, f"{intensidade:.0f}", (cx-15, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Salvar imagem de debug
    filename = image_path.replace('.jpg', '').replace('.png', '')
    debug_filename = f"debug_{os.path.basename(filename)}.png"
    cv2.imwrite(debug_filename, debug_img)

# ===========================================
# SEÇÃO 3: GEMINI - ANÁLISE INTELIGENTE DE IMAGENS
# ===========================================

def configurar_gemini():
    """Configura o Gemini API"""
    if not GEMINI_DISPONIVEL:
        print("❌ Gemini não está disponível")
        print("💡 Para instalar: pip install google-generativeai")
        return None
        
    try:
        # Configure sua API key do Gemini aqui
        # Obtenha em: https://makersuite.google.com/app/apikey
        GEMINI_API_KEY = "AIzaSyCZJ0GhpbMi2koxkrdjjCqWYys6yIVM4v0"  # Substitua pela sua chave
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Testar conexão
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        print("✅ Gemini configurado com sucesso!")
        return model
        
    except Exception as e:
        print(f"❌ Erro ao configurar Gemini: {e}")
        print("💡 Certifique-se de:")
        print("   1. Instalar: pip install google-generativeai")
        print("   2. Configurar sua API key do Gemini")
        print("   3. Verificar sua conexão com internet")
        return None

def converter_imagem_para_base64(image_path):
    """Converte imagem para base64 para envio ao Gemini"""
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            
        # Converter para PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        return image
        
    except Exception as e:
        print(f"❌ Erro ao converter imagem: {e}")
        return None

def analisar_cartao_com_gemini(model, image_path, tipo_analise="gabarito"):
    """
    Usa Gemini Vision para analisar cartão resposta
    """
    if not model:
        print("⚠️ Gemini não configurado, pulando análise inteligente")
        return None
        
    try:
        # Converter imagem
        image = converter_imagem_para_base64(image_path)
        if not image:
            return None
        
        # Prompt especializado para cartão resposta
        if tipo_analise == "gabarito":
            prompt = """
            Analise esta imagem de um cartão resposta (GABARITO) e identifique APENAS as bolinhas PRETAS marcadas.

            INSTRUÇÕES ESPECÍFICAS:
            1. Este é um cartão com 52 questões organizadas em 4 colunas (1-13, 14-26, 27-39, 40-52)
            2. Cada questão tem 4 alternativas: A, B, C, D
            3. DETECTE APENAS bolinhas completamente PRETAS/PINTADAS com tinta preta
            4. IGNORE qualquer outra cor (verde, azul, vermelho, etc.)
            5. IGNORE círculos vazios ou apenas contornados
            6. IGNORE marcações que não sejam tinta preta sólida
      

            FOCO: Apenas marcações PRETAS sólidas e bem preenchidas.

            FORMATO DE RESPOSTA:
            Retorne apenas uma lista Python com 52 elementos, exemplo:
            ['A', 'B', '?', 'C', 'D', 'A', '?', 'B', 'C', 'D', 'A', 'B', 'C', '?', 'D', 'A', 'B', 'C', 'D', 'A', 'B', '?', 'C', 'D', 'A', 'B', 'C', 'D', 'A', '?', 'D', 'A', 'B', 'C', 'D', 'A', 'A', 'B', 'C', 'D', 'B', 'A', 'B', 'C', 'D', 'C', 'A', 'B', 'C', 'D', 'D', '?']

            Seja EXTREMAMENTE rigoroso - apenas bolinhas COMPLETAMENTE PRETAS.
            """
        else:  # resposta_aluno
            prompt = """
            Analise esta imagem de um cartão resposta de ALUNO e identifique APENAS as bolinhas PRETAS marcadas.

            INSTRUÇÕES ESPECÍFICAS:
            1. Este é um cartão com 52 questões organizadas em 4 colunas (1-13, 14-26, 27-39, 40-52)
            2. Cada questão tem 4 alternativas: A, B, C, D
            3. DETECTE APENAS bolinhas completamente PRETAS/PINTADAS pelo aluno
            4. IGNORE qualquer cor que não seja PRETA (correções do professor em verde, azul, etc.)
            5. IGNORE círculos vazios ou apenas contornados
            6. IGNORE rabiscos, riscos ou outras marcações
            7. FOQUE apenas em bolinhas SÓLIDAS PRETAS bem preenchidas

            FOCO: Apenas as marcações ORIGINAIS PRETAS do aluno.

            FORMATO DE RESPOSTA:
            Retorne apenas uma lista Python com 52 elementos, exemplo:
            ['A', 'B', '?', 'C', 'D', 'A', '?', 'B', 'C', 'D', 'A', 'B', 'C', '?', 'D', 'A', 'B', 'C', 'D', 'A', 'B', '?', 'C', 'D', 'A', 'B', 'C', 'D', 'A', '?', 'D', 'A', 'B', 'C', 'D', 'A', 'A', 'B', 'C', 'D', 'B', 'A', 'B', 'C', 'D', 'C', 'A', 'B', 'C', 'D', 'D', '?']

            Seja EXTREMAMENTE rigoroso - apenas bolinhas COMPLETAMENTE PRETAS do aluno.
            """
        
        # Fazer análise com Gemini
        print(f"🤖 Analisando {tipo_analise} com Gemini...")
        response = model.generate_content([prompt, image])
        
        # Extrair lista da resposta
        resposta_texto = response.text.strip()
        
        # Tentar extrair lista Python da resposta
        import ast
        try:
            # Procurar por lista no texto
            inicio = resposta_texto.find('[')
            fim = resposta_texto.rfind(']') + 1
            
            if inicio >= 0 and fim > inicio:
                lista_str = resposta_texto[inicio:fim]
                respostas = ast.literal_eval(lista_str)
                
                # Validar se tem 52 elementos
                if len(respostas) == 52:
                    print(f"✅ Gemini analisou {len(respostas)} questões!")
                    return respostas
                else:
                    print(f"⚠️ Gemini retornou {len(respostas)} questões, esperado 52")
                    # Ajustar para 52 elementos
                    while len(respostas) < 52:
                        respostas.append('?')
                    return respostas[:52]
            else:
                print("❌ Não foi possível extrair lista da resposta do Gemini")
                print(f"Resposta recebida: {resposta_texto}")
                return None
                
        except Exception as e:
            print(f"❌ Erro ao processar resposta do Gemini: {e}")
            print(f"Resposta recebida: {resposta_texto}")
            return None
            
    except Exception as e:
        print(f"❌ Erro na análise com Gemini: {e}")
        return None

def extrair_cabecalho_com_gemini(model, image_path):
    """
    Usa Gemini Vision para extrair informações do cabeçalho do cartão resposta
    """
    if not model:
        print("⚠️ Gemini não configurado, usando OCR")
        return None
        
    try:
        # Converter imagem
        image = converter_imagem_para_base64(image_path)
        if not image:
            return None
        
        # Prompt especializado para extrair dados do cabeçalho
        prompt = """
        Analise esta imagem de um cartão resposta e extraia APENAS as seguintes informações do cabeçalho:

        1. NOME DA ESCOLA - procure por campos como "Nome da Escola:", "Escola:", etc.
        2. NOME DO ALUNO - procure por campos como "Nome completo:", "Nome:", "Aluno:", etc.
        3. TURMA - procure por campos como "Turma:", "Série:", "Ano:", etc.
        4. DATA DE NASCIMENTO - procure por campos como "Data de nascimento:", "Nascimento:", etc.

        INSTRUÇÕES:
        - Extraia APENAS o conteúdo, SEM os rótulos (ex: se tem "Nome: João Silva", extraia apenas "João Silva")
        - Se alguma informação não estiver visível ou legível, retorne "N/A"
        - Seja preciso na leitura dos textos
        - Ignore títulos como "AVALIAÇÃO DIAGNÓSTICA", "CARTÃO-RESPOSTA", etc.

        FORMATO DE RESPOSTA (retorne exatamente neste formato JSON):
        {
            "escola": "nome da escola ou N/A",
            "aluno": "nome do aluno ou N/A", 
            "turma": "turma ou N/A",
            "nascimento": "data ou N/A"
        }
        """
        
        # Gerar resposta
        response = model.generate_content([prompt, image])
        resposta_texto = response.text.strip()
        
        print(f"🤖 GEMINI CABEÇALHO - Resposta bruta: {resposta_texto}")
        
        # Tentar extrair JSON da resposta
        try:
            import json
            import re
            
            # Procurar por JSON na resposta
            json_match = re.search(r'\{.*\}', resposta_texto, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                dados = json.loads(json_str)
                
                # Validar estrutura
                if all(key in dados for key in ['escola', 'aluno', 'turma', 'nascimento']):
                    print(f"✅ GEMINI extraiu dados do cabeçalho:")
                    print(f"   🏫 Escola: {dados['escola']}")
                    print(f"   👤 Aluno: {dados['aluno']}")
                    print(f"   📚 Turma: {dados['turma']}")
                    print(f"   📅 Nascimento: {dados['nascimento']}")
                    return dados
                else:
                    print("❌ JSON não tem todas as chaves necessárias")
                    return None
            else:
                print("❌ Não foi possível extrair JSON da resposta")
                return None
                
        except Exception as e:
            print(f"❌ Erro ao processar JSON do Gemini: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Erro na extração do cabeçalho com Gemini: {e}")
        return None

def extrair_cabecalho_com_ocr_fallback(image_path):
    """
    Função de fallback usando OCR tradicional quando Gemini falha
    """
    try:
        print("🔄 Usando OCR como fallback...")
        
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Erro ao carregar imagem: {image_path}")
            return None
            
        # Converter para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Pegar apenas a parte superior da imagem (cabeçalho)
        height = gray.shape[0]
        header_region = gray[0:int(height * 0.3)]  # 30% superior
        
        # Melhorar contraste para OCR
        header_region = cv2.threshold(header_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Extrair texto
        texto_completo = pytesseract.image_to_string(header_region, lang='por', config='--psm 6')
        
        # Processar texto extraído
        linhas = texto_completo.split('\n')
        dados = {
            "escola": "N/A",
            "aluno": "N/A", 
            "turma": "N/A",
            "nascimento": "N/A"
        }
        
        # Procurar padrões no texto
        for linha in linhas:
            linha = linha.strip()
            if not linha:
                continue
                
            linha_lower = linha.lower()
            
            # Procurar escola
            if any(palavra in linha_lower for palavra in ['escola', 'colégio', 'instituto', 'centro']):
                if 'escola' in linha_lower or 'colégio' in linha_lower:
                    dados["escola"] = linha
                    
            # Procurar nome do aluno  
            if any(palavra in linha_lower for palavra in ['nome', 'aluno']):
                # Pular se for apenas o rótulo
                if len(linha) > 10 and not linha_lower.startswith('nome'):
                    dados["aluno"] = linha
                    
            # Procurar turma
            if any(palavra in linha_lower for palavra in ['turma', 'série', 'ano']):
                # Extrair números da linha
                numeros = re.findall(r'\d+', linha)
                if numeros:
                    dados["turma"] = numeros[0]
                    
            # Procurar data de nascimento
            if any(palavra in linha_lower for palavra in ['nascimento', 'data']):
                # Procurar padrão de data
                data_match = re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', linha)
                if data_match:
                    dados["nascimento"] = data_match.group()
        
        print(f"✅ OCR extraiu dados básicos")
        return dados
        
    except Exception as e:
        print(f"❌ Erro no OCR fallback: {e}")
        return None

def extrair_cabecalho_com_fallback(model, image_path):
    """
    Função principal que tenta Gemini primeiro, depois OCR como fallback
    """
    # Tentar Gemini primeiro
    if model:
        try:
            dados_gemini = extrair_cabecalho_com_gemini(model, image_path)
            if dados_gemini:
                print("✅ Gemini extraiu dados com sucesso")
                return dados_gemini
            else:
                print("⚠️ Gemini falhou, tentando OCR...")
        except Exception as e:
            print(f"⚠️ Gemini com erro ({str(e)[:50]}...), usando OCR")
    
    # Fallback para OCR
    dados_ocr = extrair_cabecalho_com_ocr_fallback(image_path)
    if dados_ocr:
        return dados_ocr
    
    # Se tudo falhar, retornar dados vazios
    print("❌ Ambos Gemini e OCR falharam")
    return {
        "escola": "N/A",
        "aluno": "N/A", 
        "turma": "N/A",
        "nascimento": "N/A"
    }

def comparar_omr_vs_gemini(respostas_omr, respostas_gemini, tipo=""):
    """Compara resultados OMR vs Gemini e gera relatório"""
    if not respostas_gemini:
        print("⚠️ Gemini não disponível, usando apenas OMR")
        return respostas_omr
    
    diferencas = []
    concordancias = 0
    
    print(f"\n🔍 COMPARAÇÃO OMR vs GEMINI ({tipo}):")
    print("Questão | OMR | Gemini | Status")
    print("-" * 32)
    
    for i in range(min(len(respostas_omr), len(respostas_gemini))):
        omr = respostas_omr[i]
        gemini = respostas_gemini[i]
        
        if omr == gemini:
            status = "✅"
            concordancias += 1
        else:
            status = "⚠️"
            diferencas.append({
                'questao': i + 1,
                'omr': omr,
                'gemini': gemini
            })
        
        print(f"   {i+1:02d}   | {omr:^3} | {gemini:^6} | {status}")
    
    total = len(respostas_omr)
    percentual_concordancia = (concordancias / total * 100) if total > 0 else 0
    
    print(f"\n📊 ESTATÍSTICAS:")
    print(f"Concordâncias: {concordancias}/{total} ({percentual_concordancia:.1f}%)")
    print(f"Diferenças: {len(diferencas)}")
    
    # Decidir qual usar baseado na concordância
    if percentual_concordancia >= 80:
        print("✅ Alta concordância - usando resultado OMR")
        return respostas_omr
    elif percentual_concordancia >= 50:
        print("⚠️ Concordância média - usando híbrido OMR/Gemini")
        # Criar versão híbrida (usar Gemini quando OMR detecta '?')
        resultado_hibrido = []
        for i in range(len(respostas_omr)):
            if respostas_omr[i] == '?' and i < len(respostas_gemini):
                resultado_hibrido.append(respostas_gemini[i])
            else:
                resultado_hibrido.append(respostas_omr[i])
        return resultado_hibrido
    else:
        print("🤖 Baixa concordância - usando resultado Gemini")
        return respostas_gemini

# ===========================================
# SEÇÃO 4: INTEGRAÇÃO GOOGLE DRIVE & SHEETS
# ===========================================

def carregar_credenciais(scopes: List[str]) -> Optional[Credentials]:
    """Carrega credenciais do serviço do arquivo JSON."""
    try:
        credentials = Credentials.from_service_account_file('credenciais_google.json', scopes=scopes)
        return credentials
    except FileNotFoundError:
        print("❌ Arquivo 'credenciais_google.json' não encontrado!")
        print("📝 Certifique-se de que o arquivo está no diretório atual")
        return None
    except Exception as e:
        print(f"❌ Erro ao carregar credenciais: {e}")
        return None


def configurar_google_sheets():
    """Configura conexão com Google Sheets"""
    scope = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    credentials = carregar_credenciais(scope)
    if not credentials:
        return None

    try:
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        print(f"❌ Erro ao conectar com Google Sheets: {e}")
        return None


def configurar_google_drive_service(scopes: Optional[List[str]] = None):
    """Configura conexão com Google Drive e retorna serviço da API."""
    scopes = scopes or ['https://www.googleapis.com/auth/drive.readonly']
    credentials = carregar_credenciais(scopes)
    if not credentials:
        return None

    try:
        service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
        return service
    except HttpError as http_err:
        print(f"❌ Erro HTTP ao conectar no Google Drive: {http_err}")
    except Exception as e:
        print(f"❌ Erro ao configurar Google Drive: {e}")
    return None

def configurar_google_drive_service_completo():
    """Configura conexão com Google Drive com permissões completas para mover arquivos."""
    scopes = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file'
    ]
    credentials = carregar_credenciais(scopes)
    if not credentials:
        return None

    try:
        service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
        return service
    except HttpError as http_err:
        print(f"❌ Erro HTTP ao conectar no Google Drive: {http_err}")
    except Exception as e:
        print(f"❌ Erro ao configurar Google Drive: {e}")
    return None

def encontrar_ou_criar_pasta_processados(service, pasta_origem_id: str) -> str:
    """Usa a pasta 'cartoes-processados' específica no Google Drive."""
    pasta_processados_id = "1fVFfewF2qUe-wgORQ5p15on5apOQ2G_i"
    
    try:
        # Verificar se a pasta existe e é acessível
        pasta_info = service.files().get(fileId=pasta_processados_id, fields='id, name').execute()
        print(f"📁 Pasta encontrada: {pasta_info.get('name')}")
        return pasta_processados_id
        
    except Exception as e:
        print(f"❌ Erro ao acessar pasta 'cartoes-processados' (ID: {pasta_processados_id}): {e}")
        print("   Verifique se a pasta existe e o ID está correto")
        return None

def mover_arquivo_no_drive(service, arquivo_id: str, pasta_origem_id: str, pasta_destino_id: str, nome_arquivo: str) -> bool:
    """Move um arquivo de uma pasta para outra no Google Drive."""
    try:
        # Obter pais atuais do arquivo
        file_metadata = service.files().get(fileId=arquivo_id, fields='parents').execute()
        previous_parents = ",".join(file_metadata.get('parents'))
        
        # Mover arquivo (remover da pasta origem e adicionar à pasta destino)
        service.files().update(
            fileId=arquivo_id,
            addParents=pasta_destino_id,
            removeParents=previous_parents,
            fields='id, parents'
        ).execute()
        return True
        
    except Exception as e:
        print(f"❌ Erro ao mover arquivo {nome_arquivo}: {e}")
        return False

def obter_metadados_pasta_drive(service, pasta_id: str) -> dict:
    """Obtém metadados de todos os arquivos da pasta do Google Drive."""
    metadados = {}
    try:
        query = f"parents in '{pasta_id}' and trashed=false"
        campos = "nextPageToken, files(id, name, mimeType, size, modifiedTime)"
        
        page_token = None
        while True:
            response = service.files().list(
                q=query,
                fields=campos,
                pageToken=page_token
            ).execute()
            
            arquivos = response.get('files', [])
            for arquivo in arquivos:
                nome = arquivo.get('name', '')
                if nome.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf')):
                    metadados[nome] = {
                        'id': arquivo['id'],
                        'nome': nome,
                        'mime_type': arquivo.get('mimeType', ''),
                        'tamanho': arquivo.get('size', '0'),
                        'modificado': arquivo.get('modifiedTime', '')
                    }
            
            page_token = response.get('nextPageToken')
            if not page_token:
                break
                
        print(f"📋 Metadados obtidos para {len(metadados)} arquivos")
        return metadados
        
    except Exception as e:
        print(f"❌ Erro ao obter metadados: {e}")
        return {}

def mover_arquivos_processados_drive(service, pasta_origem_id: str, metadados: dict):
    """Move arquivos processados (exceto gabarito) para pasta 'cartoes-processados'."""
    try:
        # Configurar serviço com permissões completas
        service_completo = configurar_google_drive_service_completo()
        if not service_completo:
            print("❌ Não foi possível obter permissões para mover arquivos")
            return
        
        # Encontrar ou criar pasta de processados
        pasta_processados_id = encontrar_ou_criar_pasta_processados(service_completo, pasta_origem_id)
        if not pasta_processados_id:
            print("❌ Não foi possível criar pasta 'cartoes-processados'")
            return
        
        arquivos_movidos = 0
        
        # Mover todos os arquivos exceto o gabarito
        for nome_arquivo, dados in metadados.items():
            # Pular arquivo de gabarito
            if nome_arquivo.lower().startswith('gabarito'):
                continue
            
            # Mover arquivo
            if mover_arquivo_no_drive(
                service_completo, 
                dados['id'], 
                pasta_origem_id, 
                pasta_processados_id, 
                nome_arquivo
            ):
                arquivos_movidos += 1
        
        print(f"✅ {arquivos_movidos} arquivos movidos para 'cartoes-processados' no Drive")
        
    except Exception as e:
        print(f"❌ Erro ao mover arquivos processados: {e}")


def sanitizar_nome_arquivo(nome: str, extensao_padrao: str = "") -> str:
    """Remove caracteres inválidos e garante extensão válida."""
    nome_limpo = re.sub(r'[<>:"/\\|?*]+', '_', nome).strip()
    if not nome_limpo:
        nome_limpo = 'arquivo'
    if extensao_padrao and not nome_limpo.lower().endswith(extensao_padrao.lower()):
        nome_limpo += extensao_padrao
    return nome_limpo


def baixar_cartoes_da_pasta_drive(service, pasta_id: str, destino: str, formatos_validos: Optional[Dict[str, str]] = None) -> List[str]:
    """Baixa todos os cartões (gabarito + alunos) de uma pasta do Google Drive."""
    if not service:
        print("❌ Serviço do Google Drive não configurado.")
        return []

    if not pasta_id:
        print("❌ ID da pasta do Google Drive não informado.")
        return []

    os.makedirs(destino, exist_ok=True)
    formatos_validos = formatos_validos or DRIVE_MIME_TO_EXT
    arquivos_baixados: List[str] = []

    query = f"'{pasta_id}' in parents and trashed = false"
    campos = "nextPageToken, files(id, name, mimeType, modifiedTime)"
    page_token = None


    try:
        while True:
            response = service.files().list(
                q=query,
                fields=campos,
                pageToken=page_token
            ).execute()

            arquivos = response.get('files', [])
            if not arquivos:
                break

            for arquivo in arquivos:
                mime_type = arquivo.get('mimeType', '')
                nome_original = arquivo.get('name', 'arquivo')
                extensao_padrao = formatos_validos.get(mime_type, '')

                base, ext = os.path.splitext(nome_original)
                ext_final = ext.lower() if ext else extensao_padrao

                if not ext_final:
                    continue

                if ext_final and ext_final.lower() not in EXTENSOES_SUPORTADAS and ext_final not in formatos_validos.values():
                    print(f"⚠️ Ignorando '{nome_original}' (tipo não suportado: {mime_type})")
                    continue

                nome_final = sanitizar_nome_arquivo(nome_original, extensao_padrao=ext_final)
                caminho_destino = os.path.join(destino, nome_final)

                # Resolver conflitos de nome
                contador = 1
                while os.path.exists(caminho_destino):
                    nome_sem_ext, ext_arquivo = os.path.splitext(nome_final)
                    caminho_destino = os.path.join(destino, f"{nome_sem_ext}_{contador}{ext_arquivo}")
                    contador += 1

                print(f"⬇️ Baixando: {nome_final}")
                request = service.files().get_media(fileId=arquivo['id'])
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        progresso = int(status.progress() * 100)
                        print(f"   Progresso: {progresso}%", end='\r')

                with open(caminho_destino, 'wb') as destino_arquivo:
                    destino_arquivo.write(fh.getbuffer())

                arquivos_baixados.append(caminho_destino)

            page_token = response.get('nextPageToken')
            if not page_token:
                break

    except HttpError as http_err:
        print(f"❌ Erro HTTP ao baixar arquivos do Drive: {http_err}")
        return []
    except Exception as e:
        print(f"❌ Erro inesperado ao baixar arquivos do Drive: {e}")
        return []

    print(f"✅ Download concluído: {len(arquivos_baixados)} arquivos salvos em {destino}")
    return arquivos_baixados


def baixar_e_processar_pasta_drive(
    pasta_id: str,
    usar_gemini: bool = True,
    debug_mode: bool = False,
    enviar_para_sheets: bool = True,
    manter_pasta_temporaria: bool = False,
    mover_processados: bool = True,
    apenas_gabarito: bool = False
):
    """Workflow completo: baixa do Drive, processa cartões e envia resultados."""

    service = configurar_google_drive_service()
    if not service:
        print("❌ Não foi possível configurar o Google Drive. Abortando.")
        return []

    pasta_temporaria = tempfile.mkdtemp(prefix="cartoes_drive_")
    print(f"📁 Pasta temporária criada: {pasta_temporaria}")

    try:
        # Obter metadados dos arquivos durante o download
        arquivos_metadata = obter_metadados_pasta_drive(service, pasta_id)
        arquivos_baixados = baixar_cartoes_da_pasta_drive(service, pasta_id, pasta_temporaria)
        
        if not arquivos_baixados:
            print("❌ Nenhum arquivo válido foi baixado do Drive.")
            return []

        # Se é apenas para gabarito, retornar o diretório temporário
        if apenas_gabarito:
            return pasta_temporaria

        if enviar_para_sheets:
            resultados = processar_pasta_gabaritos_com_sheets(
                pasta_temporaria,
                usar_gemini=usar_gemini,
                debug_mode=debug_mode
            )
        else:
            resultados = processar_pasta_gabaritos_sem_sheets(
                pasta_temporaria,
                usar_gemini=usar_gemini,
                debug_mode=debug_mode
            )

        # Mover arquivos processados se houve sucesso e está habilitado
        if resultados and mover_processados:
            print(f"\n📦 Movendo arquivos processados no Google Drive...")
            mover_arquivos_processados_drive(service, pasta_id, arquivos_metadata)

        return resultados

    finally:
        if manter_pasta_temporaria:
            print(f"🗂️ Mantendo pasta temporária em: {pasta_temporaria}")
        else:
            shutil.rmtree(pasta_temporaria, ignore_errors=True)

def enviar_para_planilha(client, dados_aluno, resultado_comparacao, planilha_id=None):
    """Envia dados para Google Sheets"""
    try:
        if planilha_id:
            sheet = client.open_by_key(planilha_id)
            pass
        else:
            planilhas = client.list_spreadsheet_files()
            print(f"📊 Você tem {len(planilhas)} planilhas no Drive")
            
            nome_planilha = "Correção Cartão Resposta"
            try:
                sheet = client.open(nome_planilha)
                print(f"✅ Planilha '{nome_planilha}' encontrada!")
            except gspread.SpreadsheetNotFound:
                print(f"📄 Criando nova planilha '{nome_planilha}'...")
                sheet = client.create(nome_planilha)
        
        # Usar primeira aba
        worksheet = sheet.sheet1
        
        # Verificar se há cabeçalho
        if not worksheet.get_all_values():
            cabecalho = [
                "Data", "Escola", "Nome completo", "Nascimento", "Turma", "Acertos", "Erros", "Porcentagem"
            ]
            worksheet.append_row(cabecalho)
            print("📋 Cabeçalho criado na planilha")
        
        # Preparar dados completos
        agora = datetime.now().strftime("%d/%m/%Y")
        
        # Garantir que os dados estejam no formato correto
        escola = dados_aluno.get("Escola", "N/A")
        if escola == "N/A" or not escola.strip():
            escola = "N/A"
        
        aluno = dados_aluno.get("Aluno", "N/A")
        if aluno == "N/A" or not aluno.strip():
            aluno = "N/A"
            
        nascimento = dados_aluno.get("Nascimento", "N/A")
        if nascimento == "N/A" or not nascimento.strip():
            nascimento = "N/A"
            
        turma = dados_aluno.get("Turma", "N/A")
        if turma == "N/A" or not turma.strip():
            turma = "N/A"
        
        linha_dados = [
            agora,
            escola,
            aluno, 
            nascimento,
            turma,
            resultado_comparacao["acertos"],
            resultado_comparacao["erros"],
            f"{resultado_comparacao['percentual']:.1f}%"
        ]
        
        # Adicionar linha
        worksheet.append_row(linha_dados)
        print(f"📊 Registro adicionado:")
        print(f"   🏫 Escola: {escola}")
        print(f"   👤 Aluno: {aluno}")
        print(f"   📅 Nascimento: {nascimento}")
        print(f"   📚 Turma: {turma}")
        print(f"   📊 Resultado: {resultado_comparacao['acertos']} acertos | {resultado_comparacao['erros']} erros | {resultado_comparacao['percentual']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao enviar dados para Google Sheets: {e}")
        return False

def criar_planilha_detalhada(client, dados_aluno, resultado_comparacao):
    """Cria aba detalhada com todas as questões"""
    try:
        # Abrir planilha existente
        PLANILHA_ID = "1VJ0_w9eoQcc-ouBnRoq5lFQdR2fVZkqEtR-KArZMuvk"
        sheet = client.open_by_key(PLANILHA_ID)
        
        # Nome da nova aba
        nome_aba = f"Detalhes_{dados_aluno['Aluno'].replace(' ', '_')[:15]}_{datetime.now().strftime('%d%m_%H%M')}"
        
        # Criar nova aba
        worksheet = sheet.add_worksheet(title=nome_aba, rows=35, cols=6)
        
        # Cabeçalho da aba detalhada
        cabecalho_detalhado = [
            "Questão", "Gabarito", "Resposta Aluno", "Status", "Resultado", "Observação"
        ]
        worksheet.append_row(cabecalho_detalhado)
        
        # Dados detalhados
        for detalhe in resultado_comparacao["detalhes"]:
            linha = [
                detalhe["questao"],
                detalhe["gabarito"],
                detalhe["aluno"],
                detalhe["status"],
                "ACERTO" if detalhe["status"] == "✓" else "ERRO",
                "" if detalhe["status"] == "✓" else f"Esperado: {detalhe['gabarito']}, Marcado: {detalhe['aluno']}"
            ]
            worksheet.append_row(linha)
        
        print(f"✅ Planilha detalhada '{nome_aba}' criada com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao criar planilha detalhada: {e}")
        return False

def comparar_respostas(respostas_gabarito, respostas_aluno):
    """Compara as respostas do gabarito com as do aluno"""
    if len(respostas_gabarito) != len(respostas_aluno):
        print("⚠️  ATENÇÃO: Número de questões diferentes entre gabarito e resposta do aluno!")
        min_questoes = min(len(respostas_gabarito), len(respostas_aluno))
    else:
        min_questoes = len(respostas_gabarito)
    
    acertos = 0
    erros = 0
    detalhes = []
    
    for i in range(min_questoes):
        questao = i + 1
        gabarito = respostas_gabarito[i] if i < len(respostas_gabarito) else "N/A"
        aluno = respostas_aluno[i] if i < len(respostas_aluno) else "N/A"
        
        if gabarito == aluno:
            status = "✓"
            acertos += 1
        else:
            status = "✗"
            erros += 1
        
        detalhes.append({
            "questao": questao,
            "gabarito": gabarito,
            "aluno": aluno,
            "status": status
        })
    
    percentual = (acertos / min_questoes * 100) if min_questoes > 0 else 0
    
    return {
        "total": min_questoes,
        "acertos": acertos,
        "erros": erros,
        "percentual": percentual,
        "detalhes": detalhes,
        "questoes_detectadas": min_questoes
    }

def exibir_resultados(dados_aluno, resultado):
    """Exibe os resultados formatados"""
    print("\n" + "="*50)
    print("         CORREÇÃO DO CARTÃO RESPOSTA")
    print("="*50)
    
    print("\n=== DADOS DO ALUNO ===")
    for campo, valor in dados_aluno.items():
        print(f"{campo}: {valor}")
    
    print("\n=== RESULTADO GERAL ===")
    print(f"Total de questões: {resultado['total']}")
    print(f"Acertos: {resultado['acertos']} ✓")
    print(f"Erros: {resultado['erros']} ✗")
    print(f"Percentual de acerto: {resultado['percentual']:.2f}%")
    
    print("\n=== DETALHAMENTO POR QUESTÃO ===")
    print("Questão | Gabarito | Aluno | Status")
    print("-" * 35)
    
    for detalhe in resultado["detalhes"]:
        print(f"   {detalhe['questao']:02d}   |    {detalhe['gabarito']}     |   {detalhe['aluno']}   |   {detalhe['status']}")
    
    # Mostrar apenas questões erradas
    erros_detalhados = [d for d in resultado["detalhes"] if d["status"] == "✗"]
    if erros_detalhados:
        print("\n=== QUESTÕES ERRADAS ===")
        for erro in erros_detalhados:
            print(f"Questão {erro['questao']:02d}: Gabarito {erro['gabarito']} ≠ Aluno {erro['aluno']} ✗")

def exibir_gabarito_simples(respostas_gabarito):
    """Exibe o gabarito em formato simples: 1-A, 2-B, 3-C"""
    print("\n📋 GABARITO DAS QUESTÕES:")
    print("=" * 30)
    
    # Agrupar as questões em linhas de 10 para melhor visualização
    for i in range(0, len(respostas_gabarito), 10):
        linha = []
        for j in range(i, min(i + 10, len(respostas_gabarito))):
            if respostas_gabarito[j] != '?':
                linha.append(f"{j+1}-{respostas_gabarito[j]}")
            else:
                linha.append(f"{j+1}-?")
        print("  ".join(linha))
    
    print("=" * 30)

def processar_apenas_gabarito(drive_folder_id: str = "13KIDX3GtQWxIxlAsX-2XS0ypJvOnnqZX", debug_mode: bool = False):
    """Processa apenas o gabarito e exibe as respostas em formato simples"""
    print("📋 PROCESSANDO APENAS GABARITO")
    print("=" * 40)
    
    try:
        # Baixar arquivos do Google Drive
        print(f"📥 Baixando arquivos da pasta do Drive: {drive_folder_id}")
        diretorio_temp = baixar_e_processar_pasta_drive(
            pasta_id=drive_folder_id,
            usar_gemini=False,
            debug_mode=debug_mode,
            enviar_para_sheets=False,
            manter_pasta_temporaria=True,
            mover_processados=False,
            apenas_gabarito=True
        )
        
        if not diretorio_temp:
            print("❌ Erro ao baixar arquivos do Drive")
            return
        
        # Procurar arquivo de gabarito
        arquivos = [f for f in os.listdir(diretorio_temp) 
                    if f.lower().endswith(EXTENSOES_SUPORTADAS)]
        
        gabarito_file = None
        for arquivo in arquivos:
            if 'gabarito' in arquivo.lower():
                gabarito_file = arquivo
                break
        
        if not gabarito_file:
            print("❌ Arquivo de gabarito não encontrado (deve conter 'gabarito' no nome)")
            return
        
        print(f"📋 Gabarito encontrado: {gabarito_file}")
        
        # Preprocessar gabarito
        gabarito_path = os.path.join(diretorio_temp, gabarito_file)
        gabarito_img = preprocessar_arquivo(gabarito_path, "gabarito")
        
        # Detectar respostas do gabarito
        if "page_" in gabarito_img and (gabarito_img.endswith(".png") or gabarito_img.endswith(".jpg")):
            respostas_gabarito = detectar_respostas_pdf(gabarito_img, debug=debug_mode)
        else:
            respostas_gabarito = detectar_respostas(gabarito_img, debug=debug_mode)
        
        questoes_gabarito = sum(1 for r in respostas_gabarito if r != '?')
        print(f"✅ Gabarito processado: {questoes_gabarito}/52 questões detectadas")
        
        # Exibir gabarito em formato simples
        exibir_gabarito_simples(respostas_gabarito)
        
        if questoes_gabarito < 40:
            print("⚠️ ATENÇÃO: Poucas questões detectadas no gabarito.")
        
        # Limpar arquivos temporários
        if os.path.exists(diretorio_temp):
            shutil.rmtree(diretorio_temp, ignore_errors=True)
        
    except Exception as e:
        print(f"❌ Erro ao processar gabarito: {e}")

# ===========================================
# PROCESSAMENTO EM LOTE
# ===========================================

def processar_pasta_gabaritos(diretorio: str = "./gabaritos", usar_gemini: bool = True, debug_mode: bool = False):
    """
    Processa todos os arquivos de uma pasta com cartões (gabarito + alunos)
    - 1 gabarito (template) para comparar com múltiplos alunos
    - Sem comparações desnecessárias de dados
    
    Args:
        diretorio: Caminho da pasta contendo gabarito e cartões dos alunos
        usar_gemini: Se deve usar Gemini para cabeçalho
        debug_mode: Se deve mostrar debug detalhado
        
    Returns:
        Lista de resultados de cada aluno processado
    """
    
    print("🚀 SISTEMA DE CORREÇÃO - PASTA GABARITOS")
    print("=" * 60)
    
    diretorio_gabaritos = diretorio
    
    if not os.path.exists(diretorio_gabaritos):
        print(f"❌ ERRO: Pasta '{diretorio_gabaritos}' não encontrada!")
        print("💡 Crie a pasta informada e adicione os arquivos do gabarito e dos alunos")
        return []
    
    # Configurar suporte a PDF se disponível
    if PDF_PROCESSOR_AVAILABLE:
        print("\n🔧 Configurando suporte a PDF...")
        pdf_ok = setup_pdf_support()
        if not pdf_ok:
            print("⚠️ Suporte a PDF limitado - apenas imagens serão processadas")
    
    # Listar arquivos suportados na pasta gabaritos
    print(f"\n📁 Analisando arquivos na pasta: {os.path.abspath(diretorio_gabaritos)}")
    arquivos = listar_arquivos_suportados(diretorio_gabaritos)
    
    if not arquivos['todos']:
        print("❌ Nenhum arquivo suportado encontrado na pasta gabaritos!")
        print("💡 Formatos suportados: PDF, PNG, JPG, JPEG, BMP, TIFF")
        return []
    
    print(f"✅ Encontrados {len(arquivos['todos'])} arquivos:")
    for arquivo in arquivos['todos']:
        print(f"   📄 {arquivo}")
    
    # ===========================================
    # IDENTIFICAR GABARITO (LÓGICA SIMPLIFICADA)
    # ===========================================
    
    print("\n📋 Identificando arquivo de gabarito...")
    gabarito_file = None
    
    # Buscar por qualquer arquivo que comece com "gabarito" (case insensitive)
    for arquivo in arquivos['todos']:
        if arquivo.lower().startswith('gabarito'):
            gabarito_file = arquivo
            break
    
    if not gabarito_file:
        print("❌ ERRO: Nenhum arquivo 'gabarito.*' encontrado!")
        print("💡 Renomeie o arquivo do gabarito para: gabarito.png, gabarito.pdf, etc.")
        return []
    
    print(f"✅ Gabarito identificado: {gabarito_file}")
    
    # ===========================================
    # IDENTIFICAR ARQUIVOS DOS ALUNOS (LÓGICA SIMPLIFICADA)
    # ===========================================
    
    print("\n👥 Identificando arquivos dos alunos...")
    
    # TODOS os arquivos que NÃO começam com "gabarito" são alunos
    arquivos_alunos = [f for f in arquivos['todos'] if not f.lower().startswith('gabarito')]
    
    if not arquivos_alunos:
        print("❌ ERRO: Nenhum arquivo de aluno encontrado!")
        print("💡 Adicione arquivos dos alunos na pasta gabaritos (qualquer nome exceto gabarito.*)")
        return []
    
    print(f"✅ Encontrados {len(arquivos_alunos)} alunos para processar:")
    for i, aluno in enumerate(arquivos_alunos, 1):
        print(f"   {i:02d}. {aluno}")
    
    # ===========================================
    # CONFIGURAR GEMINI
    # ===========================================
    
    model_gemini = None
    if usar_gemini:
        try:
            model_gemini = configurar_gemini()
        except Exception as e:
            print(f"❌ Erro ao configurar Gemini: {e}")
            usar_gemini = False
    
    # ===========================================
    # PROCESSAR GABARITO (UMA VEZ APENAS)
    # ===========================================
    
    print(f"\n{'='*60}")
    print("📋 PROCESSANDO GABARITO")
    print(f"{'='*60}")
    
    try:
        # Preprocessar gabarito
        gabarito_path = os.path.join(diretorio_gabaritos, gabarito_file)
        gabarito_img = preprocessar_arquivo(gabarito_path, "gabarito")
        
        # Detectar respostas do gabarito
        if "page_" in gabarito_img and (gabarito_img.endswith(".png") or gabarito_img.endswith(".jpg")):
            respostas_gabarito = detectar_respostas_pdf(gabarito_img, debug=debug_mode)
        else:
            respostas_gabarito = detectar_respostas(gabarito_img, debug=debug_mode)
        
        questoes_gabarito = sum(1 for r in respostas_gabarito if r != '?')
        print(f"✅ Gabarito processado: {questoes_gabarito}/52 questões detectadas")
        
        if questoes_gabarito < 40:
            print("⚠️ ATENÇÃO: Poucas questões detectadas no gabarito.")
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO ao processar gabarito: {e}")
        return []
    
    # ===========================================
    # PROCESSAR TODOS OS ALUNOS
    # ===========================================
    
    resultados_lote = []
    
    print(f"\n{'='*60}")
    print(f"👥 PROCESSANDO {len(arquivos_alunos)} ALUNOS")
    print(f"{'='*60}")
    
    for i, aluno_file in enumerate(arquivos_alunos, 1):
        print(f"\n🔄 [{i:02d}/{len(arquivos_alunos)}] Processando: {aluno_file}")
        print("-" * 50)
        
        try:
            # Preprocessar arquivo do aluno
            aluno_path = os.path.join(diretorio_gabaritos, aluno_file)
            aluno_img = preprocessar_arquivo(aluno_path, f"aluno_{i}")
            
            # Extrair dados do cabeçalho (opcional com Gemini)
            dados_aluno = {
                "Aluno": os.path.splitext(aluno_file)[0],
                "Escola": "N/A",
                "Nascimento": "N/A", 
                "Turma": "N/A"
            }
            
            if usar_gemini and model_gemini:
                try:
                    dados_extraidos = extrair_cabecalho_com_fallback(model_gemini, aluno_img)
                    if dados_extraidos:
                        # Mapear chaves minúsculas do Gemini para maiúsculas do sistema
                        mapeamento = {
                            "escola": "Escola",
                            "aluno": "Aluno", 
                            "turma": "Turma",
                            "nascimento": "Nascimento"
                        }
                        
                        # Atualizar dados com mapeamento correto
                        for chave_gemini, chave_sistema in mapeamento.items():
                            if chave_gemini in dados_extraidos and dados_extraidos[chave_gemini] and dados_extraidos[chave_gemini] != "N/A":
                                dados_aluno[chave_sistema] = dados_extraidos[chave_gemini]
                        
                        print(f"✅ Dados extraídos: {dados_aluno['Aluno']} ({dados_aluno['Escola']})")
                except Exception as e:
                    print(f"⚠️ Gemini falhou, usando nome do arquivo")
            
            # Detectar respostas do aluno
            if "page_" in aluno_img and (aluno_img.endswith(".png") or aluno_img.endswith(".jpg")):
                respostas_aluno = detectar_respostas_pdf(aluno_img, debug=debug_mode)
            else:
                respostas_aluno = detectar_respostas(aluno_img, debug=debug_mode)
            
            questoes_aluno = sum(1 for r in respostas_aluno if r != '?')
            
            # Calcular resultado
            resultado = comparar_respostas(respostas_gabarito, respostas_aluno)
            
            # Armazenar resultado com dados completos
            resultado_completo = {
                "arquivo": aluno_file,
                "dados_completos": dados_aluno,  # Dados completos do cabeçalho
                "acertos": resultado['acertos'],
                "total": resultado['total'],
                "percentual": resultado['percentual'],
                "questoes_detectadas": questoes_aluno
            }
            resultados_lote.append(resultado_completo)
            
            print(f"📊 Resultado: {resultado['acertos']}/{resultado['total']} acertos ({resultado['percentual']:.1f}%)")
            
        except Exception as e:
            print(f"❌ ERRO ao processar {aluno_file}: {e}")
            resultado_erro = {
                "arquivo": aluno_file,
                "dados_completos": {
                    "Aluno": os.path.splitext(aluno_file)[0],
                    "Escola": "N/A",
                    "Nascimento": "N/A",
                    "Turma": "N/A"
                },
                "acertos": 0,
                "total": 52,
                "percentual": 0.0,
                "questoes_detectadas": 0,
                "erro": str(e)
            }
            resultados_lote.append(resultado_erro)
    
    # ===========================================
    # RELATÓRIO FINAL SIMPLIFICADO
    # ===========================================
    
    print(f"\n{'='*60}")
    print("📊 RELATÓRIO FINAL")
    print(f"{'='*60}")
    
    if resultados_lote:
        print(f"\n=== TOTAL DE ALUNOS: {len(resultados)} + RESULTADOS ===")
        
        # Ordenar por percentual (decrescente)
        resultados_ordenados = sorted(resultados_lote, key=lambda x: x["percentual"], reverse=True)
        
        for i, r in enumerate(resultados_ordenados, 1):
            dados = r["dados_completos"]
            nome = dados["Aluno"]
            escola = dados["Escola"]
            nascimento = dados["Nascimento"]
            turma = dados["Turma"]
            acertos = r["acertos"]
            
            status = "❌" if "erro" in r else "✅"
            
            # Formato: aluno X (nome completo, escola, nascimento, turma) - acertou Y questões
            print(f"{status} aluno {i} ({nome}, {escola}, {nascimento}, {turma}) - acertou {acertos} questões")
        
        # Estatísticas
        resultados_validos = [r for r in resultados_lote if "erro" not in r]
        if resultados_validos:
            acertos = [r["acertos"] for r in resultados_validos]
            percentuais = [r["percentual"] for r in resultados_validos]
            
            print(f"\n=== ESTATÍSTICAS ===")
            print(f"Média de acertos: {sum(acertos)/len(acertos):.1f}/52 questões")
            print(f"Média percentual: {sum(percentuais)/len(percentuais):.1f}%")
    
    # ===========================================
    # ENVIAR PARA GOOGLE SHEETS (OPCIONAL)
    # ===========================================
    
    print(f"\n📤 Enviando resultados para Google Sheets...")
    try:
        client = configurar_google_sheets()
        if client:
            sucessos = 0
            for resultado in resultados_lote:
                if "erro" not in resultado:
                    try:
                        dados_completos = resultado["dados_completos"]
                        dados_simples = {
                            "Aluno": dados_completos["Aluno"], 
                            "Escola": dados_completos["Escola"],
                            "Nascimento": dados_completos["Nascimento"],
                            "Turma": dados_completos["Turma"]
                        }
                        resultado_comparacao = {
                            "acertos": resultado["acertos"],
                            "erros": resultado["total"] - resultado["acertos"],
                            "percentual": resultado["percentual"]
                        }
                        enviar_para_planilha(client, dados_simples, resultado_comparacao)
                        sucessos += 1
                    except Exception as e:
                        print(f"⚠️ Erro ao enviar {dados_completos['Aluno']}: {e}")
            print(f"✅ {sucessos}/{len(resultados_lote)} resultados enviados!")
        else:
            print("❌ Não foi possível conectar ao Google Sheets")
    except Exception as e:
        print(f"⚠️ Erro ao enviar para Sheets: {e}")
    
    return resultados_lote

def processar_lote_alunos(diretorio=".", usar_gemini=True, debug_mode=False):
    """
    Processa múltiplos cartões de alunos em lote
    
    Args:
        diretorio: Diretório contendo os arquivos
        usar_gemini: Se deve usar Gemini para cabeçalho
        debug_mode: Se deve mostrar debug detalhado
        
    Returns:
        Lista de resultados de cada aluno processado
    """
    
    print("🚀 SISTEMA DE CORREÇÃO EM LOTE - MÚLTIPLOS ALUNOS")
    print("=" * 60)
    
    # Configurar suporte a PDF se disponível
    if PDF_PROCESSOR_AVAILABLE:
        print("\n🔧 Configurando suporte a PDF...")
        pdf_ok = setup_pdf_support()
        if not pdf_ok:
            print("⚠️ Suporte a PDF limitado - apenas imagens serão processadas")
    
    # Listar arquivos suportados no diretório
    print(f"\n📁 Analisando arquivos no diretório: {os.path.abspath(diretorio)}")
    arquivos = listar_arquivos_suportados(diretorio)
    
    if not arquivos['todos']:
        print("❌ Nenhum arquivo suportado encontrado!")
        print("💡 Formatos suportados: PDF, PNG, JPG, JPEG, BMP, TIFF")
        return []
    
    print(f"✅ Encontrados {len(arquivos['todos'])} arquivos suportados:")
    if arquivos['imagens']:
        print(f"   🖼️ Imagens: {', '.join(arquivos['imagens'])}")
    if arquivos['pdfs']:
        print(f"   📄 PDFs: {', '.join(arquivos['pdfs'])}")
    
    # ===========================================
    # IDENTIFICAR GABARITO
    # ===========================================
    
    print("\n📋 Identificando arquivo de gabarito...")
    gabarito_file = None
    
    # Buscar arquivo do gabarito (priorizar nomes específicos)
    gabarito_candidates = [
        "gabarito.pdf", "gabarito.png", "gabarito.jpg",
        "resposta_gabarito.pdf", "resposta_gabarito.png", "resposta_gabarito.jpg",
        "resposta_gabarito_teste.pdf", "master.pdf", "template.pdf"
    ]
    
    for candidate in gabarito_candidates:
        if candidate in arquivos['todos']:
            gabarito_file = candidate
            break
    
    # Se não encontrou pelos nomes, usar o primeiro arquivo (assumindo que o primeiro é gabarito)
    if not gabarito_file and arquivos['todos']:
        gabarito_file = sorted(arquivos['todos'])[0]
        print(f"⚠️ Gabarito não identificado pelos nomes padrão. Usando: {gabarito_file}")
    
    if not gabarito_file:
        print("❌ ERRO: Nenhum arquivo de gabarito encontrado!")
        print("💡 Renomeie o arquivo do gabarito para: gabarito.pdf ou gabarito.png")
        return []
    
    print(f"✅ Gabarito identificado: {gabarito_file}")
    
    # ===========================================
    # IDENTIFICAR ARQUIVOS DOS ALUNOS
    # ===========================================
    
    print("\n👥 Identificando arquivos dos alunos...")
    
    # Todos os arquivos exceto o gabarito são considerados alunos
    arquivos_alunos = [f for f in arquivos['todos'] if f != gabarito_file]
    
    if not arquivos_alunos:
        print("❌ ERRO: Nenhum arquivo de aluno encontrado!")
        print("💡 Adicione arquivos dos alunos na pasta (qualquer nome, exceto gabarito)")
        return []
    
    print(f"✅ Encontrados {len(arquivos_alunos)} alunos para processar:")
    for i, aluno in enumerate(arquivos_alunos, 1):
        print(f"   {i:02d}. {aluno}")
    
    # ===========================================
    # CONFIGURAR GEMINI E GOOGLE SHEETS
    # ===========================================
    
    model_gemini = None
    if usar_gemini:
        print("\n🤖 Configurando Gemini...")
        try:
            model_gemini = configurar_gemini()
            print("✅ Gemini configurado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao configurar Gemini: {e}")
            print("⚠️ Continuando sem Gemini (apenas OCR)")
            usar_gemini = False
    
    # ===========================================
    # PROCESSAR GABARITO (UMA VEZ APENAS)
    # ===========================================
    
    print(f"\n{'='*60}")
    print("📋 PROCESSANDO GABARITO")
    print(f"{'='*60}")
    
    try:
        # Preprocessar gabarito
        gabarito_img = preprocessar_arquivo(gabarito_file, "gabarito")
        
        if debug_mode:
            print("\n=== DEBUG GABARITO ===")
        
        # Detectar respostas do gabarito
        if "page_" in gabarito_img and (gabarito_img.endswith(".png") or gabarito_img.endswith(".jpg")):
            print("🔍 Usando detecção especializada para PDF...")
            respostas_gabarito = detectar_respostas_pdf(gabarito_img, debug=debug_mode)
        else:
            respostas_gabarito = detectar_respostas(gabarito_img, debug=debug_mode)
        
        questoes_gabarito = sum(1 for r in respostas_gabarito if r != '?')
        print(f"✅ Gabarito processado: {questoes_gabarito}/52 questões detectadas")
        
        # Exibir gabarito em formato simples
        exibir_gabarito_simples(respostas_gabarito)
        
        if questoes_gabarito < 40:
            print("⚠️ ATENÇÃO: Poucas questões detectadas no gabarito. Verifique a qualidade da imagem.")
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO ao processar gabarito: {e}")
        return []
    
    # ===========================================
    # PROCESSAR TODOS OS ALUNOS
    # ===========================================
    
    resultados_lote = []
    alunos_processados = 0
    alunos_com_erro = 0
    
    print(f"\n{'='*60}")
    print(f"👥 PROCESSANDO {len(arquivos_alunos)} ALUNOS")
    print(f"{'='*60}")
    
    for i, aluno_file in enumerate(arquivos_alunos, 1):
        print(f"\n🔄 [{i:02d}/{len(arquivos_alunos)}] Processando: {aluno_file}")
        print("-" * 50)
        
        try:
            # Preprocessar arquivo do aluno
            aluno_img = preprocessar_arquivo(aluno_file, f"aluno_{i}")
            
            # Extrair dados do cabeçalho
            dados_aluno = {"Escola": "N/A", "Aluno": "N/A", "Nascimento": "N/A", "Turma": "N/A"}
            
            if usar_gemini and model_gemini:
                try:
                    dados_extraidos = extrair_cabecalho_com_fallback(model_gemini, aluno_img)
                    if dados_extraidos:
                        dados_aluno.update(dados_extraidos)
                        print("✅ Dados extraídos pelo Gemini:")
                        for campo, valor in dados_extraidos.items():
                            print(f"   📝 {campo}: {valor}")
                except Exception as e:
                    print(f"⚠️ Erro no Gemini para {aluno_file}: {e}")
                    dados_aluno["Aluno"] = os.path.splitext(aluno_file)[0]  # Usar nome do arquivo
            else:
                dados_aluno["Aluno"] = os.path.splitext(aluno_file)[0]  # Usar nome do arquivo
            
            # Detectar respostas do aluno
            if "page_" in aluno_img and (aluno_img.endswith(".png") or aluno_img.endswith(".jpg")):
                respostas_aluno = detectar_respostas_pdf(aluno_img, debug=debug_mode)
            else:
                respostas_aluno = detectar_respostas(aluno_img, debug=debug_mode)
            
            questoes_aluno = sum(1 for r in respostas_aluno if r != '?')
            print(f"✅ Respostas processadas: {questoes_aluno}/52 questões detectadas")
            
            # Calcular resultado
            resultado = comparar_respostas(respostas_gabarito, respostas_aluno)
            
            # Armazenar resultado
            resultado_completo = {
                "arquivo": aluno_file,
                "dados": dados_aluno,
                "respostas": respostas_aluno,
                "resultado": resultado,
                "questoes_detectadas": questoes_aluno
            }
            resultados_lote.append(resultado_completo)
            
            print(f"📊 Resultado: {resultado['acertos']}/{resultado['total']} acertos ({resultado['percentual']:.1f}%)")
            alunos_processados += 1
            
        except Exception as e:
            print(f"❌ ERRO ao processar {aluno_file}: {e}")
            alunos_com_erro += 1
            # Adicionar resultado de erro
            resultado_erro = {
                "arquivo": aluno_file,
                "dados": {"Aluno": os.path.splitext(aluno_file)[0], "Erro": str(e)},
                "respostas": ['?'] * 52,
                "resultado": {"total": 52, "acertos": 0, "erros": 52, "percentual": 0.0},
                "questoes_detectadas": 0
            }
            resultados_lote.append(resultado_erro)
    
    # ===========================================
    # RELATÓRIO FINAL E ESTATÍSTICAS
    # ===========================================
    
    print(f"\n{'='*60}")
    print("📊 RELATÓRIO FINAL DO LOTE")
    print(f"{'='*60}")
    
    print(f"\n=== ESTATÍSTICAS GERAIS ===")
    print(f"Total de alunos: {len(arquivos_alunos)}")
    print(f"Processados com sucesso: {alunos_processados} ✅")
    print(f"Erros de processamento: {alunos_com_erro} ❌")
    
    if resultados_lote:
        # Calcular estatísticas
        acertos_totais = [r["resultado"]["acertos"] for r in resultados_lote if "Erro" not in r["dados"]]
        percentuais = [r["resultado"]["percentual"] for r in resultados_lote if "Erro" not in r["dados"]]
        
        if acertos_totais:
            print(f"\n=== ESTATÍSTICAS DE DESEMPENHO ===")
            print(f"Média de acertos: {sum(acertos_totais)/len(acertos_totais):.1f}/52")
            print(f"Média percentual: {sum(percentuais)/len(percentuais):.1f}%")
            print(f"Melhor resultado: {max(acertos_totais)}/52 ({max(percentuais):.1f}%)")
            print(f"Pior resultado: {min(acertos_totais)}/52 ({min(percentuais):.1f}%)")
        
        # Mostrar ranking
        print(f"\n=== RANKING DOS ALUNOS ===")
        resultados_validos = [r for r in resultados_lote if "Erro" not in r["dados"]]
        resultados_ordenados = sorted(resultados_validos, key=lambda x: x["resultado"]["percentual"], reverse=True)
        
        for i, r in enumerate(resultados_ordenados[:10], 1):  # Top 10
            nome = r["dados"].get("Aluno", "N/A")
            acertos = r["resultado"]["acertos"]
            percentual = r["resultado"]["percentual"]
            print(f"   {i:02d}. {nome:<25} | {acertos:02d}/52 | {percentual:5.1f}%")
        
        if len(resultados_ordenados) > 10:
            print(f"   ... e mais {len(resultados_ordenados) - 10} alunos")
    
    return resultados_lote

def processar_pasta_gabaritos_sem_sheets(diretorio: str = "./gabaritos", usar_gemini: bool = True, debug_mode: bool = False):
    """
    Versão da função que NÃO tenta enviar para Google Sheets
    (evita problema de cota do Drive)
    """
    
    print("🚀 SISTEMA DE CORREÇÃO - PASTA GABARITOS (SEM GOOGLE SHEETS)")
    print("=" * 60)
    
    diretorio_gabaritos = diretorio
    
    if not os.path.exists(diretorio_gabaritos):
        print(f"❌ ERRO: Pasta '{diretorio_gabaritos}' não encontrada!")
        print("💡 Crie a pasta informada e adicione os arquivos do gabarito e dos alunos")
        return []
    
    # Configurar suporte a PDF se disponível
    if PDF_PROCESSOR_AVAILABLE:
        print("\n🔧 Configurando suporte a PDF...")
        pdf_ok = setup_pdf_support()
        if not pdf_ok:
            print("⚠️ Suporte a PDF limitado - apenas imagens serão processadas")
    
    # Listar arquivos suportados na pasta gabaritos
    print(f"\n📁 Analisando arquivos na pasta: {os.path.abspath(diretorio_gabaritos)}")
    arquivos = listar_arquivos_suportados(diretorio_gabaritos)
    
    if not arquivos['todos']:
        print("❌ Nenhum arquivo suportado encontrado na pasta gabaritos!")
        print("💡 Formatos suportados: PDF, PNG, JPG, JPEG, BMP, TIFF")
        return []
    
    print(f"✅ Encontrados {len(arquivos['todos'])} arquivos:")
    for arquivo in arquivos['todos']:
        print(f"   📄 {arquivo}")
    
    # ===========================================
    # IDENTIFICAR GABARITO (LÓGICA SIMPLIFICADA)
    # ===========================================
    
    print("\n📋 Identificando arquivo de gabarito...")
    gabarito_file = None
    
    # Buscar por qualquer arquivo que comece com "gabarito" (case insensitive)
    for arquivo in arquivos['todos']:
        if arquivo.lower().startswith('gabarito'):
            gabarito_file = arquivo
            break
    
    if not gabarito_file:
        print("❌ ERRO: Nenhum arquivo 'gabarito.*' encontrado!")
        print("💡 Renomeie o arquivo do gabarito para: gabarito.png, gabarito.pdf, etc.")
        return []
    
    print(f"✅ Gabarito identificado: {gabarito_file}")
    
    # ===========================================
    # IDENTIFICAR ARQUIVOS DOS ALUNOS (LÓGICA SIMPLIFICADA)
    # ===========================================
    
    print("\n👥 Identificando arquivos dos alunos...")
    
    # TODOS os arquivos que NÃO começam com "gabarito" são alunos
    arquivos_alunos = [f for f in arquivos['todos'] if not f.lower().startswith('gabarito')]
    
    if not arquivos_alunos:
        print("❌ ERRO: Nenhum arquivo de aluno encontrado!")
        print("💡 Adicione arquivos dos alunos na pasta gabaritos (qualquer nome exceto gabarito.*)")
        return []
    
    print(f"✅ Encontrados {len(arquivos_alunos)} alunos para processar:")
    for i, aluno in enumerate(arquivos_alunos, 1):
        print(f"   {i:02d}. {aluno}")
    
    # ===========================================
    # CONFIGURAR GEMINI
    # ===========================================
    
    model_gemini = None
    if usar_gemini:
        try:
            model_gemini = configurar_gemini()
        except Exception as e:
            print(f"❌ Erro ao configurar Gemini: {e}")
            usar_gemini = False
    
    # ===========================================
    # PROCESSAR GABARITO (UMA VEZ APENAS)
    # ===========================================
    
    print(f"\n{'='*60}")
    print("📋 PROCESSANDO GABARITO")
    print(f"{'='*60}")
    
    try:
        # Preprocessar gabarito
        gabarito_path = os.path.join(diretorio_gabaritos, gabarito_file)
        gabarito_img = preprocessar_arquivo(gabarito_path, "gabarito")
        
        # Detectar respostas do gabarito
        if "page_" in gabarito_img and (gabarito_img.endswith(".png") or gabarito_img.endswith(".jpg")):
            respostas_gabarito = detectar_respostas_pdf(gabarito_img, debug=debug_mode)
        else:
            respostas_gabarito = detectar_respostas(gabarito_img, debug=debug_mode)
        
        questoes_gabarito = sum(1 for r in respostas_gabarito if r != '?')
        print(f"✅ Gabarito processado: {questoes_gabarito}/52 questões detectadas")
        
        if questoes_gabarito < 40:
            print("⚠️ ATENÇÃO: Poucas questões detectadas no gabarito.")
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO ao processar gabarito: {e}")
        return []
    
    # ===========================================
    # PROCESSAR TODOS OS ALUNOS
    # ===========================================
    
    resultados_lote = []
    
    print(f"\n{'='*60}")
    print(f"👥 PROCESSANDO {len(arquivos_alunos)} ALUNOS")
    print(f"{'='*60}")
    
    for i, aluno_file in enumerate(arquivos_alunos, 1):
        print(f"\n🔄 [{i:02d}/{len(arquivos_alunos)}] Processando: {aluno_file}")
        print("-" * 50)
        
        try:
            # Preprocessar arquivo do aluno
            aluno_path = os.path.join(diretorio_gabaritos, aluno_file)
            aluno_img = preprocessar_arquivo(aluno_path, f"aluno_{i}")
            
            # Extrair dados do cabeçalho (opcional com Gemini)
            dados_aluno = {
                "Aluno": os.path.splitext(aluno_file)[0],
                "Escola": "N/A",
                "Nascimento": "N/A", 
                "Turma": "N/A"
            }
            
            if usar_gemini and model_gemini:
                try:
                    dados_extraidos = extrair_cabecalho_com_fallback(model_gemini, aluno_img)
                    if dados_extraidos:
                        # Mapear chaves minúsculas do Gemini para maiúsculas do sistema
                        mapeamento = {
                            "escola": "Escola",
                            "aluno": "Aluno", 
                            "turma": "Turma",
                            "nascimento": "Nascimento"
                        }
                        
                        # Atualizar dados com mapeamento correto
                        for chave_gemini, chave_sistema in mapeamento.items():
                            if chave_gemini in dados_extraidos and dados_extraidos[chave_gemini] and dados_extraidos[chave_gemini] != "N/A":
                                dados_aluno[chave_sistema] = dados_extraidos[chave_gemini]
                        
                        print(f"✅ Dados extraídos: {dados_aluno['Aluno']} ({dados_aluno['Escola']})")
                except Exception as e:
                    print(f"⚠️ Gemini falhou, usando nome do arquivo")
            
            # Detectar respostas do aluno
            if "page_" in aluno_img and (aluno_img.endswith(".png") or aluno_img.endswith(".jpg")):
                respostas_aluno = detectar_respostas_pdf(aluno_img, debug=debug_mode)
            else:
                respostas_aluno = detectar_respostas(aluno_img, debug=debug_mode)
            
            questoes_aluno = sum(1 for r in respostas_aluno if r != '?')
            print(f"✅ Respostas processadas: {questoes_aluno}/52 questões detectadas")
            
            # Calcular resultado
            resultado = comparar_respostas(respostas_gabarito, respostas_aluno)
            
            # Armazenar resultado com dados completos
            resultado_completo = {
                "arquivo": aluno_file,
                "dados_completos": dados_aluno,  # Dados completos do cabeçalho
                "acertos": resultado['acertos'],
                "total": resultado['total'],
                "percentual": resultado['percentual'],
                "questoes_detectadas": questoes_aluno
            }
            resultados_lote.append(resultado_completo)
            
            print(f"📊 Resultado: {resultado['acertos']}/{resultado['total']} acertos ({resultado['percentual']:.1f}%)")
            
        except Exception as e:
            print(f"❌ ERRO ao processar {aluno_file}: {e}")
            resultado_erro = {
                "arquivo": aluno_file,
                "dados_completos": {
                    "Aluno": os.path.splitext(aluno_file)[0],
                    "Escola": "N/A",
                    "Nascimento": "N/A",
                    "Turma": "N/A"
                },
                "acertos": 0,
                "total": 52,
                "percentual": 0.0,
                "questoes_detectadas": 0,
                "erro": str(e)
            }
            resultados_lote.append(resultado_erro)
    
    # ===========================================
    # RELATÓRIO FINAL SIMPLIFICADO
    # ===========================================
    
    print(f"\n{'='*60}")
    print("📊 RELATÓRIO FINAL")
    print(f"{'='*60}")
    
    if resultados_lote:
        print(f"\n=== RESULTADOS DETALHADOS ===")
        
        # Ordenar por percentual (decrescente)
        resultados_ordenados = sorted(resultados_lote, key=lambda x: x["percentual"], reverse=True)
        
        for i, r in enumerate(resultados_ordenados, 1):
            dados = r["dados_completos"]
            nome = dados["Aluno"]
            escola = dados["Escola"]
            nascimento = dados["Nascimento"]
            turma = dados["Turma"]
            acertos = r["acertos"]
            
            status = "❌" if "erro" in r else "✅"
            
            # Formato: aluno X (nome completo, escola, nascimento, turma) - acertou Y questões
            print(f"{status} aluno {i} ({nome}, {escola}, {nascimento}, {turma}) - acertou {acertos} questões")
        
        # Estatísticas
        resultados_validos = [r for r in resultados_lote if "erro" not in r]
        if resultados_validos:
            acertos = [r["acertos"] for r in resultados_validos]
            percentuais = [r["percentual"] for r in resultados_validos]
            
            print(f"\n=== ESTATÍSTICAS ===")
            print(f"Alunos processados: {len(resultados_validos)}/{len(arquivos_alunos)}")
            print(f"Média de acertos: {sum(acertos)/len(acertos):.1f}/52 questões")
            print(f"Média percentual: {sum(percentuais)/len(percentuais):.1f}%")
    
    # ===========================================
    # NÃO ENVIAR PARA GOOGLE SHEETS (PROBLEMA DE COTA)
    # ===========================================
    
    print(f"\n📄 Google Sheets DESABILITADO (evitando problema de cota do Drive)")
    print(f"💡 Todos os resultados foram exibidos acima")
    
    return resultados_lote

def processar_pasta_gabaritos_com_sheets(
    diretorio: str = "./gabaritos",
    usar_gemini: bool = True,
    debug_mode: bool = False
):
    """
    Versão da função que ENVIA para Google Sheets com controle de rate limiting
    """
    import time
    
    print("🚀 SISTEMA DE CORREÇÃO - PASTA GABARITOS (COM GOOGLE SHEETS)")
    print("=" * 60)
    
    diretorio_gabaritos = diretorio
    
    if not os.path.exists(diretorio_gabaritos):
        print(f"❌ ERRO: Pasta '{diretorio_gabaritos}' não encontrada!")
        print("💡 Crie a pasta informada e adicione os arquivos do gabarito e dos alunos")
        return []
    
    # Configurar suporte a PDF se disponível
    if PDF_PROCESSOR_AVAILABLE:
        print("\n🔧 Configurando suporte a PDF...")
        pdf_ok = setup_pdf_support()
        if not pdf_ok:
            print("⚠️ Suporte a PDF limitado - apenas imagens serão processadas")
    
    # Listar arquivos suportados na pasta gabaritos
    print(f"\n📁 Analisando arquivos na pasta: {os.path.abspath(diretorio_gabaritos)}")
    arquivos = listar_arquivos_suportados(diretorio_gabaritos)
    
    if not arquivos['todos']:
        print("❌ Nenhum arquivo suportado encontrado na pasta gabaritos!")
        print("💡 Formatos suportados: PDF, PNG, JPG, JPEG, BMP, TIFF")
        return []
    
    print(f"✅ Encontrados {len(arquivos['todos'])} arquivos:")
    for arquivo in arquivos['todos']:
        print(f"   📄 {arquivo}")
    
    # ===========================================
    # IDENTIFICAR GABARITO (LÓGICA SIMPLIFICADA)
    # ===========================================
    
    print("\n📋 Identificando arquivo de gabarito...")
    gabarito_file = None
    
    # Buscar por qualquer arquivo que comece com "gabarito" (case insensitive)
    for arquivo in arquivos['todos']:
        if arquivo.lower().startswith('gabarito'):
            gabarito_file = arquivo
            break
    
    if not gabarito_file:
        print("❌ ERRO: Nenhum arquivo 'gabarito.*' encontrado!")
        print("💡 Renomeie o arquivo do gabarito para: gabarito.png, gabarito.pdf, etc.")
        return []
    
    print(f"✅ Gabarito identificado: {gabarito_file}")
    
    # ===========================================
    # IDENTIFICAR ARQUIVOS DOS ALUNOS (LÓGICA SIMPLIFICADA)
    # ===========================================
    
    print("\n👥 Identificando arquivos dos alunos...")
    
    # TODOS os arquivos que NÃO começam com "gabarito" são alunos
    arquivos_alunos = [f for f in arquivos['todos'] if not f.lower().startswith('gabarito')]
    
    if not arquivos_alunos:
        print("❌ ERRO: Nenhum arquivo de aluno encontrado!")
        print("💡 Adicione arquivos dos alunos na pasta gabaritos (qualquer nome exceto gabarito.*)")
        return []
    
    print(f"✅ Encontrados {len(arquivos_alunos)} alunos para processar:")
    for i, aluno in enumerate(arquivos_alunos, 1):
        print(f"   {i:02d}. {aluno}")
    
    # ===========================================
    # CONFIGURAR GEMINI
    # ===========================================
    
    model_gemini = None
    if usar_gemini:
        print("\n🤖 Configurando Gemini...")
        try:
            model_gemini = configurar_gemini()
            print("✅ Gemini configurado!")
        except Exception as e:
            print(f"❌ Erro ao configurar Gemini: {e}")
            usar_gemini = False
    
    # ===========================================
    # CONFIGURAR GOOGLE SHEETS COM RATE LIMITING
    # ===========================================
    try:
        client = configurar_google_sheets()
        if client:
            PLANILHA_ID = "1VJ0_w9eoQcc-ouBnRoq5lFQdR2fVZkqEtR-KArZMuvk"
        else:
            print("❌ Erro ao configurar Google Sheets - continuando sem envio")
            client = None
    except Exception as e:
        print(f"❌ Erro ao configurar Google Sheets: {e}")
        client = None
    
    # ===========================================
    # PROCESSAR GABARITO (UMA VEZ APENAS)
    # ===========================================
    
    print(f"\n{'='*60}")
    print("📋 PROCESSANDO GABARITO")
    print(f"{'='*60}")
    
    try:
        # Preprocessar gabarito
        gabarito_path = os.path.join(diretorio_gabaritos, gabarito_file)
        gabarito_img = preprocessar_arquivo(gabarito_path, "gabarito")
        
        # Detectar respostas do gabarito
        if "page_" in gabarito_img and (gabarito_img.endswith(".png") or gabarito_img.endswith(".jpg")):
            respostas_gabarito = detectar_respostas_pdf(gabarito_img, debug=debug_mode)
        else:
            respostas_gabarito = detectar_respostas(gabarito_img, debug=debug_mode)
        
        questoes_gabarito = sum(1 for r in respostas_gabarito if r != '?')
        print(f"✅ Gabarito processado: {questoes_gabarito}/52 questões detectadas")
        
        # Exibir gabarito em formato simples
        exibir_gabarito_simples(respostas_gabarito)
        
        if questoes_gabarito < 40:
            print("⚠️ ATENÇÃO: Poucas questões detectadas no gabarito.")
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO ao processar gabarito: {e}")
        return []
    
    # ===========================================
    # PROCESSAR TODOS OS ALUNOS COM RATE LIMITING
    # ===========================================
    
    resultados_lote = []
    alunos_enviados_sheets = 0
    
    print(f"\n{'='*60}")
    print(f"👥 PROCESSANDO {len(arquivos_alunos)} ALUNOS")
    print(f"{'='*60}")
    
    for i, aluno_file in enumerate(arquivos_alunos, 1):
        print(f"\n🔄 [{i:02d}/{len(arquivos_alunos)}] Processando: {aluno_file}")
        print("-" * 50)
        
        try:
            # Preprocessar arquivo do aluno
            aluno_path = os.path.join(diretorio_gabaritos, aluno_file)
            aluno_img = preprocessar_arquivo(aluno_path, f"aluno_{i}")
            
            # Extrair dados do cabeçalho (opcional com Gemini)
            dados_aluno = {
                "Aluno": os.path.splitext(aluno_file)[0],
                "Escola": "N/A",
                "Nascimento": "N/A", 
                "Turma": "N/A"
            }
            
            if usar_gemini and model_gemini:
                try:
                    dados_extraidos = extrair_cabecalho_com_fallback(model_gemini, aluno_img)
                    if dados_extraidos:
                        # Mapear chaves minúsculas do Gemini para maiúsculas do sistema
                        mapeamento = {
                            "escola": "Escola",
                            "aluno": "Aluno", 
                            "turma": "Turma",
                            "nascimento": "Nascimento"
                        }
                        
                        # Atualizar dados com mapeamento correto
                        for chave_gemini, chave_sistema in mapeamento.items():
                            if chave_gemini in dados_extraidos and dados_extraidos[chave_gemini] and dados_extraidos[chave_gemini] != "N/A":
                                dados_aluno[chave_sistema] = dados_extraidos[chave_gemini]
                        
                        print(f"✅ Dados extraídos: {dados_aluno['Aluno']} ({dados_aluno['Escola']})")
                except Exception as e:
                    print(f"⚠️ Gemini falhou, usando nome do arquivo")
            
            # Detectar respostas do aluno
            if "page_" in aluno_img and (aluno_img.endswith(".png") or aluno_img.endswith(".jpg")):
                respostas_aluno = detectar_respostas_pdf(aluno_img, debug=debug_mode)
            else:
                respostas_aluno = detectar_respostas(aluno_img, debug=debug_mode)
            
            questoes_aluno = sum(1 for r in respostas_aluno if r != '?')
            print(f"✅ Respostas processadas: {questoes_aluno}/52 questões detectadas")
            
            # ===========================================
            # DEBUG: MOSTRAR DETECÇÕES DE QUESTÕES
            # ===========================================
            
            if debug_mode:
                print(f"\n🔍 DEBUG - RESPOSTAS DETECTADAS PARA {dados_aluno['Aluno']}:")
                print("=" * 60)
                
                # Mostrar respostas em formato organizado (4 colunas, 13 linhas)
                for linha in range(13):  # 13 linhas de questões
                    questoes_linha = []
                    for coluna in range(4):  # 4 colunas (A, B, C, D)
                        questao_num = linha * 4 + coluna + 1
                        if questao_num <= 52:
                            resposta = respostas_aluno[questao_num - 1] if questao_num <= len(respostas_aluno) else '?'
                            questoes_linha.append(f"Q{questao_num:02d}:{resposta}")
                    
                    print("   " + "  ".join(f"{q:<6}" for q in questoes_linha))
                
                # Mostrar estatísticas de detecção
                detectadas_por_alternativa = {
                    'A': sum(1 for r in respostas_aluno if r == 'A'),
                    'B': sum(1 for r in respostas_aluno if r == 'B'), 
                    'C': sum(1 for r in respostas_aluno if r == 'C'),
                    'D': sum(1 for r in respostas_aluno if r == 'D'),
                    '?': sum(1 for r in respostas_aluno if r == '?')
                }
                
                print(f"\n📊 ESTATÍSTICAS DE DETECÇÃO:")
                print(f"   🅰️ Alternativa A: {detectadas_por_alternativa['A']} questões")
                print(f"   🅱️ Alternativa B: {detectadas_por_alternativa['B']} questões") 
                print(f"   🅲 Alternativa C: {detectadas_por_alternativa['C']} questões")
                print(f"   🅳 Alternativa D: {detectadas_por_alternativa['D']} questões")
                print(f"   ❓ Não detectadas: {detectadas_por_alternativa['?']} questões")
                print(f"   ✅ Total detectado: {questoes_aluno}/52 questões ({(questoes_aluno/52)*100:.1f}%)")
                
                # Mostrar questões não detectadas se houver
                if detectadas_por_alternativa['?'] > 0:
                    nao_detectadas = [i+1 for i, r in enumerate(respostas_aluno) if r == '?']
                    print(f"   ⚠️ Questões não detectadas: {nao_detectadas}")
                
                print("=" * 60)
            
            # Calcular resultado
            resultado = comparar_respostas(respostas_gabarito, respostas_aluno)
            
            # Armazenar resultado com dados completos
            resultado_completo = {
                "arquivo": aluno_file,
                "dados_completos": dados_aluno,  # Dados completos do cabeçalho
                "acertos": resultado['acertos'],
                "total": resultado['total'],
                "percentual": resultado['percentual'],
                "questoes_detectadas": questoes_aluno
            }
            resultados_lote.append(resultado_completo)
            
            print(f"📊 Resultado: {resultado['acertos']}/{resultado['total']} acertos ({resultado['percentual']:.1f}%)")
            
            # ===========================================
            # ENVIAR PARA GOOGLE SHEETS COM RATE LIMITING
            # ===========================================
            
            if client:
                try:
                    print(f"📤 Enviando para Google Sheets (aluno {i}/{len(arquivos_alunos)})...")
                    
                    # RATE LIMITING: Aguardar entre envios para evitar quota
                    if i > 1:  # Não aguardar no primeiro
                        time.sleep(2)
                    
                    if enviar_para_planilha(client, dados_aluno, resultado, planilha_id=PLANILHA_ID):
                        alunos_enviados_sheets += 1
                        print(f"✅ Enviado para Google Sheets ({alunos_enviados_sheets}/{len(arquivos_alunos)})")
                    else:
                        print("⚠️ Falha no envio para Google Sheets")
                        
                except Exception as e:
                    print(f"❌ Erro ao enviar para Google Sheets: {e}")
                    if "quota" in str(e).lower() or "rate" in str(e).lower():
                        print("⚠️ Limite de quota atingido - aumentando delay...")
                        time.sleep(5)  # Delay maior em caso de quota
            
        except Exception as e:
            print(f"❌ ERRO ao processar {aluno_file}: {e}")
            resultado_erro = {
                "arquivo": aluno_file,
                "dados_completos": {
                    "Aluno": os.path.splitext(aluno_file)[0],
                    "Escola": "N/A",
                    "Nascimento": "N/A",
                    "Turma": "N/A"
                },
                "acertos": 0,
                "total": 52,
                "percentual": 0.0,
                "questoes_detectadas": 0,
                "erro": str(e)
            }
            resultados_lote.append(resultado_erro)
    
    # ===========================================
    # RELATÓRIO FINAL COM GOOGLE SHEETS
    # ===========================================
    
    print(f"\n{'='*60}")
    print("📊 RELATÓRIO FINAL")
    print(f"{'='*60}")
    
    if resultados_lote:
        print(f"\n=== RESULTADOS DETALHADOS ===")
        
        # Ordenar por percentual (decrescente)
        resultados_ordenados = sorted(resultados_lote, key=lambda x: x["percentual"], reverse=True)
        
        for i, r in enumerate(resultados_ordenados, 1):
            dados = r["dados_completos"]
            nome = dados["Aluno"]
            escola = dados["Escola"]
            nascimento = dados["Nascimento"]
            turma = dados["Turma"]
            acertos = r["acertos"]
            
            status = "❌" if "erro" in r else "✅"
            
            # Formato: aluno X (nome completo, escola, nascimento, turma) - acertou Y questões
            print(f"{status} aluno {i} ({nome}, {escola}, {nascimento}, {turma}) - acertou {acertos} questões")
        
        # Estatísticas
        resultados_validos = [r for r in resultados_lote if "erro" not in r]
        if resultados_validos:
            acertos = [r["acertos"] for r in resultados_validos]
            percentuais = [r["percentual"] for r in resultados_validos]
            
            print(f"\n=== ESTATÍSTICAS ===")
            print(f"Alunos processados: {len(resultados_validos)}/{len(arquivos_alunos)}")
            print(f"Média de acertos: {sum(acertos)/len(acertos):.1f}/52 questões")
            print(f"Média percentual: {sum(percentuais)/len(percentuais):.1f}%")
    
    # ===========================================
    # RELATÓRIO DO GOOGLE SHEETS
    # ===========================================
    
    if client:
        print(f"✅ Alunos enviados com sucesso: {alunos_enviados_sheets}/{len(arquivos_alunos)}")
        if alunos_enviados_sheets == len(arquivos_alunos):
            pass
        else:
            print("⚠️ Alguns alunos podem não ter sido enviados devido a limites de quota")
    else:
        print(f"\n📊 Google Sheets não configurado - apenas resultados locais")
    
    return resultados_lote


# ===========================================
# EXECUÇÃO PRINCIPAL
# ===========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sistema automatizado de correção de cartões resposta com Google Drive e Google Sheets."
    )
    parser.add_argument(
        "--drive-folder",
        dest="drive_folder_id",
        default="13KIDX3GtQWxIxlAsX-2XS0ypJvOnnqZX",
        help="ID da pasta do Google Drive contendo gabarito e cartões dos alunos"
    )
    parser.add_argument(
        "--gabarito",
        action="store_true",
        help="Exibe apenas o gabarito das questões em formato simples (1-A, 2-B, 3-C)"
    )

    args = parser.parse_args()

    print("🚀 SISTEMA AUTOMATIZADO DE CORREÇÃO DE CARTÃO RESPOSTA")
    print("=" * 60)
    print("✅ Configuração automática:")
    print("   • Google Sheets: ATIVADO")
    print("   • Gemini AI: ATIVADO") 
    print("   • Logs detalhados: ATIVADO")
    print("   • Mover arquivos processados: ATIVADO")

    if PDF_PROCESSOR_AVAILABLE:
        print("\n🔧 Configurando suporte a PDF...")
        pdf_ok = setup_pdf_support()
        if not pdf_ok:
            print("⚠️ Suporte a PDF limitado - apenas imagens serão processadas")

    # Configurações fixas para automação total
    usar_gemini = True
    enviar_para_sheets = True
    debug_mode = True
    mover_processados = True
    manter_temp = False

    # Sempre usar Google Drive
    drive_folder_id = args.drive_folder_id or os.getenv("DRIVE_FOLDER_ID")

    # Modo especial: apenas exibir gabarito
    if args.gabarito:
        processar_apenas_gabarito(drive_folder_id, debug_mode)
        exit(0)

        
    resultados = baixar_e_processar_pasta_drive(
        pasta_id=drive_folder_id,
        usar_gemini=usar_gemini,
        debug_mode=debug_mode,
        enviar_para_sheets=enviar_para_sheets,
        manter_pasta_temporaria=manter_temp,
        mover_processados=mover_processados
    )

    if resultados:
        pass
    else:
        print("\n❌ Nenhum resultado obtido.")
    
