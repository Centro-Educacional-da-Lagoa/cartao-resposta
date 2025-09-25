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
from datetime import datetime
import os
import base64
import io
from sklearn.cluster import KMeans

# Importação do processador de PDF
try:
    from pdf_processor_simple import process_pdf_file, is_pdf_file, setup_pdf_support
    PDF_PROCESSOR_AVAILABLE = True
    print("✅ Processador de PDF disponível")
except ImportError:
    PDF_PROCESSOR_AVAILABLE = False
    print("⚠️ Processador de PDF não disponível")

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
        print(f"📄 Arquivo PDF detectado - convertendo para imagem...")
        try:
            best_image, temp_files = process_pdf_file(file_path, keep_temp_files=False)
            print(f"✅ PDF convertido com sucesso!")
            print(f"   📁 Imagem gerada: {os.path.basename(best_image)}")
            return best_image
        except Exception as e:
            print(f"❌ Erro ao converter PDF: {e}")
            if "poppler" in str(e).lower():
                print("\n💡 SOLUÇÃO PARA POPPLER:")
                print("1. Baixe poppler: https://github.com/oschwartz10612/poppler-windows/releases")
                print("2. Extraia para C:\\poppler")
                print("3. Ou execute como admin: choco install poppler")
            raise e
    
    # Se for imagem, verificar se é válida
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"🖼️ Arquivo de imagem detectado - verificando...")
        try:
            # Tentar carregar a imagem para validar
            img = Image.open(file_path)
            img.verify()  # Verificar se a imagem é válida
            print(f"✅ Imagem válida!")
            print(f"   📐 Dimensões: {img.size[0]}x{img.size[1]} pixels")
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
    print(f"📸 Imagem do cabeçalho salva: debug_cabecalho.png (tamanho: {cabecalho.size})")
    
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
        print(f"Bolhas pintadas detectadas: {len(bolhas_pintadas)}")
        print(f"Área do crop: {crop.shape[1]}x{crop.shape[0]} pixels")
        print(f"Parâmetros usados - Área: {area_min}-{area_max}, Circ: {circularity_min:.2f}, Int: {intensity_max}")
        for i, (cx, cy, _, intensidade, area, circ, preenchimento) in enumerate(bolhas_pintadas):
            print(f"Bolha {i+1}: posição ({cx}, {cy}), intensidade: {intensidade:.1f}, área: {area:.0f}, circularidade: {circ:.2f}, preenchimento: {preenchimento:.2f}")
    
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
                        print(f"Questão {questao}: {resposta} (col {col_idx + 1}, linha {linha_idx + 1}, x: {cx}, intensidade: {intensidade:.1f}, preenchimento: {preenchimento:.2f})")
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
            print(f"Bolha {i+1}: posição ({cx}, {cy}), intensidade: {intensidade:.1f}, área: {area:.0f}, circularidade: {circ:.2f}, preenchimento: {preenchimento:.2f}")
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
                    print(f"Questão {q + 1}: {letra} (col {col_idx + 1}, linha {questao_idx + 1}, x: {cx}, intensidade: {intensidade:.1f}, preenchimento: {preenchimento:.2f})")
            
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
        GEMINI_API_KEY = "AIzaSyCj-A5_3ferd5ZPDww4v9wQymGld6LHALQ"  # Substitua pela sua chave
        
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
# SEÇÃO 4: INTEGRAÇÃO GOOGLE SHEETS
# ===========================================

def configurar_google_sheets():
    """Configura conexão com Google Sheets"""
    try:
        # Escopo das APIs
        scope = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Carregar credenciais
        credentials = Credentials.from_service_account_file('credenciais_google.json', scopes=scope)
        client = gspread.authorize(credentials)
        
        print("✅ Conexão com Google Sheets estabelecida!")
        return client
        
    except FileNotFoundError:
        print("❌ Arquivo 'credenciais_google.json' não encontrado!")
        print("📝 Certifique-se de que o arquivo está no diretório atual")
        return None
    except Exception as e:
        print(f"❌ Erro ao conectar com Google Sheets: {e}")
        return None

def enviar_para_planilha(client, dados_aluno, resultado_comparacao, planilha_id=None):
    """Envia dados para Google Sheets"""
    try:
        if planilha_id:
            sheet = client.open_by_key(planilha_id)
            print("✅ Planilha encontrada pelo ID!")
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
                "Data", "Nome da Escola", "Nome Completo", "Total de Acertos", "Total de Erros", "Porcentagem"
            ]
            worksheet.append_row(cabecalho)
            print("📋 Cabeçalho criado na planilha")
        
        # Preparar dados completos
        agora = datetime.now().strftime("%d/%m/%Y")
        
        # Limpar nome da escola (remover "RESULTADO FINAL")
        escola_limpa = dados_aluno["Escola"].replace("RESULTADO FINAL", "").strip()
        
        linha_dados = [
            agora,
            escola_limpa,
            dados_aluno["Aluno"],
            dados_aluno.get("Nascimento", "N/A"),
            dados_aluno.get("Turma", "N/A"),
            resultado_comparacao["acertos"],
            resultado_comparacao["erros"],
            f"{resultado_comparacao['percentual']:.1f}%"
        ]
        
        # Adicionar linha
        worksheet.append_row(linha_dados)
        print(f"✅ Dados enviados para Google Sheets com sucesso!")
        print(f"📊 Registro adicionado:")
        print(f"   🏫 Escola: {escola_limpa}")
        print(f"   👤 Aluno: {dados_aluno['Aluno']}")
        print(f"   📅 Nascimento: {dados_aluno.get('Nascimento', 'N/A')}")
        print(f"   📚 Turma: {dados_aluno.get('Turma', 'N/A')}")
        print(f"   📊 Resultado: {resultado_comparacao['acertos']} acertos | {resultado_comparacao['erros']} erros | {resultado_comparacao['percentual']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao enviar dados para Google Sheets: {e}")
        return False

def criar_planilha_detalhada(client, dados_aluno, resultado_comparacao):
    """Cria aba detalhada com todas as questões"""
    try:
        # Abrir planilha existente
        PLANILHA_ID = "1gUCK9ssOrZVxD-X2ccLkUVuJnKA-GUAzVfOY0NSDRHU"
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

# ===========================================
# PROCESSAMENTO EM LOTE
# ===========================================

def processar_pasta_gabaritos(usar_gemini=True, debug_mode=False):
    """
    Processa todos os arquivos da pasta 'gabaritos'
    - 1 gabarito (template) para comparar com múltiplos alunos
    - Sem comparações desnecessárias de dados
    
    Args:
        usar_gemini: Se deve usar Gemini para cabeçalho
        debug_mode: Se deve mostrar debug detalhado
        
    Returns:
        Lista de resultados de cada aluno processado
    """
    
    print("🚀 SISTEMA DE CORREÇÃO - PASTA GABARITOS")
    print("=" * 60)
    
    # Diretório fixo: pasta gabaritos
    diretorio_gabaritos = "./gabaritos"
    
    if not os.path.exists(diretorio_gabaritos):
        print("❌ ERRO: Pasta 'gabaritos' não encontrada!")
        print("💡 Crie a pasta 'gabaritos' e adicione os arquivos dos alunos e do gabarito")
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
                    dados_extraidos = extrair_cabecalho_com_gemini(model_gemini, aluno_img)
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
                    dados_extraidos = extrair_cabecalho_com_gemini(model_gemini, aluno_img)
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

def processar_pasta_gabaritos_sem_sheets(usar_gemini=True, debug_mode=False):
    """
    Versão da função que NÃO tenta enviar para Google Sheets
    (evita problema de cota do Drive)
    """
    
    print("🚀 SISTEMA DE CORREÇÃO - PASTA GABARITOS (SEM GOOGLE SHEETS)")
    print("=" * 60)
    
    # Diretório fixo: pasta gabaritos
    diretorio_gabaritos = "./gabaritos"
    
    if not os.path.exists(diretorio_gabaritos):
        print("❌ ERRO: Pasta 'gabaritos' não encontrada!")
        print("💡 Crie a pasta 'gabaritos' e adicione os arquivos dos alunos e do gabarito")
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
                    dados_extraidos = extrair_cabecalho_com_gemini(model_gemini, aluno_img)
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

def processar_pasta_gabaritos_com_sheets(usar_gemini=True, debug_mode=False):
    """
    Versão da função que ENVIA para Google Sheets com controle de rate limiting
    """
    import time
    
    print("🚀 SISTEMA DE CORREÇÃO - PASTA GABARITOS (COM GOOGLE SHEETS)")
    print("=" * 60)
    
    # Diretório fixo: pasta gabaritos
    diretorio_gabaritos = "./gabaritos"
    
    if not os.path.exists(diretorio_gabaritos):
        print("❌ ERRO: Pasta 'gabaritos' não encontrada!")
        print("💡 Crie a pasta 'gabaritos' e adicione os arquivos dos alunos e do gabarito")
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
    
    print("\n📊 Configurando Google Sheets...")
    try:
        client = configurar_google_sheets()
        if client:
            print("✅ Google Sheets configurado!")
            PLANILHA_ID = "1gUCK9ssOrZVxD-X2ccLkUVuJnKA-GUAzVfOY0NSDRHU"
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
                    dados_extraidos = extrair_cabecalho_com_gemini(model_gemini, aluno_img)
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
                        print("⏳ Aguardando 2 segundos (rate limiting)...")
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
        print(f"\n📊 GOOGLE SHEETS HABILITADO")
        print(f"✅ Alunos enviados com sucesso: {alunos_enviados_sheets}/{len(arquivos_alunos)}")
        if alunos_enviados_sheets == len(arquivos_alunos):
            print("🎉 Todos os resultados foram enviados para a planilha!")
        else:
            print("⚠️ Alguns alunos podem não ter sido enviados devido a limites de quota")
    else:
        print(f"\n📊 Google Sheets não configurado - apenas resultados locais")
    
    return resultados_lote


# ===========================================
# EXECUÇÃO PRINCIPAL
# ===========================================

if __name__ == "__main__":
    # ===========================================
    # EXECUÇÃO AUTOMÁTICA - COM GOOGLE SHEETS E RATE LIMITING
    # ===========================================
    
    print("🚀 SISTEMA DE CORREÇÃO DE CARTÃO RESPOSTA - MODO AUTOMÁTICO")
    print("=" * 60)
    
    # Configurar suporte a PDF se disponível
    if PDF_PROCESSOR_AVAILABLE:
        print("\n🔧 Configurando suporte a PDF...")
        pdf_ok = setup_pdf_support()
        if not pdf_ok:
            print("⚠️ Suporte a PDF limitado - apenas imagens serão processadas")
    
    # Verificar se pasta gabaritos existe
    if not os.path.exists("./gabaritos"):
        print("❌ ERRO: Pasta 'gabaritos' não encontrada!")
        print("💡 Crie a pasta 'gabaritos' e adicione:")
        print("   📋 1 arquivo de gabarito (gabarito.png, gabarito.pdf, etc.)")
        print("   📄 Arquivos dos alunos (qualquer nome)")
        exit(1)
    
    print(f"\n🎯 PROCESSANDO PASTA GABARITOS AUTOMATICAMENTE")
    print("⚙️ Configurações: Gemini=SIM, Debug=SIM, Google Sheets=SIM")
    print("💡 Lógica: Qualquer arquivo que NÃO comece com 'gabarito' = aluno")
    print("⏳ Rate limiting habilitado: 2s entre envios para Google Sheets")
    print("🔍 Debug habilitado: Mostrando detecções de questões para cada aluno")
    
    # Executar processamento automático da pasta gabaritos (COM GOOGLE SHEETS)
    # Configurações fixas para automação total
    resultados = processar_pasta_gabaritos_com_sheets(
        usar_gemini=True,    # Sempre usar Gemini para extrair dados
        debug_mode=True      # HABILITAR DEBUG para mostrar detecções
    )
    
    if resultados:
        print(f"\n🎉 PROCESSAMENTO AUTOMÁTICO CONCLUÍDO!")
        print(f"📊 {len(resultados)} alunos processados com sucesso.")
    else:
        print("\n❌ Nenhum resultado obtido.")
        exit(1)
    
