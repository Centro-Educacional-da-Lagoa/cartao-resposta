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
import time
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
import os 
from dotenv import load_dotenv
import base64
import io
import tempfile
import shutil
import argparse
from typing import List, Dict, Optional
from sklearn.cluster import KMeans

load_dotenv()

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

def converter_para_preto_e_branco(image_path: str, threshold: int = 180, salvar: bool = True) -> str:
    """
    Converte uma imagem colorida para preto e branco puro (binarizado)
    
    Args:
        image_path: Caminho da imagem original
        threshold: Valor de threshold (0-255). Menor = mais preto, Maior = mais branco
        salvar: Se deve salvar a imagem convertida
        
    Returns:
        Caminho da imagem convertida em preto e branco
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Não foi possível carregar a imagem: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_pb = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        if salvar:
            nome_base = os.path.splitext(image_path)[0]
            extensao = os.path.splitext(image_path)[1]
            output_path = f"{nome_base}_pb{extensao}"
            cv2.imwrite(output_path, img_pb)
            return output_path
        else:
            return image_path
            
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return image_path

def corrigir_rotacao_documento(image_path: str, debug: bool = False) -> str:
    """
    🔧 CORREÇÃO DE ROTAÇÃO - VERSÃO MELHORADA
    
    Detecta e corrige inclinação de documentos com precisão.
    
    Args:
        image_path: Caminho da imagem
        debug: Se deve salvar imagens intermediárias
        
    Returns:
        Caminho da imagem corrigida
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ═══════════════════════════════════════════════════
        # MÉTODO 1: Detectar contorno do documento
        # ═══════════════════════════════════════════════════
        
        # Binarização adaptativa (melhor para iluminação irregular)
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        angle_correcao = None
        
        if contours:
            # Pegar o maior contorno (documento)
            maior_contorno = max(contours, key=cv2.contourArea)
            
            # MinAreaRect retorna: ((center_x, center_y), (width, height), angle)
            rect = cv2.minAreaRect(maior_contorno)
            angle_raw = rect[2]
            
            # 🔧 CORREÇÃO DE ÂNGULO - OpenCV usa -90 a 0
            # Se width > height, o ângulo está na orientação errada
            box_width, box_height = rect[1]
            
            if box_width < box_height:
                angle_correcao = angle_raw
            else:
                angle_correcao = angle_raw + 90
            
            # Normalizar para -45° a 45°
            if angle_correcao > 45:
                angle_correcao = angle_correcao - 90
            elif angle_correcao < -45:
                angle_correcao = angle_correcao + 90
            
            if debug:
                print(f"   📐 Método 1 (Contorno): {angle_correcao:.3f}°")
        
        # ═══════════════════════════════════════════════════
        # MÉTODO 2: Hough Lines (Fallback)
        # ═══════════════════════════════════════════════════
        
        if angle_correcao is None or abs(angle_correcao) < 0.05:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
            
            if lines is not None and len(lines) > 5:
                angles = []
                for line in lines[:20]:
                    rho, theta = line[0]
                    angle_deg = np.degrees(theta) - 90
                    
                    if -45 <= angle_deg <= 45:
                        angles.append(angle_deg)
                
                if angles:
                    angle_correcao = np.median(angles)
                    
                    if debug:
                        print(f"   📐 Método 2 (Hough): {angle_correcao:.3f}°")
        
        # ═══════════════════════════════════════════════════
        # Aplicar Rotação
        # ═══════════════════════════════════════════════════
        
        if angle_correcao is None:
            print("   ⚠️ Não foi possível detectar ângulo")
            return image_path
        
        if abs(angle_correcao) < 0.05:
            if debug:
                print(f"   ✅ Rotação insignificante ({angle_correcao:.3f}°)")
            return image_path
        
        print(f"   🔄 Corrigindo rotação: {angle_correcao:.3f}°")
        
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_correcao, 1.0)
        
        img_rotated = cv2.warpAffine(
            img,
            rotation_matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
            flags=cv2.INTER_CUBIC
        )
        
        nome_base = os.path.splitext(image_path)[0]
        extensao = os.path.splitext(image_path)[1]
        output_path = f"{nome_base}_deskewed{extensao}"
        
        cv2.imwrite(output_path, img_rotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if debug:
            # Salvar imagem de debug com linhas detectadas
            debug_img = img.copy()
            cv2.drawContours(debug_img, [maior_contorno], -1, (0, 255, 0), 3)
            cv2.imwrite(f"{nome_base}_debug_contorno.png", debug_img)
        
        return output_path
        
    except Exception as e:
        print(f"   ❌ Erro na correção: {e}")
        return image_path

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
            
            # Retornar imagem sem correção
            best_image_corrigido = corrigir_rotacao_documento(best_image, debug=True)
            return best_image_corrigido
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
    Lista todos os arquivos suportados no diretório (imagens e PDFs).
    
    Args:
        diretorio: Caminho do diretório a ser listado (padrão: diretório atual)
    
    Returns:
        Dicionário com chaves 'imagens', 'pdfs' e 'todos' contendo listas de nomes de arquivos
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

# ===========================================
# SEÇÃO 2: OMR - DETECÇÃO DE ALTERNATIVAS MARCADAS
# ===========================================

def calcular_preenchimento_real(gray, contorno) -> float:
    """
    🆕 CALCULA PREENCHIMENTO REAL DA BOLHA
    Analisa pixels DENTRO do contorno para determinar % de área pintada
    """
    # Criar máscara do contorno
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contorno], -1, 255, -1)
    
    # Pegar apenas pixels dentro do contorno
    pixels_contorno = cv2.bitwise_and(gray, gray, mask=mask)
    pixels_validos = pixels_contorno[mask == 255]
    
    if len(pixels_validos) == 0:
        return 0.0
    
    # Contar pixels escuros (< 180 = pintados)
    pixels_pintados = np.sum(pixels_validos < 180)
    percentual = (pixels_pintados / len(pixels_validos)) * 100
    
    return percentual

def analisar_qualidade_marcacao(gray, contorno) -> dict:
    """
    🆕 ANÁLISE AVANÇADA DE MARCAÇÃO
    Retorna múltiplas métricas para validação rigorosa
    """
    # Calcular métricas básicas
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    circularidade = 4 * np.pi * area / (perimetro * perimetro) if perimetro > 0 else 0
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contorno)
    aspect_ratio = w / h if h > 0 else 0
    
    # Intensidade média
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contorno], -1, 255, -1)
    intensidade_media = cv2.mean(gray, mask=mask)[0]
    
    # Preenchimento real
    preenchimento = calcular_preenchimento_real(gray, contorno)
    
    # Desvio padrão (uniformidade da marcação)
    pixels_contorno = gray[mask == 255]
    desvio_padrao = np.std(pixels_contorno) if len(pixels_contorno) > 0 else 0
    
    return {
        'area': area,
        'circularidade': circularidade,
        'aspect_ratio': aspect_ratio,
        'intensidade': intensidade_media,
        'preenchimento': preenchimento,
        'desvio_padrao': desvio_padrao,
        'centro': (x + w//2, y + h//2),
        'contorno': contorno
    }

def eh_marcacao_valida(metricas: dict, debug: bool = False) -> tuple: #Nessa área a gente coloca como universal a validação da área, circularidade, aspect raiot, intensidade, preenchimento e uniformidade das bolhas. Caso queira que o bot pegue mais ou menos bolhas, o ajuste tem que ser feito por aqui
    """
    🆕 VALIDAÇÃO RIGOROSA DE MARCAÇÃO
    Retorna (é_válida, motivo_rejeição)
    """
    motivos_rejeicao = []
    
    # 1️⃣ ÁREA: Entre 80-2000 pixels (mais permissivo para círculos maiores)
    if not (60 <= metricas['area'] <= 2000):
        motivos_rejeicao.append(f"Área fora do padrão ({metricas['area']:.0f}px)") #Aqui regula o total de pixel junto que o bot vai determinar se é uma marcação ou não
    
    # 2️⃣ CIRCULARIDADE: > 0.12 (mais permissivo para círculos pintados à mão)
    if metricas['circularidade'] < 0.12:
        motivos_rejeicao.append(f"Baixa circularidade ({metricas['circularidade']:.2f})") #Circularidade da bolha, quanto mais próximo de 1, mais circular é a bolha.
    
    # 3️⃣ ASPECT RATIO: Entre 0.20-1.6 (aceitar formas levemente alongadas)
    if not (0.18 <= metricas['aspect_ratio'] <= 1.6):
        motivos_rejeicao.append(f"Forma não circular (ratio={metricas['aspect_ratio']:.2f})") #alongamento da bolha, quanto mais próximo de 1, mais circular é a bolha.
    
    # 4️⃣ INTENSIDADE: < 140 (mais permissivo para marcações com caneta)
    if metricas['intensidade'] >= 140:
        motivos_rejeicao.append(f"Intensidade alta ({metricas['intensidade']:.0f})") #Determina quão escura tem que ser a marcação, quanto menor o valor, mais escura é a marcação.
    
    # 5️⃣ PREENCHIMENTO: > 12% (mais permissivo para círculos pintados)
    if metricas['preenchimento'] < 12:
        motivos_rejeicao.append(f"Preenchimento baixo ({metricas['preenchimento']:.1f}%)") #Percentual de preenchimento real da bolha, quanto maior o valor, mais preenchida está a bolha. Valores menores podem dar falsos positivos
    
    # 6️⃣ UNIFORMIDADE: Desvio padrão < 65 (mais permissivo para marcação manual)
    if metricas['desvio_padrao'] > 65:
        motivos_rejeicao.append(f"Marcação irregular (dp={metricas['desvio_padrao']:.1f})") #Desvio padrão da intensidade dos pixels, quanto menor, mais uniforme é a marcação.
    
    eh_valida = len(motivos_rejeicao) == 0
    motivo = " | ".join(motivos_rejeicao) if motivos_rejeicao else "OK"
    
    return eh_valida, motivo

def salvar_debug_deteccao(image_path: str, bolhas_pintadas: list, crop: np.ndarray) -> None:
    """
    Salva imagem de debug com as bolhas detectadas marcadas em verde.
    
    Args:
        image_path: Caminho da imagem original
        bolhas_pintadas: Lista de tuplas (cx, cy, contorno, intensidade, area, circularidade, preenchimento)
        crop: Array numpy com a região recortada da imagem
    """
    debug_img = crop.copy()
    
    for cx, cy, cnt, intensidade, area, circ, preenchimento in bolhas_pintadas:
        cv2.circle(debug_img, (cx, cy), 8, (0, 255, 0), 2)
        cv2.putText(debug_img, f"{intensidade:.0f}", (cx-15, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    filename = image_path.replace('.jpg', '').replace('.png', '')
    debug_filename = f"debug_{os.path.basename(filename)}.png"
    cv2.imwrite(debug_filename, debug_img)

def detectar_respostas_pdf(image_path: str, debug: bool = False) -> list:
    """
    Detecta as respostas marcadas no cartão resposta convertido de PDF.
    Otimizado para imagens de alta resolução com parâmetros específicos para PDFs.
    
    Args:
        image_path: Caminho da imagem do PDF convertido
        debug: Se deve exibir informações de debug
    
    Returns:
        Lista com as respostas detectadas (44 ou 52 questões dependendo do cartão)
    VERSÃO UNIVERSAL: Detecta automaticamente se é 44 ou 52 questões.
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
    crop = image[int(height*0.55):int(height*0.92), int(width*0.02):int(width*0.98)]
    
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
        
        # PARÂMETROS MENOS RIGOROSOS - Alta resolução
        area_min = 300     # ↓ Reduzido (era 600)
        area_max = 10000   # ↑ Aumentado (era 6000)
        circularity_min = 0.06  # ↓ Muito flexível (era 0.12)
        intensity_max = 90      # ↑ Aumentado (era 60)
        
    else:
        # PARÂMETROS MENOS RIGOROSOS - Resolução normal
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 30, 155, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        area_min = 80      # ↓ Reduzido (era 100)
        area_max = 1500    # ↑ Aumentado (era 800)
        circularity_min = 0.10  # ↓ Muito flexível (era 0.25)
        intensity_max = 60      # ↑ Aumentado (era 35)
    
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
                                
                                # CRITÉRIOS MENOS RIGOROSOS para PDFs - Aceita mais marcações
                                aceita_marcacao = False
                                
                                # 1) Marcação escura com preenchimento mínimo
                                if intensidade_media < intensity_max and percentual_preenchimento > 0.15:  # ↓ 15% (era 25%)
                                    aceita_marcacao = True
                                
                                # 2) Marcação circular pouco preenchida
                                elif circularity > 0.15 and 0.08 <= percentual_preenchimento <= 0.95 and intensidade_media < intensity_max + 30:  # Muito mais tolerante
                                    aceita_marcacao = True
                                
                                # 3) Marcação grande com baixa intensidade
                                elif area > area_min * 2 and intensidade_media < intensity_max + 30 and percentual_preenchimento > 0.15:  # Bem flexível
                                    aceita_marcacao = True
                                
                                if aceita_marcacao:
                                    bolhas_pintadas.append((cx, cy, cnt, intensidade_media, area, circularity, percentual_preenchimento))
    
    # DETECÇÃO AUTOMÁTICA: Decidir se é 44 ou 52 questões baseado no número de bolhas
    num_bolhas = len(bolhas_pintadas)
    
    # Se detectar cerca de 44 bolhas (±20%), usar 44 questões
    # Se detectar mais ou menos, usar 52 questões
    if 35 <= num_bolhas <= 50:
        num_questoes = 44
        questoes_por_coluna = 11
        print(f"📋 PDF: Detectado cartão com 44 questões ({num_bolhas} bolhas)")
    else:
        num_questoes = 52
        questoes_por_coluna = 13
        print(f"📋 PDF: Detectado cartão com 52 questões ({num_bolhas} bolhas)")
    
    if debug:
        print(f"=== DEBUG PDF - ALTA RESOLUÇÃO ===")
        print(f"Área do crop: {crop.shape[1]}x{crop.shape[0]} pixels")
        print(f"Parâmetros usados - Área: {area_min}-{area_max}, Circ: {circularity_min:.2f}, Int: {intensity_max}")
        print(f"Bolhas detectadas: {num_bolhas}, Questões estimadas: {num_questoes}")

    # Verificar se temos bolhas suficientes
    if len(bolhas_pintadas) < 6:  # Mínimo mais baixo para PDFs
        print(f"⚠️ Poucas bolhas detectadas em PDF ({len(bolhas_pintadas)}). Tentando processamento simplificado.")
        if len(bolhas_pintadas) < 2:
            return ['?'] * num_questoes
    
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
        respostas = ['?'] * num_questoes
        questoes_por_coluna_calc = num_questoes // num_colunas
        extra_questoes = num_questoes % num_colunas
        
        questao = 1
        
        for col_idx, coluna in enumerate(colunas):
            # Calcular quantas questões esta coluna deve ter
            questoes_nesta_coluna = questoes_por_coluna_calc + (1 if col_idx < extra_questoes else 0)
            
            for linha_idx, (cx, cy, cnt, intensidade, area, circ, preenchimento) in enumerate(coluna):
                if linha_idx < questoes_nesta_coluna and questao <= num_questoes:
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
        return ['?'] * num_questoes


def detectar_respostas_52_questoes(image_path: str, debug: bool = False, eh_gabarito: bool = False) -> list:
    """
    OMR: Detecta APENAS alternativas pintadas usando OpenCV para cartões com 52 questões.
    Layout: 4 colunas x 13 linhas = 52 questões
    
    Args:
        image_path: Caminho da imagem
        debug: Se deve mostrar informações de debug
        eh_gabarito: Se True, usa crop otimizado para gabaritos (impressão limpa)
    
    Returns:
        Lista com 52 respostas detectadas (A/B/C/D ou '?' para não detectadas)
    """
    img_cv = cv2.imread(image_path)
    height, width = img_cv.shape[:2]
    
    # CROPS ESPECÍFICOS PARA CARTÕES DE 52 QUESTÕES
    if eh_gabarito:
        # GABARITO: Crop mais centralizado (impressão limpa e consistente)
        # Altura: 60% a 94% (mais restrito, gabaritos têm layout preciso)
        # Largura: 4% a 96% (margens maiores, área bem definida)
        crop = img_cv[int(height*0.60):int(height*0.98), int(width*0.02):int(width*0.98)]
    else:
        # ALUNOS: Crop mais amplo (marcações manuais podem variar)
        # Altura: 58% a 96% (mais tolerante para capturar marcações em diferentes posições)
        # Largura: 2% a 98% (margens mínimas para não perder marcações nas bordas)
        crop = img_cv[int(height*0.59):int(height*0.98), int(width*0.02):int(width*0.98)]
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro suave
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # FOCO: Threshold MUITO restritivo para detectar APENAS marcações PRETAS
    _, thresh = cv2.threshold(blur, 30, 155, cv2.THRESH_BINARY_INV) 
    
    # Operações morfológicas para preencher bolhas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 🆕 EXTRAIR BOLHAS COM ANÁLISE AVANÇADA
    bolhas_pintadas = []
    total_bolhas_validas = 0
    total_bolhas_rejeitadas = 0
    
    crop_height, crop_width = gray.shape
    
    for contour in contornos:
        metricas = analisar_qualidade_marcacao(gray, contour)
        cx, cy = metricas['centro']
        
        # Verificar se está na região das questões
        if not (20 < cx < crop_width - 20 and 20 < cy < crop_height - 20):
            continue
        
        # Validação rigorosa
        eh_valida, motivo = eh_marcacao_valida(metricas, debug)
        
        if not eh_valida:
            total_bolhas_rejeitadas += 1
            continue
        
        total_bolhas_validas += 1
        bolhas_pintadas.append((cx, cy, metricas['contorno'], metricas['intensidade'], 
                                metricas['area'], metricas['circularidade'], metricas['preenchimento']))
    
    if debug or eh_gabarito:
        print(f"\n📊 Análise de Bolhas (52 questões):")
        print(f"   ✅ Válidas: {total_bolhas_validas}")
        if total_bolhas_validas + total_bolhas_rejeitadas > 0:
            print(f"   📈 Taxa de aceitação: {total_bolhas_validas/(total_bolhas_validas+total_bolhas_rejeitadas)*100:.1f}%\n")
    
    if debug:
        salvar_debug_deteccao(image_path, bolhas_pintadas, crop)
    
    # Verificar se temos bolhas suficientes para processamento
    if len(bolhas_pintadas) < 4:
        print(f"⚠️ Poucas bolhas detectadas ({len(bolhas_pintadas)}). Retornando lista vazia.")
        return ['?'] * 52
    

    # 1) Após montar bolhas_pintadas, separe só os 'cx' (centros X)
    xs = np.array([b[0] for b in bolhas_pintadas], dtype=np.float32).reshape(-1, 1)

    # 2) Determinar número de colunas baseado no número de bolhas
    num_colunas = 4  # Pelo menos 3 bolhas por coluna
    
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
    letras = ['a', 'b', 'c', 'd']
    respostas_finais = ['?'] * 52


    for col_idx, bolhas_coluna in enumerate(bolhas_por_coluna):
        if not bolhas_coluna:
            continue

        # Se há bolhas suficientes na coluna, tentar detectar alternativas A-D
        if len(bolhas_coluna) >= 4:
            xs_col = np.array([b[0] for b in bolhas_coluna], dtype=np.float32).reshape(-1, 1)
            # Sempre usar 4 clusters para as 4 alternativas (A, B, C, D)
            k_opts = KMeans(n_clusters=4, n_init=10, random_state=0).fit(xs_col)
            centros_opts = k_opts.cluster_centers_.flatten()
            
            # 🔧 VALIDAÇÃO: Verificar se há centros duplicados ou muito próximos
            centros_ordenados_temp = sorted(centros_opts)
            tem_duplicados = False
            for i in range(len(centros_ordenados_temp) - 1):
                distancia = abs(centros_ordenados_temp[i+1] - centros_ordenados_temp[i])
                if distancia < 5:  # Se distância < 5 pixels, consideramos duplicado
                    tem_duplicados = True
                    break
            
            if tem_duplicados:
                # FALLBACK: Usar ordenação direta das posições X únicas
                xs_unicos = sorted(list(set([b[0] for b in bolhas_coluna])))
                if len(xs_unicos) >= 4:
                    # Agrupar posições próximas (< 15px) e usar mediana
                    grupos = []
                    for x in xs_unicos:
                        if not grupos or abs(x - np.median(grupos[-1])) > 15:
                            grupos.append([x])
                        else:
                            grupos[-1].append(x)
                    
                    # Verificar se conseguimos 4 grupos
                    if len(grupos) >= 4:
                        # Pegar mediana de cada grupo
                        centros_opts = np.array([np.median(g) for g in grupos[:4]], dtype=np.float32)
                    else:
                        # FALLBACK DO FALLBACK: Dividir espaço igualmente
                        x_min = min(xs_unicos)
                        x_max = max(xs_unicos)
                        espacamento = (x_max - x_min) / 3
                        centros_opts = np.array([x_min, x_min + espacamento, x_min + 2*espacamento, x_max], dtype=np.float32)
            
            ordem_opts = np.argsort(centros_opts)  # esquerda→direita ⇒ A,B,C,D
        else:
            # Processamento simplificado se há poucas bolhas
            ordem_opts = list(range(len(bolhas_coluna)))
            centros_opts = [b[0] for b in bolhas_coluna]

        # Agrupe por LINHAS usando tolerância mais flexível
        ys = sorted([b[1] for b in bolhas_coluna])
        dy = np.median(np.diff(ys)) if len(ys) > 5 else 25  # Espaçamento base maior
        tolerance_y = max(18, int(dy * 0.5))

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
            espacamento_questao = altura_total / 12 if len(linhas) > 1 else 25
        else:
            continue
        
        # AJUSTE ESPECÍFICO PARA COLUNA 3 (índice 2)
        if col_idx == 2:
            tolerancia_multiplicador = 1.5  # Muito mais flexível para coluna 3
            if debug:
                print(f"🔧 Coluna 3: Usando tolerância aumentada ({tolerancia_multiplicador}x)")
        else:
            tolerancia_multiplicador = 1.5
            
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
            
            # TOLERÂNCIA AJUSTADA
            tolerancia = espacamento_questao * tolerancia_multiplicador
            
            for idx, linha in enumerate(linhas):
                if idx in linhas_usadas:  # Pular linhas já usadas
                    continue
                    
                y_linha = linha[0][1]  # Y da primeira bolha da linha
                distancia = abs(y_linha - y_esperado)
                
                # Tolerância: aceitar linha se estiver dentro de uma janela mais ampla
                if distancia < tolerancia and distancia < menor_distancia:
                    menor_distancia = distancia
                    linha_mais_proxima = linha
                    linha_mais_proxima_idx = idx
            
            if linha_mais_proxima is not None:
                # Marcar linha como usada
                linhas_usadas.add(linha_mais_proxima_idx)
                
                # 🔍 DETECÇÃO DE DUPLA MARCAÇÃO
                # Threshold: intensidade abaixo de 50 = marcada
                threshold_marcada = 50
                bolhas_marcadas = [b for b in linha_mais_proxima if b[3] < threshold_marcada]
                
                letra = '?'
                
                if len(bolhas_marcadas) == 0:
                    # Nenhuma bolha marcada (todas muito claras)
                    # TENTATIVA DE RECUPERAÇÃO: Pegar a bolha mais escura se estiver razoável
                    bolha_mais_escura = min(linha_mais_proxima, key=lambda b: b[3])
                    
                    # Para coluna 3, ser mais permissivo na recuperação
                    threshold_recuperacao = 85 if col_idx != 2 else 95
                    
                    if bolha_mais_escura[3] < threshold_recuperacao:
                        # Verificar se há diferença clara entre a mais escura e as outras
                        segunda_mais_escura = sorted(linha_mais_proxima, key=lambda b: b[3])[1] if len(linha_mais_proxima) > 1 else None
                        
                        if segunda_mais_escura and (segunda_mais_escura[3] - bolha_mais_escura[3]) > 30:
                            bolhas_marcadas = [bolha_mais_escura]
                        else:
                            letra = '?'
                    else:
                        letra = '?'
                
                if len(bolhas_marcadas) >= 2:
                    # ❌ DUPLA MARCAÇÃO DETECTADA - ANULAR QUESTÃO
                    letra = '?'
                    if eh_gabarito:
                        print(f"⚠️ GABARITO Q{q+1}: DUPLA MARCAÇÃO detectada! Questão ANULADA")
                
                elif len(bolhas_marcadas) == 1 and letra == '?':
                    # ✅ UMA marcação (correto!)
                    bolha_marcada = bolhas_marcadas[0]
                    
                    # 🎯 DETECÇÃO POR ZONAS (Método Melhorado)
                    cx = bolha_marcada[0]  # Posição X da bolha marcada
                    
                    # Se há alternativas suficientes detectadas
                    if len(centros_opts) >= 4:
                        # Ordenar centros da esquerda → direita
                        centros_ordenados = sorted(centros_opts)
                        
                        # Calcular ZONAS DE TOLERÂNCIA entre cada centro
                        # Zona A: [início, meio entre A e B]
                        # Zona B: [meio entre A e B, meio entre B e C]
                        # Zona C: [meio entre B e C, meio entre C e D]
                        # Zona D: [meio entre C e D, fim]
                        
                        zonas = []
                        for i in range(len(centros_ordenados)):
                            if i == 0:
                                # Primeira alternativa (A): desde o início até meio do caminho para B
                                limite_inferior = 50
                                limite_superior = (centros_ordenados[0] + centros_ordenados[1]) / 2 if len(centros_ordenados) > 1 else centros_ordenados[0] + 50
                            elif i == len(centros_ordenados) - 1:
                                # Última alternativa (D): desde meio do caminho de C até o fim
                                limite_inferior = (centros_ordenados[i-1] + centros_ordenados[i]) / 2
                                limite_superior = float('inf')
                            else:
                                # Alternativas do meio (B, C): entre os meios dos intervalos adjacentes
                                limite_inferior = (centros_ordenados[i-1] + centros_ordenados[i]) / 2
                                limite_superior = (centros_ordenados[i] + centros_ordenados[i+1]) / 2
                            
                            zonas.append((limite_inferior, limite_superior))
                        # Verificar em qual ZONA a bolha marcada está
                        for idx, (lim_inf, lim_sup) in enumerate(zonas):
                            if lim_inf <= cx < lim_sup:
                                # Encontrou a zona! Mapear para letra
                                # Encontrar qual índice original este centro ordenado tinha
                                centro_original_idx = np.where(centros_opts == centros_ordenados[idx])[0][0]
                                letra_idx = int(np.where(ordem_opts == centro_original_idx)[0][0])
                                
                                if 0 <= letra_idx < len(letras):
                                    letra = letras[letra_idx]
                                break
            else:
                # NÃO ENCONTROU LINHA PRÓXIMA
                letra = '?'

            respostas_finais[q] = letra
    
    return respostas_finais

def detectar_respostas_44_questoes(image_path: str, debug: bool = False, eh_gabarito: bool = False) -> list:
    """
    OMR: Detecta APENAS alternativas pintadas usando OpenCV para cartões com 44 questões.
    Layout: 4 colunas x 11 linhas = 44 questões
    
    Args:
        image_path: Caminho da imagem
        debug: Se deve mostrar informações de debug
        eh_gabarito: Se True, usa crop otimizado para gabaritos (impressão limpa)
    
    Returns:
        Lista com 44 respostas detectadas (A/B/C/D ou '?' para não detectadas)
    """
    img_cv = cv2.imread(image_path)
    height, width = img_cv.shape[:2]
    
    # CROPS ESPECÍFICOS PARA CARTÕES DE 44 QUESTÕES
    if eh_gabarito:
        # GABARITO: Crop mais centralizado (impressão limpa e consistente)
        crop = img_cv[int(height*0.59):int(height*0.93), int(width*0.02):int(width*0.98)]
    else:
        # ALUNOS: Crop mais amplo (marcações manuais podem variar)
        crop = img_cv[int(height*0.58):int(height*0.96), int(width*0.02):int(width*0.98)]
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro suave
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # FOCO: Threshold MUITO restritivo para detectar APENAS marcações PRETAS
    _, thresh = cv2.threshold(blur, 40, 200, cv2.THRESH_BINARY_INV) 
    
    # Operações morfológicas para preencher bolhas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 🆕 EXTRAIR BOLHAS COM ANÁLISE AVANÇADA
    bolhas_pintadas = []
    total_bolhas_validas = 0
    total_bolhas_rejeitadas = 0
    
    crop_height, crop_width = gray.shape
    
    for contour in contornos:
        metricas = analisar_qualidade_marcacao(gray, contour)
        cx, cy = metricas['centro']
        
        # Verificar se está na região das questões
        if not (20 < cx < crop_width - 20 and 20 < cy < crop_height - 20):
            continue
        
        # Validação rigorosa
        eh_valida, motivo = eh_marcacao_valida(metricas, debug)
        
        if not eh_valida:
            total_bolhas_rejeitadas += 1
            continue
        
        total_bolhas_validas += 1
        bolhas_pintadas.append((cx, cy, metricas['contorno'], metricas['intensidade'], 
                                metricas['area'], metricas['circularidade'], metricas['preenchimento']))
    
    if debug or eh_gabarito:
        print(f"\n📊 Análise de Bolhas (44 questões):")
        print(f"   ✅ Válidas: {total_bolhas_validas}")
        if total_bolhas_validas + total_bolhas_rejeitadas > 0:
            print(f"   📈 Taxa de aceitação: {total_bolhas_validas/(total_bolhas_validas+total_bolhas_rejeitadas)*100:.1f}%\n")
    
    if debug:
        salvar_debug_deteccao(image_path, bolhas_pintadas, crop)
    
    # Verificar se temos bolhas suficientes para processamento
    if len(bolhas_pintadas) < 4:
        print(f"⚠️ Poucas bolhas detectadas ({len(bolhas_pintadas)}). Retornando lista vazia.")
        return ['?'] * 44
    
    # MELHORIA: Organização mais precisa usando KMeans para detectar as 4 colunas
    xs = np.array([b[0] for b in bolhas_pintadas], dtype=np.float32).reshape(-1, 1)
    num_colunas = min(4, max(1, len(bolhas_pintadas) // 3))
    
    if num_colunas < 4:
        print(f"⚠️ Detectadas apenas {num_colunas} colunas possíveis. Processamento simplificado.")
    
    k_cols = KMeans(n_clusters=num_colunas, n_init=10, random_state=0).fit(xs)
    col_idx_por_bolha = k_cols.predict(xs)
    centros_cols = sorted(k_cols.cluster_centers_.flatten())

    ordem_cols = np.argsort(k_cols.cluster_centers_.flatten())
    remap = {int(c): i for i, c in enumerate(ordem_cols)}

    bolhas_por_coluna = [[] for _ in range(num_colunas)]
    for bolha, c_orig in zip(bolhas_pintadas, col_idx_por_bolha):
        if remap[int(c_orig)] < num_colunas:
            bolhas_por_coluna[remap[int(c_orig)]].append(bolha)

    # Para CADA coluna, processar as questões
    letras = ['a', 'b', 'c', 'd']
    respostas_finais = ['?'] * 44

    for col_idx, bolhas_coluna in enumerate(bolhas_por_coluna):
        if not bolhas_coluna:
            continue

        # Se há bolhas suficientes na coluna, tentar detectar alternativas A-D
        if len(bolhas_coluna) >= 4:
            xs_col = np.array([b[0] for b in bolhas_coluna], dtype=np.float32).reshape(-1, 1)
            # Sempre usar 4 clusters para as 4 alternativas (A, B, C, D)
            k_opts = KMeans(n_clusters=4, n_init=10, random_state=0).fit(xs_col)
            centros_opts = k_opts.cluster_centers_.flatten()
            
            # 🔧 VALIDAÇÃO: Verificar se há centros duplicados ou muito próximos
            centros_ordenados_temp = sorted(centros_opts)
            tem_duplicados = False
            for i in range(len(centros_ordenados_temp) - 1):
                distancia = abs(centros_ordenados_temp[i+1] - centros_ordenados_temp[i])
                if distancia < 5:  # Se distância < 5 pixels, consideramos duplicado
                    tem_duplicados = True
                    break
            
            if tem_duplicados:
                # FALLBACK: Usar ordenação direta das posições X únicas
                xs_unicos = sorted(list(set([b[0] for b in bolhas_coluna])))
                if len(xs_unicos) >= 4:
                    # Agrupar posições próximas (< 15px) e usar mediana
                    grupos = []
                    for x in xs_unicos:
                        if not grupos or abs(x - np.median(grupos[-1])) > 15:
                            grupos.append([x])
                        else:
                            grupos[-1].append(x)
                    
                    # Verificar se conseguimos 4 grupos
                    if len(grupos) >= 4:
                        # Pegar mediana de cada grupo
                        centros_opts = np.array([np.median(g) for g in grupos[:4]], dtype=np.float32)
                    else:
                        # FALLBACK DO FALLBACK: Dividir espaço igualmente
                        x_min = min(xs_unicos)
                        x_max = max(xs_unicos)
                        espacamento = (x_max - x_min) / 3
                        centros_opts = np.array([x_min, x_min + espacamento, x_min + 2*espacamento, x_max], dtype=np.float32)
            
            ordem_opts = np.argsort(centros_opts)
        else:
            ordem_opts = list(range(len(bolhas_coluna)))
            centros_opts = [b[0] for b in bolhas_coluna]

        # Agrupe por LINHAS
        ys = sorted([b[1] for b in bolhas_coluna])
        dy = np.median(np.diff(ys)) if len(ys) > 5 else 25
        tolerance_y = max(18, int(dy * 0.5))

        linhas = []
        for bolha in sorted(bolhas_coluna, key=lambda b: b[1]):
            cy = bolha[1]
            if not linhas or abs(cy - linhas[-1][0][1]) > tolerance_y:
                linhas.append([bolha])
            else:
                linhas[-1].append(bolha)

        linhas.sort(key=lambda linha: linha[0][1])
        linhas_usadas = set()

        # Cada coluna tem 11 questões - MAPEAMENTO CORRETO PARA 44 QUESTÕES
        offset_questao = col_idx * 11
        
        if linhas:
            y_min = min(linha[0][1] for linha in linhas)
            y_max = max(linha[0][1] for linha in linhas)
            altura_total = y_max - y_min
            espacamento_questao = altura_total / 10 if len(linhas) > 1 else 25  # 10 intervalos para 11 questões
        else:
            continue
        
        # AJUSTE ESPECÍFICO PARA AS COLUNAS
        if col_idx == 0:
            tolerancia_multiplicador = 2.5  
        else:
            tolerancia_multiplicador = 1.5

        if col_idx == 1:
            tolerancia_multiplicador = 2.5  
        else:
            tolerancia_multiplicador = 1.5

        if col_idx == 2:
            tolerancia_multiplicador = 2.5
        else:
            tolerancia_multiplicador = 1.5
            
        # Para cada questão (0-10) na coluna
        num_questoes = min(11, 44 - offset_questao)
        
        for questao_idx in range(num_questoes):
            q = offset_questao + questao_idx
            if q >= 44:
                break
                
            y_esperado = y_min + (questao_idx * espacamento_questao)
            
            linha_mais_proxima = None
            linha_mais_proxima_idx = -1
            menor_distancia = float('inf')
            
            # TOLERÂNCIA AJUSTADA
            tolerancia = espacamento_questao * tolerancia_multiplicador
            
            for idx, linha in enumerate(linhas):
                if idx in linhas_usadas:
                    continue
                    
                y_linha = linha[0][1]
                distancia = abs(y_linha - y_esperado)
                
                if distancia < tolerancia and distancia < menor_distancia:
                    menor_distancia = distancia
                    linha_mais_proxima = linha
                    linha_mais_proxima_idx = idx
            
            if linha_mais_proxima is not None:
                linhas_usadas.add(linha_mais_proxima_idx)
                
                # 🔍 DETECÇÃO DE DUPLA MARCAÇÃO
                # Threshold: intensidade abaixo de 70 = marcada
                threshold_marcada = 70
                bolhas_marcadas = [b for b in linha_mais_proxima if b[3] < threshold_marcada]
                
                letra = '?'
                
                if len(bolhas_marcadas) == 0:
                    # Nenhuma bolha marcada (todas muito claras)
                    # TENTATIVA DE RECUPERAÇÃO: Pegar a bolha mais escura se estiver razoável
                    bolha_mais_escura = min(linha_mais_proxima, key=lambda b: b[3])
                    
                    # Para coluna 3, ser mais permissivo na recuperação
                    threshold_recuperacao = 95 if col_idx != 2 else 105
                    
                    if bolha_mais_escura[3] < threshold_recuperacao:
                        # Verificar se há diferença clara entre a mais escura e as outras
                        segunda_mais_escura = sorted(linha_mais_proxima, key=lambda b: b[3])[1] if len(linha_mais_proxima) > 1 else None
                        
                        if segunda_mais_escura and (segunda_mais_escura[3] - bolha_mais_escura[3]) > 30:
                            bolhas_marcadas = [bolha_mais_escura]
                        else:
                            letra = '?'
                    else:
                        letra = '?'
                
                if len(bolhas_marcadas) >= 2:
                    # ❌ DUPLA MARCAÇÃO DETECTADA - ANULAR QUESTÃO
                    letra = '?'
                    if eh_gabarito:
                        print(f"⚠️ GABARITO Q{q+1}: DUPLA MARCAÇÃO detectada! Questão ANULADA")
                
                elif len(bolhas_marcadas) == 1 and letra == '?':
                    # ✅ UMA marcação (correto!)
                    bolha_marcada = bolhas_marcadas[0]
                    
                    # 🎯 DETECÇÃO POR ZONAS (Método Melhorado)
                    cx = bolha_marcada[0]  # Posição X da bolha marcada
                    
                    # Se há alternativas suficientes detectadas
                    if len(centros_opts) >= 4:
                        # Ordenar centros da esquerda → direita
                        centros_ordenados = sorted(centros_opts)
                        
                        # Calcular ZONAS DE TOLERÂNCIA entre cada centro
                        zonas = []
                        for i in range(len(centros_ordenados)):
                            if i == 0:
                                # Primeira alternativa (A): desde o início até meio do caminho para B
                                limite_inferior = 50
                                limite_superior = (centros_ordenados[0] + centros_ordenados[1]) / 2 if len(centros_ordenados) > 1 else centros_ordenados[0] + 50
                            elif i == len(centros_ordenados) - 1:
                                # Última alternativa (D): desde meio do caminho de C até o fim
                                limite_inferior = (centros_ordenados[i-1] + centros_ordenados[i]) / 2
                                limite_superior = float('inf')
                            else:
                                # Alternativas do meio (B, C): entre os meios dos intervalos adjacentes
                                limite_inferior = (centros_ordenados[i-1] + centros_ordenados[i]) / 2
                                limite_superior = (centros_ordenados[i] + centros_ordenados[i+1]) / 2
                            
                            zonas.append((limite_inferior, limite_superior))
                        
                        # Verificar em qual ZONA a bolha marcada está
                        for idx, (lim_inf, lim_sup) in enumerate(zonas):
                            if lim_inf <= cx < lim_sup:
                                # Encontrou a zona! Mapear para letra
                                centro_original_idx = np.where(centros_opts == centros_ordenados[idx])[0][0]
                                letra_idx = int(np.where(ordem_opts == centro_original_idx)[0][0])
                                
                                if 0 <= letra_idx < len(letras):
                                    letra = letras[letra_idx]
                                break
            else:
                # NÃO ENCONTROU LINHA PRÓXIMA
                letra = '?'

            respostas_finais[q] = letra
    
    return respostas_finais
   


def detectar_respostas_universal(image_path: str, debug: bool = False) -> list:
    """
    Função universal que detecta automaticamente se o cartão tem 44 ou 52 questões
    e chama a função apropriada.
    
    Args:
        image_path: Caminho da imagem do cartão resposta
        debug: Se deve exibir informações de debug
        
    Returns:
        Lista com as respostas detectadas (tamanho 44 ou 52 dependendo do cartão)
    """
    # Primeiro, detectar bolhas para estimar quantidade de questões
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"❌ Erro ao carregar imagem: {image_path}")
        return ['?'] * 52  # Retorna 52 por padrão em caso de erro
    
    height, width = img_cv.shape[:2]
    crop = img_cv[int(height*0.60):int(height*0.94), int(width*0.02):int(width*0.98)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 30, 155, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Contar bolhas válidas - PARÂMETROS MENOS RIGOROSOS
    num_bolhas = 0
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if 50 < area < 1500:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.25:  # ↓ 0.10 (era 0.25) - MUITO MAIS FLEXÍVEL
                    num_bolhas += 1
    
    # Decidir qual função usar baseado no número de bolhas detectadas
    # Se detectar cerca de 44 bolhas (±20%), usar função de 44 questões
    # Se detectar cerca de 52 bolhas (±20%), usar função de 52 questões
    
    if debug:
        print(f"🔍 Detecção Universal: {num_bolhas} bolhas encontradas")
    
    # Thresholds para decidir o tipo de cartão
    # Se tem entre 35-50 bolhas, provavelmente é cartão de 44 questões
    # Se tem mais de 50 bolhas ou menos de 35, provavelmente é cartão de 52 questões
    
    if 35 <= num_bolhas <= 50:
        if debug:
            print("📋 Detectado cartão com 44 questões")
        return detectar_respostas_44_questoes(image_path, debug)
    else:
        if debug:
            print("📋 Detectado cartão com 52 questões")
        return detectar_respostas_52_questoes(image_path, debug)

def detectar_respostas_por_tipo(image_path: str, num_questoes: int = 52, debug: bool = False, eh_gabarito: bool = False) -> list:
    """
    Função auxiliar que escolhe a detecção correta baseada no número de questões.
    
    Args:
        image_path: Caminho da imagem do cartão
        num_questoes: Número de questões (44 ou 52)
        debug: Se deve exibir informações de debug
        eh_gabarito: Se True, usa crop específico para gabaritos
        
    Returns:
        Lista com as respostas detectadas (A/B/C/D ou '?')
    """
    if num_questoes == 44:
        return detectar_respostas_44_questoes(image_path, debug=debug, eh_gabarito=eh_gabarito)
    else:
        return detectar_respostas_52_questoes(image_path, debug=debug, eh_gabarito=eh_gabarito)

# ===========================================
# SEÇÃO 3: GEMINI - ANÁLISE INTELIGENTE DE IMAGENS
# ===========================================

def configurar_gemini():
    """
    Configura o Gemini API usando a chave do arquivo .env.
    
    Returns:
        Model do Gemini (gemini-2.5-flash) ou None se houver erro
    """
    if not GEMINI_DISPONIVEL:
        print("❌ Gemini não está disponível")
        print("💡 Para instalar: pip install google-generativeai")
        return None
        
    try:
        # Configure sua API key do Gemini aqui
        # Obtenha em: https://makersuite.google.com/app/apikey
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        if not GEMINI_API_KEY:
            print("GEMINI_API_KEY não encontrado ou arquivo .env faltando")  # Substitua pela sua chave
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Testar conexão
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        print("✅ Gemini configurado com sucesso!")
        return model
        
    except Exception as e:
        print(f"❌ Erro ao configurar Gemini: {e}")
        return None

def converter_imagem_para_base64(image_path: str):
    """
    Converte imagem para objeto PIL Image para envio ao Gemini.
    
    Args:
        image_path: Caminho do arquivo de imagem
    
    Returns:
        PIL Image ou None em caso de erro
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            
        # Converter para PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        return image
        
    except Exception as e:
        print(f"❌ Erro ao converter imagem: {e}")
        return None

def extrair_cabecalho_com_gemini(model, image_path: str) -> Optional[dict]:
    """
    Usa Gemini Vision para extrair informações do cabeçalho do cartão resposta.
    
    Args:
        model: Instância do modelo Gemini configurado
        image_path: Caminho da imagem do cartão
    
    Returns:
        Dicionário com chaves 'escola', 'aluno', 'turma', 'nascimento' ou None se falhar
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
        - Ignore nomes como flamengo, santos, etc. (Todo e qualquer nome de time deve ser ignorado)
        - Nomes de personagens fictícios também deverão ser ignorados. (Naruto, Goku, etc.)

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
                    return dados
                else:
                    print("❌ JSON não tem todas as chaves necessárias")
                    return None
            else:
                print("❌ Não foi possível extrair JSON da resposta")
                return None
                
        except Exception as e:
            print(f"❌ Erro ao processar JSON do Gemini")
            return None
            
    except Exception as e:
        print(f"❌ Erro na extração do cabeçalho com Gemini")
        return None

def extrair_cabecalho_com_ocr_fallback(image_path: str) -> dict:
    """
    Função de fallback usando OCR tradicional (Tesseract) quando Gemini falha.
    
    Args:
        image_path: Caminho da imagem do cartão
    
    Returns:
        Dicionário com chaves 'escola', 'aluno', 'turma', 'nascimento' (pode conter 'N/A')
    """
    try:
        print("🔍 Usando OCR fallback com pré-processamento avançado...")
        
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Erro ao carregar imagem: {image_path}")
            return None
            
        # Converter para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Pegar apenas a parte superior da imagem (cabeçalho - 25%)
        header_region = gray[0:int(height * 0.25)]
        
        # ═══════════════════════════════════════════════════════════
        # PRÉ-PROCESSAMENTO AVANÇADO PARA MELHORAR OCR
        # ═══════════════════════════════════════════════════════════
        
        # 1. Redimensionar se a imagem for muito pequena (aumentar para pelo menos 2000px de largura)
        if header_region.shape[1] < 2000:
            scale = 2000 / header_region.shape[1]
            new_width = int(header_region.shape[1] * scale)
            new_height = int(header_region.shape[0] * scale)
            header_region = cv2.resize(header_region, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 2. Aplicar denoising (remover ruído)
        header_region = cv2.fastNlMeansDenoising(header_region, h=10)
        
        # 3. Equalização de histograma para melhorar contraste
        header_region = cv2.equalizeHist(header_region)
        
        # 4. Binarização adaptativa (melhor que threshold simples)
        header_region = cv2.adaptiveThreshold(
            header_region, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            21, 
            10
        )
        
        # 5. Operações morfológicas para limpar a imagem
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        header_region = cv2.morphologyEx(header_region, cv2.MORPH_CLOSE, kernel)
        
        # ═══════════════════════════════════════════════════════════
        # EXTRAIR TEXTO COM MÚLTIPLAS TENTATIVAS
        # ═══════════════════════════════════════════════════════════
        
        # Configuração otimizada para Tesseract
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÇÉÊÍÓÔÕÚàáâãçéêíóôõú0123456789/:- '
        
        texto_completo = pytesseract.image_to_string(
            header_region, 
            lang='por',
            config=custom_config
        )
        
        # Limpar texto extraído
        texto_completo = texto_completo.strip()
        
        # Debug: mostrar texto extraído
        print(f"📄 Texto OCR extraído:\n{texto_completo[:200] if len(texto_completo) > 200 else texto_completo}...")
        
        # Processar texto extraído
        linhas = texto_completo.split('\n')
        dados = {
            "escola": "N/A",
            "aluno": "N/A", 
            "turma": "N/A",
            "nascimento": "N/A"
        }
        
        # ═══════════════════════════════════════════════════════════
        # EXTRAIR DADOS COM LÓGICA MELHORADA
        # ═══════════════════════════════════════════════════════════
        
        for i, linha in enumerate(linhas):
            linha = linha.strip()
            if not linha or len(linha) < 2:
                continue
            
            # Limpar linha de caracteres estranhos
            linha_limpa = re.sub(r'[^\w\sÀ-ÿ/:-]', '', linha)
            linha_lower = linha_limpa.lower()
            
            # 1. ESCOLA - procurar linha com "escola" e pegar próxima linha se necessário
            if 'escola' in linha_lower or 'colegio' in linha_lower or 'colégio' in linha_lower:
                # Se a linha tem apenas o label, pegar próxima linha
                if len(linha_limpa) < 15 and i + 1 < len(linhas):
                    dados["escola"] = linhas[i + 1].strip()
                else:
                    # Remover o label "Escola:" se presente
                    escola = re.sub(r'(?i)escola\s*:?\s*', '', linha_limpa).strip()
                    if len(escola) > 3:
                        dados["escola"] = escola
            
            # 2. NOME DO ALUNO - procurar linha com "nome" ou "completo"
            elif any(palavra in linha_lower for palavra in ['nome', 'completo']) and 'escola' not in linha_lower:
                # Se a linha tem apenas o label, pegar próxima linha
                if len(linha_limpa) < 15 and i + 1 < len(linhas):
                    proximo = linhas[i + 1].strip()
                    # Validar que não é data nem número
                    if not re.match(r'^\d+[/\-]', proximo) and len(proximo) > 5:
                        dados["aluno"] = proximo
                else:
                    # Remover labels
                    nome = re.sub(r'(?i)(nome|completo)\s*:?\s*', '', linha_limpa).strip()
                    # Validar que parece um nome (tem letras, não é muito curto)
                    if len(nome) > 5 and re.search(r'[a-zA-ZÀ-ÿ]{3,}', nome):
                        # Remover números do nome
                        nome = re.sub(r'\d+', '', nome).strip()
                        if len(nome) > 3:
                            dados["aluno"] = nome
            
            # 3. TURMA - procurar padrão de turma (número + letra opcional)
            elif 'turma' in linha_lower or 'série' in linha_lower or 'ano' in linha_lower:
                # Procurar padrão tipo "9A", "5 B", "7º ano"
                match = re.search(r'(\d{1,2})\s*([A-Za-z])?', linha)
                if match:
                    turma = match.group(1)
                    if match.group(2):
                        turma += match.group(2).upper()
                    dados["turma"] = turma
            
            # 4. DATA DE NASCIMENTO - procurar padrão de data
            elif 'nascimento' in linha_lower or 'data' in linha_lower or re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', linha):
                # Procurar data DD/MM/YYYY ou DD/MM/YY
                match = re.search(r'(\d{1,2})\s*[/\-]\s*(\d{1,2})\s*[/\-]\s*(\d{2,4})', linha)
                if match:
                    dia, mes, ano = match.groups()
                    # Validar data
                    if 1 <= int(dia) <= 31 and 1 <= int(mes) <= 12:
                        if len(ano) == 2:
                            ano = "20" + ano if int(ano) < 50 else "19" + ano
                        dados["nascimento"] = f"{dia.zfill(2)}/{mes.zfill(2)}/{ano}"
        
        # ═══════════════════════════════════════════════════════════
        # FALLBACK: Se não encontrou nome do aluno, tentar pegar maior linha de texto
        # ═══════════════════════════════════════════════════════════
        if dados["aluno"] == "N/A":
            linhas_validas = []
            for linha in linhas:
                linha = linha.strip()
                # Filtrar linhas que parecem ser nomes (só letras e espaços, tamanho razoável)
                if (10 < len(linha) < 50 and 
                    re.match(r'^[A-Za-zÀ-ÿ\s]+$', linha) and
                    'escola' not in linha.lower() and
                    'nome' not in linha.lower()):
                    linhas_validas.append(linha)
            
            if linhas_validas:
                # Pegar a linha mais longa que parece ser um nome
                dados["aluno"] = max(linhas_validas, key=len)
        
        # Exibir resultado
        print(f"✅ OCR extraiu:")
        print(f"   🏫 Escola: {dados.get('escola', 'N/A')}")
        print(f"   👤 Aluno: {dados.get('aluno', 'N/A')}")
        print(f"   📚 Turma: {dados.get('turma', 'N/A')}")
        print(f"   📅 Nascimento: {dados.get('nascimento', 'N/A')}")
        
        return dados
        
    except Exception as e:
        print(f"❌ Erro no OCR fallback: {e}")
        import traceback
        traceback.print_exc()
        return None

def extrair_dados_completos_com_gemini(model, image_path: str, nome_arquivo: str = None) -> Optional[dict]:
    """
    🆕 OTIMIZADO - Extrai cabeçalho + detecta ano em UMA ÚNICA chamada ao Gemini
    
    Extrai tudo de uma vez:
    - Nome da escola
    - Nome do aluno
    - Turma
    - Data de nascimento
    - Ano escolar (5° ou 9° ano) → retorna 44 ou 52 questões
    
    Args:
        model: Instância do modelo Gemini configurado
        image_path: Caminho da imagem do cartão
        nome_arquivo: Nome do arquivo (opcional, para detecção adicional)
    
    Returns:
        Dicionário com chaves: 'escola', 'aluno', 'turma', 'nascimento', 'num_questoes'
        ou None se falhar
    """
    if not model:
        return None
    
    try:
        # Converter imagem
        image = converter_imagem_para_base64(image_path)
        if not image:
            return None
        
        # 🎯 PROMPT OTIMIZADO - Extrai tudo de uma vez
        prompt = """
        Analise esta imagem de cartão resposta e extraia as seguintes informações:

        1. NOME DA ESCOLA - procure por campos como "Nome da Escola:", "Escola:", etc.
        2. NOME DO ALUNO - procure por campos como "Nome completo:", "Nome:", "Aluno:", etc.
        3. TURMA - procure por campos como "Turma:", "Série:", "Ano:", etc.
        4. DATA DE NASCIMENTO - procure por campos como "Data de nascimento:", "Nascimento:", etc.
        5. ANO ESCOLAR - Procure cuidadosamente por texto que indique:
           - "5° ano" ou "5º ano" ou "quinto ano" → RESPONDA: "5ano"
           - "9° ano" ou "9º ano" ou "nono ano" → RESPONDA: "9ano"

        INSTRUÇÕES:
        - Extraia APENAS o conteúdo, SEM os rótulos
        - Se não encontrar, retorne "N/A"
        - Ignore títulos como "AVALIAÇÃO DIAGNÓSTICA", "CARTÃO-RESPOSTA"
        - Ignore nomes de times (Flamengo, Santos, etc.) e personagens (Naruto, Goku, etc.)
        - IMPORTANTE: Procure CUIDADOSAMENTE por "5° ano" ou "9° ano" no cabeçalho do cartão

        FORMATO DE RESPOSTA (JSON):
        {
            "escola": "nome da escola ou N/A",
            "aluno": "nome do aluno ou N/A",
            "turma": "turma ou N/A",
            "nascimento": "data ou N/A",
            "ano_escolar": "5ano ou 9ano ou N/A"
        }
        """
        
        # Gerar resposta
        response = model.generate_content([prompt, image])
        resposta_texto = response.text.strip()
        
        # Processar JSON
        import json
        import re
        
        json_match = re.search(r'\{.*\}', resposta_texto, re.DOTALL)
        if json_match:
            dados = json.loads(json_match.group())
            
            # Validar estrutura básica
            if not all(key in dados for key in ['escola', 'aluno', 'turma', 'nascimento']):
                return None
            
            # 🆕 CONVERTER "5ano"/"9ano" para número de questões
            ano_escolar = dados.get('ano_escolar', 'N/A')
            
            if '5' in str(ano_escolar):
                num_questoes = 44
                print(f"   ✅ Gemini detectou: 5° ano (44 questões)")
            elif '9' in str(ano_escolar):
                num_questoes = 52
                print(f"   ✅ Gemini detectou: 9° ano (52 questões)")
            else:
                # Fallback 1: tentar pelo nome do arquivo
                if nome_arquivo:
                    nome_lower = nome_arquivo.lower()
                    if '5ano' in nome_lower or '5' in nome_lower.split('_')[0]:
                        num_questoes = 44
                        print(f"   ✅ Detectado pelo nome do arquivo: 5° ano (44 questões)")
                    elif '9ano' in nome_lower or '9' in nome_lower.split('_')[0]:
                        num_questoes = 52
                        print(f"   ✅ Detectado pelo nome do arquivo: 9° ano (52 questões)")
                    else:
                        # Fallback 2: tentar pela turma
                        num_questoes = detectar_ano_por_turma(dados.get('turma', ''))
                else:
                    # Fallback: tentar pela turma
                    num_questoes = detectar_ano_por_turma(dados.get('turma', ''))
            
            # Adicionar número de questões ao resultado
            dados['num_questoes'] = num_questoes
            
            return dados
        
        return None
        
    except Exception as e:
        print(f"   ⚠️ Erro no Gemini otimizado: {e}")
        return None


def extrair_cabecalho_com_fallback(model, image_path, numero_aluno=None):
    """
    Função que tenta extrair dados com Gemini.
    Se falhar, retorna N/A para todos os campos, exceto o nome do aluno que será numerado.
    
    🆕 ATUALIZADO: Agora usa extração otimizada quando possível
    """
    # Tentar Gemini OTIMIZADO primeiro (extrai tudo de uma vez)
    if model:
        try:
            dados_completos = extrair_dados_completos_com_gemini(model, image_path)
            if dados_completos:
                # Retornar no formato antigo (sem num_questoes) para compatibilidade
                return {
                    "escola": dados_completos.get("escola", "N/A"),
                    "aluno": dados_completos.get("aluno", "N/A"),
                    "turma": dados_completos.get("turma", "N/A"),
                    "nascimento": dados_completos.get("nascimento", "N/A")
                }
        except Exception as e:
            pass  # Tentar método antigo
    
    # Fallback: tentar método antigo
    if model:
        try:
            dados_gemini = extrair_cabecalho_com_gemini(model, image_path)
            if dados_gemini:
                return dados_gemini
        except Exception as e:
            pass  # Silenciar erro do Gemini
    
    # Se Gemini falhar, retornar dados com numeração do aluno
    nome_aluno = f"Aluno {numero_aluno}" if numero_aluno else "N/A"
    return {
        "escola": "N/A",
        "aluno": nome_aluno,
        "turma": "N/A",
        "nascimento": "N/A"
    }

def detectar_ano_por_turma(turma: str) -> int:
    """
    Detecta se o aluno é do 5° ou 9° ano baseado na informação de turma.
    
    REGRA SIMPLES: 
    - Se a turma COMEÇAR COM 5 → 5° ano (44 questões)
    - Se a turma COMEÇAR COM 9 → 9° ano (52 questões)
    - Se a turma CONTIVER 5 em qualquer lugar → 5° ano (44 questões)
    - Se a turma CONTIVER 9 em qualquer lugar → 9° ano (52 questões)
    
    Args:
        turma: String contendo informação da turma (ex: "9A", "5° ano do Ensino Fundamental")
    
    Returns:
        44 (para 5° ano) ou 52 (para 9° ano)
        Padrão: 52 se não conseguir detectar
    """
    if not turma or turma == "N/A" or str(turma).strip() == "":
        print("   ⚠️ ATENÇÃO: Turma não detectada (N/A) - usando padrão: 52 questões (9° ano)")
        print("   💡 Certifique-se de que o campo 'TURMA' está visível no cartão!")
        return 52
    
    turma_str = str(turma).strip()
    
    print(f"   🔍 Analisando turma: '{turma_str}'")
    
    # 🎯 REGRA 1: Se COMEÇAR com 5 → 5° ano (prioridade máxima)
    if re.match(r'^5', turma_str, re.IGNORECASE):
        print(f"   ✅ DETECTADO: Turma começa com '5' → 5° ano (44 questões)")
        return 44
    
    # 🎯 REGRA 2: Se COMEÇAR com 9 → 9° ano (prioridade máxima)
    if re.match(r'^9', turma_str, re.IGNORECASE):
        print(f"   ✅ DETECTADO: Turma começa com '9' → 9° ano (52 questões)")
        return 52
    
    # 🎯 REGRA 3: Se CONTÉM "5" em qualquer lugar → 5° ano
    if re.search(r'5', turma_str, re.IGNORECASE):
        print(f"   ✅ DETECTADO: Turma contém '5' → 5° ano (44 questões)")
        return 44
    
    # 🎯 REGRA 4: Se CONTÉM "9" em qualquer lugar → 9° ano
    if re.search(r'9', turma_str, re.IGNORECASE):
        print(f"   ✅ DETECTADO: Turma contém '9' → 9° ano (52 questões)")
        return 52
    
    # 🎯 REGRA 5: Palavras por extenso
    turma_lower = turma_str.lower()
    if 'quinto' in turma_lower or 'quint' in turma_lower:
        print(f"   ✅ DETECTADO: Palavra 'quinto' → 5° ano (44 questões)")
        return 44
    
    if 'nono' in turma_lower or 'non' in turma_lower:
        print(f"   ✅ DETECTADO: Palavra 'nono' → 9° ano (52 questões)")
        return 52
    
    # Se não detectar nada, usar padrão (52)
    print(f"   ⚠️ NÃO RECONHECIDO: Nenhum indicador de ano encontrado em '{turma}' - usando padrão: 52 questões")
    print(f"   💡 Turma detectada: '{turma_str}' - Verifique se contém '5' ou '9'")
    return 52


def detectar_ano_com_ocr_direto(image_path: str, debug: bool = False) -> int:
    """
    🆕 DETECÇÃO DIRETA POR OCR - FALLBACK quando Gemini falhar!
    
    Lê diretamente a área do cabeçalho onde está escrito:
    "9° ano do Ensino Fundamental" ou "5° ano do Ensino Fundamental"
    
    ⚠️ IMPORTANTE: Esta função só deve ser chamada quando o Gemini falhar!
    O Gemini é a solução principal. OCR é apenas backup.
    
    Args:
        image_path: Caminho da imagem do cartão
        debug: Se deve exibir informações de debug
        
    Returns:
        44 (para 5° ano) ou 52 (para 9° ano)
    """
    try:
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            print(f"   ⚠️ Erro ao carregar imagem para OCR direto")
            return 52
        
        height, width = img.shape[:2]
        
        # 📍 CROP DA ÁREA DO BOX SUPERIOR DIREITO
        # Onde está escrito "Agosto/2025 | 9° ano | do Ensino Fundamental"
        # Área aproximada: Top 3-15% da altura, Right 60-100% da largura
        crop_box = img[int(height*0.03):int(height*0.15), int(width*0.60):int(width*1.0)]
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(crop_box, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold para melhorar OCR
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Aplicar OCR com configuração otimizada para texto em linha
        texto_ocr = pytesseract.image_to_string(thresh, lang='por', config='--psm 6')
        texto_limpo = texto_ocr.strip().lower()
        
        if debug:
            print(f"   🔍 OCR (FALLBACK) detectou no cabeçalho: '{texto_ocr.strip()}'")
        else:
            print(f"   🔍 OCR (FALLBACK) analisando cabeçalho...")
        
        # 🎯 DETECÇÃO ESPECÍFICA - Priorizar padrões completos "5° ano" ou "9° ano"
        
        # PRIORIDADE 1: Detectar "5° ano" ou "5º ano" com símbolos de grau
        if re.search(r'5[°ºª]\s*ano', texto_limpo, re.IGNORECASE):
            print(f"   ✅ OCR (FALLBACK): Detectado '5° ano' no cabeçalho → 44 questões")
            return 44
        
        # PRIORIDADE 2: Detectar "9° ano" ou "9º ano" com símbolos de grau
        if re.search(r'9[°ºª]\s*ano', texto_limpo, re.IGNORECASE):
            print(f"   ✅ OCR (FALLBACK): Detectado '9° ano' no cabeçalho → 52 questões")
            return 52
        
        # PRIORIDADE 3: Detectar palavras por extenso
        if re.search(r'quint[oa]?\s+ano', texto_limpo, re.IGNORECASE):
            print(f"   ✅ OCR (FALLBACK): Detectado 'quinto ano' no cabeçalho → 44 questões")
            return 44
        
        if re.search(r'non[oa]?\s+ano', texto_limpo, re.IGNORECASE):
            print(f"   ✅ OCR (FALLBACK): Detectado 'nono ano' no cabeçalho → 52 questões")
            return 52
        
        # PRIORIDADE 4: Apenas números seguidos de "ano" (mais permissivo)
        if re.search(r'\b5\s+ano', texto_limpo, re.IGNORECASE):
            print(f"   ✅ OCR (FALLBACK): Detectado '5 ano' no cabeçalho → 44 questões")
            return 44
        
        if re.search(r'\b9\s+ano', texto_limpo, re.IGNORECASE):
            print(f"   ✅ OCR (FALLBACK): Detectado '9 ano' no cabeçalho → 52 questões")
            return 52
        
        # PADRÃO: Se nada for detectado, usar 52 questões (9° ano)
        print(f"   ⚠️ OCR (FALLBACK): Não conseguiu detectar '5° ano' ou '9° ano' no cabeçalho")
        print(f"   ℹ️  Texto detectado: '{texto_limpo[:100]}'")  # Mostrar primeiros 100 chars
        print(f"   🎯 Usando padrão: 52 questões (9° ano)")
        return 52
        
    except Exception as e:
        print(f"   ❌ Erro no OCR direto: {e}")
        return 52



def carregar_gabaritos_automatico(pasta_gabaritos: str = ".", debug: bool = False) -> dict:
    """
    Carrega AMBOS os gabaritos (44 e 52 questões) automaticamente.
    
    Procura por arquivos de IMAGEM com os seguintes padrões:
    - gabarito_44.png/.jpg/.jpeg ou gabarito44.png/.jpg/.jpeg → Gabarito de 44 questões (5° ano)
    - gabarito_52.png/.jpg/.jpeg ou gabarito52.png/.jpg/.jpeg → Gabarito de 52 questões (9° ano)
    
    Args:
        pasta_gabaritos: Diretório onde estão os gabaritos
        debug: Se deve mostrar informações de debug
    
    Returns:
        Dicionário com os gabaritos processados:
        {
            44: {
                'arquivo': 'gabarito_44.pdf',
                'caminho': '/path/to/gabarito_44.pdf',
                'imagem': '/path/to/gabarito_44_processed.png',
                'respostas': ['A', 'B', 'C', ...]
            },
            52: {
                'arquivo': 'gabarito_52.pdf',
                'caminho': '/path/to/gabarito_52.pdf',
                'imagem': '/path/to/gabarito_52_processed.png',
                'respostas': ['A', 'B', 'C', ...]
            }
        }
    """
    print("\n" + "=" * 80)
    print("📚 SISTEMA AUTOMATIZADO - CARREGANDO GABARITOS")
    print("=" * 80)
    print("ℹ️  O sistema carregará automaticamente 2 gabaritos (PNG/JPG):")
    print("   • gabarito_44.png/jpg (ou gabarito44.png/jpg) → 5° ano (44 questões)")
    print("   • gabarito_52.png/jpg (ou gabarito52.png/jpg) → 9° ano (52 questões)")
    print("=" * 80)
    
    gabaritos = {}
    
    # Listar arquivos na pasta
    try:
        arquivos = os.listdir(pasta_gabaritos)
    except Exception as e:
        print(f"❌ Erro ao listar pasta '{pasta_gabaritos}': {e}")
        return {}
    
    # Procurar gabaritos
    gabarito_44_file = None
    gabarito_52_file = None
    
    # Extensões de imagem aceitas para gabaritos
    EXTENSOES_IMAGEM = ('.png', '.jpg', '.jpeg')
    
    for arquivo in arquivos:
        arquivo_lower = arquivo.lower()
        
        # Procurar gabarito de 44 questões (APENAS IMAGENS)
        if ('gabarito_44' in arquivo_lower or 'gabarito44' in arquivo_lower) and \
           arquivo_lower.endswith(EXTENSOES_IMAGEM):
            gabarito_44_file = arquivo
            print(f"✅ Gabarito 44 questões encontrado: {arquivo}")
        
        # Procurar gabarito de 52 questões (APENAS IMAGENS)
        elif ('gabarito_52' in arquivo_lower or 'gabarito52' in arquivo_lower) and \
             arquivo_lower.endswith(EXTENSOES_IMAGEM):
            gabarito_52_file = arquivo
            print(f"✅ Gabarito 52 questões encontrado: {arquivo}")
    
    # Validar se ambos os gabaritos foram encontrados
    if not gabarito_44_file:
        print("❌ ERRO: Gabarito de 44 questões não encontrado!")
        print("   💡 Renomeie o arquivo de imagem para: gabarito_44.png, gabarito_44.jpg, gabarito44.png, etc.")
    
    if not gabarito_52_file:
        print("❌ ERRO: Gabarito de 52 questões não encontrado!")
        print("   💡 Renomeie o arquivo de imagem para: gabarito_52.png, gabarito_52.jpg, gabarito52.png, etc.")
    
    if not gabarito_44_file or not gabarito_52_file:
        print("\n⚠️ Sistema automatizado requer AMBOS os gabaritos!")
        return {}
    
    # Processar gabarito de 44 questões
    print(f"\n🔄 Processando gabarito de 44 questões...")
    try:
        caminho_44 = os.path.join(pasta_gabaritos, gabarito_44_file)
        img_44 = preprocessar_arquivo(caminho_44, "gabarito_44")
        respostas_44 = detectar_respostas_por_tipo(img_44, num_questoes=44, debug=debug, eh_gabarito=True)
        
        questoes_detectadas_44 = sum(1 for r in respostas_44 if r != '?')
        print(f"✅ Gabarito 44: {questoes_detectadas_44}/44 questões detectadas")
        
        gabaritos[44] = {
            'arquivo': gabarito_44_file,
            'caminho': caminho_44,
            'imagem': img_44,
            'respostas': respostas_44,
            'questoes_detectadas': questoes_detectadas_44
        }
        
        if debug:
            print(f"\n📋 Gabarito 44 questões:")
            exibir_gabarito_simples(respostas_44)
            
    except Exception as e:
        print(f"❌ Erro ao processar gabarito de 44 questões: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    # Processar gabarito de 52 questões
    print(f"\n🔄 Processando gabarito de 52 questões...")
    try:
        caminho_52 = os.path.join(pasta_gabaritos, gabarito_52_file)
        img_52 = preprocessar_arquivo(caminho_52, "gabarito_52")
        respostas_52 = detectar_respostas_por_tipo(img_52, num_questoes=52, debug=debug, eh_gabarito=True)
        
        questoes_detectadas_52 = sum(1 for r in respostas_52 if r != '?')
        print(f"✅ Gabarito 52: {questoes_detectadas_52}/52 questões detectadas")
        
        gabaritos[52] = {
            'arquivo': gabarito_52_file,
            'caminho': caminho_52,
            'imagem': img_52,
            'respostas': respostas_52,
            'questoes_detectadas': questoes_detectadas_52
        }
        
        if debug:
            print(f"\n📋 Gabarito 52 questões:")
            exibir_gabarito_simples(respostas_52)
            
    except Exception as e:
        print(f"❌ Erro ao processar gabarito de 52 questões: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    print(f"\n{'=' * 80}")
    print("✅ AMBOS OS GABARITOS CARREGADOS COM SUCESSO!")
    print(f"   • 44 questões: {gabaritos[44]['questoes_detectadas']}/44 questões")
    print(f"   • 52 questões: {gabaritos[52]['questoes_detectadas']}/52 questões")
    print(f"{'=' * 80}\n")
    
    return gabaritos


def processar_cartoes_automatizado(
    diretorio: str,
    gabaritos: dict,
    usar_gemini: bool = True,
    enviar_para_sheets: bool = True,
    debug: bool = False
) -> list:
    """
    🆕 NOVA FUNÇÃO: Processa cartões de alunos com detecção automática de gabarito.
    
    Para cada cartão do aluno:
    1. Extrai dados do cabeçalho (nome, turma, escola, nascimento)
    2. Detecta automaticamente se é 5° ou 9° ano pela turma
    3. Seleciona o gabarito correto (44 ou 52 questões)
    4. Compara as respostas e calcula o resultado
    5. Envia para Google Sheets (se habilitado)
    
    Args:
        diretorio: Pasta contendo os cartões dos alunos (sem gabaritos)
        gabaritos: Dicionário retornado por carregar_gabaritos_automatico()
        usar_gemini: Se deve usar Gemini para extração de cabeçalho
        enviar_para_sheets: Se deve enviar resultados para Google Sheets
        debug: Se deve mostrar informações de debug
    
    Returns:
        Lista de resultados de cada aluno processado
    """
    print("\n" + "=" * 80)
    print("🤖 PROCESSAMENTO AUTOMATIZADO DE CARTÕES")
    print("=" * 80)
    
    # Validar gabaritos
    if not gabaritos or 44 not in gabaritos or 52 not in gabaritos:
        print("❌ ERRO: Gabaritos não foram carregados corretamente!")
        return []
    
    # Configurar Gemini
    model_gemini = None
    if usar_gemini:
        try:
            model_gemini = configurar_gemini()
            print("✅ Gemini configurado")
        except Exception as e:
            print(f"⚠️ Gemini indisponível: {e}")
            usar_gemini = False
    
    # Configurar Google Sheets
    client_sheets = None
    if enviar_para_sheets:
        try:
            client_sheets = configurar_google_sheets()
            print("✅ Google Sheets configurado")
        except Exception as e:
            print(f"⚠️ Google Sheets indisponível: {e}")
            enviar_para_sheets = False
    
    # Listar arquivos de alunos
    arquivos = listar_arquivos_suportados(diretorio)
    
    # Filtrar apenas arquivos de alunos (sem gabaritos)
    arquivos_alunos = [
        f for f in arquivos['todos'] 
        if not f.lower().startswith('gabarito')
    ]
    
    if not arquivos_alunos:
        print("❌ Nenhum cartão de aluno encontrado!")
        return []
    
    print(f"\n👥 Encontrados {len(arquivos_alunos)} cartões para processar")
    print("=" * 80)
    
    resultados = []
    
    for i, arquivo_aluno in enumerate(arquivos_alunos, 1):
        print(f"\n🔄 [{i:02d}/{len(arquivos_alunos)}] {arquivo_aluno}")
        print("-" * 70)
        
        try:
            # 1. Preprocessar cartão
            caminho_aluno = os.path.join(diretorio, arquivo_aluno)
            img_aluno = preprocessar_arquivo(caminho_aluno, f"aluno_{i}")
            
            # 2. Extrair cabeçalho
            dados_aluno = {
                "aluno": f"Aluno {i}",
                "escola": "N/A",
                "turma": "N/A",
                "nascimento": "N/A"
            }
            
            if usar_gemini and model_gemini:
                try:
                    dados_extraidos = extrair_cabecalho_com_fallback(model_gemini, img_aluno, i)
                    if dados_extraidos:
                        dados_aluno.update(dados_extraidos)
                except Exception as e:
                    if debug:
                        print(f"   ⚠️ Erro no Gemini: {e}")
            
            print(f"   👤 Aluno: {dados_aluno['aluno']}")
            print(f"   📚 Turma: {dados_aluno['turma']}")
            print(f"   🏫 Escola: {dados_aluno['escola']}")
            
            # 3. Detectar ano automaticamente pela turma
            num_questoes = detectar_ano_por_turma(dados_aluno['turma'])
            
            # 4. Selecionar gabarito correto
            gabarito_selecionado = gabaritos[num_questoes]
            respostas_gabarito = gabarito_selecionado['respostas']
            
            print(f"   📋 Usando gabarito de {num_questoes} questões")
            
            # 5. Detectar respostas do aluno
            respostas_aluno = detectar_respostas_por_tipo(
                img_aluno, 
                num_questoes=num_questoes, 
                debug=debug
            )
            
            questoes_detectadas = sum(1 for r in respostas_aluno if r != '?')
            print(f"   ✓ Detectadas: {questoes_detectadas}/{num_questoes} questões")
            
            # 6. Comparar respostas
            resultado = comparar_respostas(respostas_gabarito, respostas_aluno)
            
            # 7. Exibir resultado
            print(f"\n   {'─'*60}")
            print(f"   ✅ Acertos: {resultado['acertos']}/{resultado['total']}")
            print(f"   ❌ Erros: {resultado['erros']}")
            print(f"   📊 Percentual: {resultado['percentual']:.1f}%")
            print(f"   {'─'*60}")
            
            # 8. Enviar para Google Sheets
            if enviar_para_sheets and client_sheets:
                try:
                    # Preparar dados para envio
                    dados_envio = {
                        "Escola": dados_aluno.get("escola", "N/A"),
                        "Aluno": dados_aluno.get("aluno", "N/A"),
                        "Nascimento": dados_aluno.get("nascimento", "N/A"),
                        "Turma": dados_aluno.get("turma", "N/A")
                    }
                    
                    # Planilha ID será escolhida automaticamente dentro da função
                    enviar_para_planilha(
                        client_sheets,
                        dados_envio,
                        resultado,
                        questoes_detectadas=questoes_detectadas
                    )
                    print(f"   ✓ Enviado para Google Sheets")
                except Exception as e:
                    print(f"   ⚠️ Erro ao enviar para Sheets: {e}")
            
            # 9. Armazenar resultado
            resultados.append({
                "arquivo": arquivo_aluno,
                "aluno": dados_aluno['aluno'],
                "turma": dados_aluno['turma'],
                "escola": dados_aluno['escola'],
                "nascimento": dados_aluno['nascimento'],
                "num_questoes": num_questoes,
                "questoes_detectadas": questoes_detectadas,
                "resultado": resultado
            })
            
        except Exception as e:
            print(f"   ❌ Erro ao processar: {e}")
            if debug:
                import traceback
                traceback.print_exc()
    
    # Resumo final
    print(f"\n{'=' * 80}")
    print("✅ PROCESSAMENTO CONCLUÍDO!")
    print(f"   Total processado: {len(resultados)}/{len(arquivos_alunos)}")
    
    if resultados:
        # Separar por ano
        alunos_44 = [r for r in resultados if r['num_questoes'] == 44]
        alunos_52 = [r for r in resultados if r['num_questoes'] == 52]
        
        print(f"   • 5° ano (44 questões): {len(alunos_44)} alunos")
        print(f"   • 9° ano (52 questões): {len(alunos_52)} alunos")
        
        # Estatísticas
        media_percentual = sum(r['resultado']['percentual'] for r in resultados) / len(resultados)
        print(f"\n   📊 Média geral: {media_percentual:.1f}%")
    
    print("=" * 80)
    
    return resultados

# ===========================================
# SEÇÃO 4: INTEGRAÇÃO GOOGLE DRIVE & SHEETS
# ===========================================

def carregar_credenciais(scopes: List[str]) -> Optional[Credentials]:
    """
    Carrega credenciais do Google Service Account do arquivo JSON.
    
    Args:
        scopes: Lista de escopos de permissão do Google API
    
    Returns:
        Objeto Credentials ou None se houver erro
    """
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
    """
    Configura conexão com Google Sheets usando gspread.
    
    Returns:
        Cliente gspread autorizado ou None se houver erro
    """
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
    """
    Configura conexão com Google Drive e retorna serviço da API v3.
    
    Args:
        scopes: Lista de escopos de permissão (padrão: readonly)
    
    Returns:
        Objeto service do Google Drive API ou None se houver erro
    """
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
    """
    Configura conexão com Google Drive com permissões completas (escrita e movimentação).
    
    Returns:
        Objeto service do Google Drive API com permissões completas ou None se houver erro
    """
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
    """
    Verifica acesso à pasta 'cartoes-processados' no Google Drive.
    
    Args:
        service: Objeto service do Google Drive API
        pasta_origem_id: ID da pasta de origem (não utilizado, mantido por compatibilidade)
    
    Returns:
        ID da pasta 'cartoes-processados' ou None se houver erro
    """
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
    """
    Move um arquivo de uma pasta para outra no Google Drive.
    
    Args:
        service: Objeto service do Google Drive API
        arquivo_id: ID do arquivo a ser movido
        pasta_origem_id: ID da pasta de origem (usado para logs)
        pasta_destino_id: ID da pasta de destino
        nome_arquivo: Nome do arquivo (usado para logs)
    
    Returns:
        True se movido com sucesso, False caso contrário
    """
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
    """
    Obtém metadados de todos os arquivos da pasta do Google Drive.
    
    Args:
        service: Objeto service do Google Drive API
        pasta_id: ID da pasta no Google Drive
    
    Returns:
        Dicionário mapeando arquivo_id para metadados (id, nome, mimeType, modifiedTime)
    """
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

def mover_arquivos_processados_drive(service, pasta_origem_id: str, metadados: dict, pasta_destino_id: str):
    """Move arquivos processados (exceto gabarito) da pasta de upload para a pasta de destino."""
    try:
        # Configurar serviço com permissões completas
        service_completo = configurar_google_drive_service_completo()
        if not service_completo:
            print("❌ Não foi possível obter permissões para mover arquivos")
            return

        arquivos_movidos = 0
        
        # Mover todos os arquivos exceto o gabarito
        for nome_arquivo, dados in metadados.items():
            # Pular arquivo de gabarito
            if nome_arquivo.lower().startswith('gabarito'):
                print(f"⏭️ Gabarito ignorado: {nome_arquivo}")
                continue
            
            # Mover arquivo
            print(f"📦 Movendo: {nome_arquivo}...")
            if mover_arquivo_no_drive(
                service_completo, 
                dados['id'], 
                pasta_origem_id, 
                pasta_destino_id, 
                nome_arquivo
            ):
                arquivos_movidos += 1
                print(f"   ✅ Movido com sucesso!")
            else:
                print(f"   ❌ Falha ao mover")
        
        print(f"\n✅ Total: {arquivos_movidos} arquivos movidos para a pasta de destino no Drive")
        
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


def baixar_cartoes_da_pasta_drive(service, pasta_id: str, destino: str, formatos_validos: Optional[Dict[str, str]] = None, converter_pb: bool = True, threshold_pb: int = 180) -> List[str]:
    """
    Baixa todos os cartões (gabarito + alunos) de uma pasta do Google Drive.
    
    Args:
        service: Serviço do Google Drive
        pasta_id: ID da pasta no Drive
        destino: Diretório de destino
        formatos_validos: Dicionário de MIME types válidos
        converter_pb: Se deve converter imagens para preto e branco (padrão: True)
        threshold_pb: Threshold para conversão P&B (padrão: 180)
        
    Returns:
        Lista de caminhos dos arquivos baixados (convertidos se converter_pb=True)
    """
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

                # CONVERSÃO AUTOMÁTICA PARA PRETO E BRANCO
                # ⚠️ NÃO CONVERTER PDFs - eles serão processados separadamente
                eh_pdf = caminho_destino.lower().endswith('.pdf')
                eh_gabarito = nome_original.lower().startswith('gabarito')
                
                if converter_pb and not eh_gabarito and not eh_pdf:
                    print(f"   🎨 Convertendo para P&B (threshold={threshold_pb})...")
                    try:
                        caminho_pb = converter_para_preto_e_branco(
                            caminho_destino, 
                            threshold=threshold_pb, 
                            salvar=True
                        )
                        # Usar imagem convertida ao invés da original
                        if caminho_pb and os.path.exists(caminho_pb):
                            # Remover original e renomear convertida
                            os.remove(caminho_destino)
                            os.rename(caminho_pb, caminho_destino)
                            print(f"   ✅ Convertido para P&B")
                    except Exception as e:
                        print(f"   ⚠️ Erro na conversão P&B: {e} - usando original")
                elif eh_pdf:
                    print(f"   📄 PDF detectado - será processado separadamente")
                
                arquivos_baixados.append(caminho_destino)
                print(f"   📝 Arquivo adicionado à lista: {os.path.basename(caminho_destino)} (extensão: {os.path.splitext(caminho_destino)[1]})")

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
    pasta_destino_id: str = None,
    usar_gemini: bool = True,
    debug_mode: bool = False,
    enviar_para_sheets: bool = True,
    manter_pasta_temporaria: bool = False,
    mover_processados: bool = True,
    apenas_gabarito: bool = False,
    converter_pb: bool = True,
    threshold_pb: int = 180,
    num_questoes: int = 52
):
    """
    Workflow completo: baixa do Drive, converte para P&B, processa cartões e envia resultados.
    
    Args:
        pasta_id: ID da pasta no Google Drive (origem/upload)
        pasta_destino_id: ID da pasta de destino (5º ou 9º ano)
        usar_gemini: Se deve usar Gemini para extração de dados
        debug_mode: Se deve exibir informações de debug
        enviar_para_sheets: Se deve enviar resultados para Google Sheets
        manter_pasta_temporaria: Se deve manter arquivos temporários
        mover_processados: Se deve mover arquivos processados no Drive
        apenas_gabarito: Se deve processar apenas o gabarito
        converter_pb: Se deve converter imagens para preto e branco (padrão: True)
        threshold_pb: Threshold para conversão P&B, 0-255 (padrão: 180)
        num_questoes: Tipo de cartão (44 ou 52 questões)
    """

    service = configurar_google_drive_service()
    if not service:
        print("❌ Não foi possível configurar o Google Drive. Abortando.")
        return []

    pasta_temporaria = tempfile.mkdtemp(prefix="cartoes_drive_")
    print(f"📁 Pasta temporária criada: {pasta_temporaria}")
    
    if converter_pb:
        print(f"🎨 Conversão P&B habilitada (threshold={threshold_pb})")
        print(f"   ℹ️ Gabaritos serão mantidos originais")
        print(f"   ℹ️ Cartões de alunos serão convertidos automaticamente")

    try:
        # Obter metadados dos arquivos durante o download
        arquivos_metadata = obter_metadados_pasta_drive(service, pasta_id)
        
        # Baixar e converter automaticamente para P&B
        arquivos_baixados = baixar_cartoes_da_pasta_drive(
            service, 
            pasta_id, 
            pasta_temporaria,
            converter_pb=converter_pb,
            threshold_pb=threshold_pb
        )
        
        if not arquivos_baixados:
            print("❌ Nenhum arquivo válido foi baixado do Drive.")
            return []

        # 🆕 DETECTAR PDFs COM MÚLTIPLAS PÁGINAS
        # Verificar extensão do arquivo (caminho completo ou apenas nome)
        pdfs_multiplos = []
        arquivos_individuais = []
        
        for arquivo in arquivos_baixados:
            # Pegar apenas o nome do arquivo (não o caminho completo)
            nome_arquivo = os.path.basename(arquivo) if os.path.isabs(arquivo) else arquivo
            
            if nome_arquivo.lower().endswith('.pdf'):
                pdfs_multiplos.append(arquivo)
            else:
                arquivos_individuais.append(arquivo)
        
        # Debug: Mostrar o que foi detectado
        print(f"\n🔍 Arquivos baixados: {len(arquivos_baixados)}")
        print(f"   📄 PDFs: {len(pdfs_multiplos)}")
        print(f"   🖼️ Imagens: {len(arquivos_individuais)}")
        
        if pdfs_multiplos:
            print(f"\n{'='*80}")
            print(f"📄 DETECTADOS {len(pdfs_multiplos)} PDF(s) - Processando múltiplas páginas")
            print(f"{'='*80}")
            
            resultados_pdfs = []
            
            for pdf_file in pdfs_multiplos:
                # Construir caminho completo se necessário
                if os.path.isabs(pdf_file):
                    pdf_path = pdf_file
                else:
                    pdf_path = os.path.join(pasta_temporaria, pdf_file)
                
                print(f"\n📑 Processando PDF: {os.path.basename(pdf_path)}")
                print(f"   Caminho: {pdf_path}")
                print(f"   Existe? {os.path.exists(pdf_path)}")
                
                if not os.path.exists(pdf_path):
                    print(f"❌ ERRO: Arquivo não encontrado: {pdf_path}")
                    continue
                
                try:
                    # Processar PDF com múltiplas páginas
                    resultados_pdf = processar_pdf_multiplas_paginas(
                        pdf_path=pdf_path,
                        num_questoes=num_questoes,
                        usar_gemini=usar_gemini,
                        debug_mode=debug_mode,
                        enviar_para_sheets=enviar_para_sheets,
                        mover_para_drive=False,  # Mover manualmente depois
                        pasta_destino_id=pasta_destino_id
                    )
                    
                    if resultados_pdf:
                        resultados_pdfs.extend(resultados_pdf)
                        print(f"✅ PDF processado: {len(resultados_pdf)} cartões")
                    else:
                        print(f"⚠️ Nenhum cartão processado do PDF")
                        
                except Exception as e:
                    print(f"❌ ERRO ao processar PDF {os.path.basename(pdf_path)}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Se processou PDFs e teve sucesso, mover para pasta processada
            if resultados_pdfs and mover_processados and pasta_destino_id:
                print(f"\n📦 Movendo PDFs processados no Google Drive...")
                # Filtrar metadata apenas dos PDFs processados (arquivos_metadata é um dict)
                pdf_metadata = {nome: meta for nome, meta in arquivos_metadata.items() 
                               if nome.lower().endswith('.pdf')}
                mover_arquivos_processados_drive(service, pasta_id, pdf_metadata, pasta_destino_id)
                
            # 🆕 IMPORTANTE: Retornar resultados dos PDFs e PARAR aqui
            # Não processar PDFs novamente como arquivos individuais
            print(f"\n✅ Processamento de PDFs concluído!")
            print(f"   Total de cartões processados: {len(resultados_pdfs)}")
            print(f"\n{'='*80}")
            return resultados_pdfs

        # Se é apenas para gabarito, retornar o diretório temporário
        if apenas_gabarito:
            return pasta_temporaria

        # 📝 PROCESSAR ARQUIVOS INDIVIDUAIS (imagens normais)
        if arquivos_individuais:
            print(f"\n{'='*80}")
            print(f"🖼️ Processando {len(arquivos_individuais)} arquivos individuais")
            print(f"{'='*80}")
            
            if enviar_para_sheets:
                resultados = processar_pasta_gabaritos_com_sheets(
                    pasta_temporaria,
                    usar_gemini=usar_gemini,
                    debug_mode=debug_mode,
                    num_questoes=num_questoes
                )
            else:
                resultados = processar_pasta_gabaritos_sem_sheets(
                    pasta_temporaria,
                    usar_gemini=usar_gemini,
                    debug_mode=debug_mode,
                    num_questoes=num_questoes
                )

            # Mover arquivos processados se houve sucesso e está habilitado
            if resultados and mover_processados and pasta_destino_id:
                print(f"\n📦 Movendo arquivos individuais processados no Google Drive...")
                # Filtrar metadata apenas dos arquivos individuais (arquivos_metadata é um dict)
                individual_metadata = {nome: meta for nome, meta in arquivos_metadata.items() 
                                      if not nome.lower().endswith('.pdf')}
                mover_arquivos_processados_drive(service, pasta_id, individual_metadata, pasta_destino_id)
            elif resultados and mover_processados and not pasta_destino_id:
                print(f"\n⚠️ Pasta de destino não informada. Arquivos não serão movidos.")

            # Combinar resultados de PDFs + individuais (se houver ambos)
            if pdfs_multiplos and 'resultados_pdfs' in locals():
                return resultados_pdfs + resultados
            
            return resultados
        
        # Se só tinha PDFs, retornar seus resultados
        if pdfs_multiplos and 'resultados_pdfs' in locals():
            return resultados_pdfs
        
        return []

    finally:
        if manter_pasta_temporaria:
            print(f"🗂️ Mantendo pasta temporária em: {pasta_temporaria}")
        else:
            shutil.rmtree(pasta_temporaria, ignore_errors=True)

def enviar_para_planilha(client, dados_aluno, resultado_comparacao, planilha_id=None, questoes_detectadas=None):
    """Envia dados para Google Sheets"""

    try:
        # 👉 Determinar número total de questões (incluindo anuladas)
        total_questoes = resultado_comparacao.get("total", 0)
        questoes_validas = resultado_comparacao.get("questoes_validas", total_questoes)
        anuladas = resultado_comparacao.get("anuladas", 0)

        # 👉 Definir IDs fixos das planilhas
        GOOGLE_SHEETS_9ANO = "1VJ0_w9eoQcc-ouBnRoq5lFQdR2fVZkqEtR-KArZMuvk"
        GOOGLE_SHEETS_5ANO = "1DISO8jgKt4FQe2ha9v3kAgMUvoz9WI1HLO67xcsHEXg"

        # 👉 Escolher a planilha com base no número de questões
        if total_questoes == 44:
            planilha_id = GOOGLE_SHEETS_5ANO
            print("📄 Enviando para planilha de 44 questões...")
        elif total_questoes == 52:
            planilha_id = GOOGLE_SHEETS_9ANO
            print("📄 Enviando para planilha de 52 questões...")
        else:
            print(f"⚠️ Número de questões ({total_questoes}) não reconhecido. Registro ignorado.")
            return False

        # 👉 Abrir a planilha correta
        sheet = client.open_by_key(planilha_id)
        worksheet = sheet.sheet1
        
        # Verificar se há cabeçalho
        if not worksheet.get_all_values():
            cabecalho = [
                "Data", "Escola", "Nome completo", "Nascimento", "Turma", "Acertos Língua Portuguesa", "Acertos Matemática", "Erros Língua Portuguesa", "Erros Matemática", "Anuladas", "Porcentagem"
            ]
            worksheet.append_row(cabecalho)
            print("📋 Cabeçalho criado na planilha")
        
        # Preparar dados completos
        agora = datetime.now().strftime("%d/%m/%Y")
        
        # Garantir que os dados estejam no formato correto
        escola = dados_aluno.get("Escola", "N/A")
        if escola == "N/A" or not escola.strip().lower():
            escola = "N/A"
        else:
            escola = escola.lower()  # Converter para minúsculas
        
        aluno = dados_aluno.get("Aluno", "N/A")
        if aluno == "N/A" or not aluno.strip().lower():
            aluno = "N/A"
        else:
            aluno = aluno.lower()  # Converter para minúsculas
            
        nascimento = dados_aluno.get("Nascimento", "N/A")
        if nascimento == "N/A" or not nascimento.strip().lower():
            nascimento = "N/A"
            
        turma = dados_aluno.get("Turma", "N/A")
        if turma == "N/A" or not turma.strip().lower():
            turma = "N/A"
        else:
            turma = turma.lower()  # Converter para minúsculas
        
        linha_dados = [
            agora,
            escola,
            aluno, 
            nascimento,
            turma,
            resultado_comparacao.get("acertos_portugues", 0),
            resultado_comparacao.get("acertos_matematica", 0),
            resultado_comparacao.get("erros_portugues", 0),
            resultado_comparacao.get("erros_matematica", 0),
            resultado_comparacao.get("anuladas", 0),
            f"{resultado_comparacao['percentual']:.1f}%"
        ]
        
        # Adicionar linha
        worksheet.append_row(linha_dados)
        print(f"📊 Registro adicionado:")
        print(f"   🏫 Escola: {escola}")
        print(f"   👤 Aluno: {aluno}")
        print(f"   📅 Nascimento: {nascimento}")
        print(f"   📚 Turma: {turma}")
        if resultado_comparacao.get("anuladas", 0) > 0:
            print(f"   📊 Resultado: ✓ {resultado_comparacao.get('acertos_portugues', 0)}PT/{resultado_comparacao.get('acertos_matematica', 0)}MT | ✗ {resultado_comparacao.get('erros_portugues', 0)}PT/{resultado_comparacao.get('erros_matematica', 0)}MT | {resultado_comparacao['anuladas']} anuladas | {resultado_comparacao['percentual']:.1f}%")
        else:
            print(f"   📊 Resultado: ✓ {resultado_comparacao.get('acertos_portugues', 0)}PT/{resultado_comparacao.get('acertos_matematica', 0)}MT | ✗ {resultado_comparacao.get('erros_portugues', 0)}PT/{resultado_comparacao.get('erros_matematica', 0)}MT | {resultado_comparacao['percentual']:.1f}%")
        
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
    anuladas = 0
    detalhes = []
    
    # Contadores separados para português e matemática
    acertos_portugues = 0
    acertos_matematica = 0
    erros_portugues = 0
    erros_matematica = 0
    
    # Determinar número de questões por coluna baseado no total
    # Para 52 questões: 13 por coluna
    # Para 44 questões: 11 por coluna
    questoes_por_coluna = 13 if min_questoes == 52 else 11
    
    for i in range(min_questoes):
        questao = i + 1
        gabarito = respostas_gabarito[i] if i < len(respostas_gabarito) else "N/A"
        aluno = respostas_aluno[i] if i < len(respostas_aluno) else "N/A"
        
        # Determinar se é questão de português ou matemática
        # Colunas: 1ª português, 2ª matemática, 3ª português, 4ª matemática
        coluna = i // questoes_por_coluna  # 0, 1, 2, 3
        eh_portugues = (coluna == 0 or coluna == 2)  # Colunas 1 e 3
        
        # 🔧 Se gabarito ou aluno tem '?', anular questão (não conta no cálculo)
        if gabarito == '?' or aluno == '?':
            status = "⊘"  # Anulada
            anuladas += 1
            detalhes.append({
                "questao": questao,
                "gabarito": gabarito,
                "aluno": aluno,
                "status": "ANULADA",
                "disciplina": "Português" if eh_portugues else "Matemática"
            })
            continue
        
        if gabarito == aluno:
            status = "✓"
            acertos += 1
            # Contar acerto na disciplina correspondente
            if eh_portugues:
                acertos_portugues += 1
            else:
                acertos_matematica += 1
        else:
            status = "✗"
            erros += 1
            # Contar erro na disciplina correspondente
            if eh_portugues:
                erros_portugues += 1
            else:
                erros_matematica += 1
        
        detalhes.append({
            "questao": questao,
            "gabarito": gabarito,
            "aluno": aluno,
            "status": status,
            "disciplina": "Português" if eh_portugues else "Matemática"
        })
    
    # Calcular sobre questões válidas (excluindo anuladas)
    questoes_validas = min_questoes - anuladas
    percentual = (acertos / questoes_validas * 100) if questoes_validas > 0 else 0
    
    return {
        "total": min_questoes,
        "questoes_validas": questoes_validas,
        "anuladas": anuladas,
        "acertos": acertos,
        "acertos_portugues": acertos_portugues,
        "acertos_matematica": acertos_matematica,
        "erros": erros,
        "erros_portugues": erros_portugues,
        "erros_matematica": erros_matematica,
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
    print(f"Questões válidas: {resultado['questoes_validas']}")
    if resultado.get('anuladas', 0) > 0:
        print(f"Questões anuladas: {resultado['anuladas']} ⊘")
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

def processar_apenas_gabarito(DRIVER_FOLDER_9ANO: str = None, debug_mode: bool = False, num_questoes: int = 52):
    """Processa apenas o gabarito e exibe as respostas em formato simples"""
    print("📋 PROCESSANDO APENAS GABARITO")
    print("=" * 40)
    
    # Usar DRIVER_FOLDER_9ANOO do .env se não fornecido
    if not DRIVER_FOLDER_9ANOO:
        DRIVER_FOLDER_9ANOO = os.getenv('DRIVER_FOLDER_9ANOO')
        if not DRIVER_FOLDER_9ANOO:
            print("❌ DRIVER_FOLDER_9ANOO não encontrado no arquivo .env")
            return
    
    try:
        # Baixar arquivos do Google Drive
        print(f"📥 Baixando arquivos da pasta do Drive: {DRIVER_FOLDER_9ANOO}")
        diretorio_temp = baixar_e_processar_pasta_drive(
            pasta_id=DRIVER_FOLDER_9ANOO,
            usar_gemini=False,
            debug_mode=debug_mode,
            enviar_para_sheets=False,
            manter_pasta_temporaria=True,
            mover_processados=False,
            apenas_gabarito=True,
            num_questoes=num_questoes
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
        
        # Detectar respostas do gabarito usando o tipo específico (44 ou 52 questões) com crop de gabarito
        respostas_gabarito = detectar_respostas_por_tipo(gabarito_img, num_questoes=num_questoes, debug=debug_mode, eh_gabarito=True)
        
        questoes_gabarito = sum(1 for r in respostas_gabarito if r != '?')
        num_questoes = len(respostas_gabarito)
        print(f"✅ Gabarito processado: {questoes_gabarito}/{num_questoes} questões detectadas")
        
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

def processar_pasta_gabaritos(diretorio: str = "./gabaritos", usar_gemini: bool = True, debug_mode: bool = False, num_questoes: int = 52):
    """
    Processa todos os arquivos de uma pasta com cartões (gabarito + alunos)
    - 1 gabarito (template) para comparar com múltiplos alunos
    - Sem comparações desnecessárias de dados
    
    Args:
        diretorio: Caminho da pasta contendo gabarito e cartões dos alunos
        usar_gemini: Se deve usar Gemini para cabeçalho
        debug_mode: Se deve mostrar debug detalhado
        num_questoes: Tipo de cartão (44 ou 52 questões)
        
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
        
        # Detectar respostas do gabarito usando o tipo específico (44 ou 52 questões)
        respostas_gabarito = detectar_respostas_por_tipo(gabarito_img, num_questoes=num_questoes, debug=debug_mode, eh_gabarito=True)
        
        questoes_gabarito = sum(1 for r in respostas_gabarito if r != '?')
        num_questoes_detectadas = len(respostas_gabarito)
        print(f"✅ Gabarito processado: {questoes_gabarito}/{num_questoes_detectadas} questões detectadas")
        
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
                "Aluno": f"Aluno {i}",
                "Escola": "N/A",
                "Nascimento": "N/A", 
                "Turma": "N/A"
            }
            
            if usar_gemini and model_gemini:
                try:
                    dados_extraidos = extrair_cabecalho_com_fallback(model_gemini, aluno_img, numero_aluno=i)
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
                    print(f"⚠️ Gemini falhou, usando numeração automática")
            
            # Detectar respostas do aluno usando o tipo específico (44 ou 52 questões)
            if "page_" in aluno_img and (aluno_img.endswith(".png") or aluno_img.endswith(".jpg")):
                respostas_aluno = detectar_respostas_pdf(aluno_img, debug=debug_mode)
            else:
                respostas_aluno = detectar_respostas_por_tipo(aluno_img, num_questoes=num_questoes, debug=debug_mode)
            
            questoes_aluno = sum(1 for r in respostas_aluno if r != '?')
            
            # Calcular resultado
            resultado = comparar_respostas(respostas_gabarito, respostas_aluno)
            
            # Armazenar resultado com dados completos
            resultado_completo = {
                "arquivo": aluno_file,
                "dados_completos": dados_aluno,  # Dados completos do cabeçalho
                "acertos": resultado['acertos'],
                "acertos_portugues": resultado.get('acertos_portugues', 0),
                "acertos_matematica": resultado.get('acertos_matematica', 0),
                "total": resultado['total'],
                "percentual": resultado['percentual'],
                "questoes_detectadas": questoes_aluno
            }
            resultados_lote.append(resultado_completo)
            
            print(f"📊 Resultado: ✓ {resultado.get('acertos_portugues', 0)}PT/{resultado.get('acertos_matematica', 0)}MT | ✗ {resultado.get('erros_portugues', 0)}PT/{resultado.get('erros_matematica', 0)}MT | Total {resultado['acertos']}/{resultado['total']} ({resultado['percentual']:.1f}%)")
            
            # Delay de 10 segundos após processar cada cartão
            if i < len(arquivos_alunos):
                print(f"⏳ Aguardando 12 segundos antes do próximo cartão...")
                time.sleep(20)
            
        except Exception as e:
            print(f"❌ ERRO ao processar {aluno_file}: {e}")
            print(f"⏳ Aguardando 10 segundos antes do próximo cartão...")
            time.sleep(20)
            
        except Exception as e:
            print(f"❌ ERRO ao processar {aluno_file}: {e}")
            resultado_erro = {
                "arquivo": aluno_file,
                "dados_completos": {
                    "Aluno": f"Aluno {i}",
                    "Escola": "N/A",
                    "Nascimento": "N/A",
                    "Turma": "N/A"
                },
                "acertos": 0,
                "acertos_portugues": 0,
                "acertos_matematica": 0,
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
        print(f"\n=== TOTAL DE ALUNOS: {len(resultados_lote)} + RESULTADOS ===")
        
        # Ordenar por nome do aluno (ordem alfabética)
        resultados_ordenados = sorted(resultados_lote, key=lambda x: x["dados_completos"]["Aluno"].lower())
        
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
                            "acertos_portugues": resultado.get("acertos_portugues", 0),
                            "acertos_matematica": resultado.get("acertos_matematica", 0),
                            "erros": resultado["total"] - resultado["acertos"],
                            "erros_portugues": resultado.get("erros_portugues", 0),
                            "erros_matematica": resultado.get("erros_matematica", 0),
                            "percentual": resultado["percentual"]
                        }
                        enviar_para_planilha(client, dados_simples, resultado_comparacao, questoes_detectadas=resultado.get("questoes_detectadas"))
                        sucessos += 1
                    except Exception as e:
                        print(f"⚠️ Erro ao enviar {dados_completos['Aluno']}: {e}")
            print(f"✅ {sucessos}/{len(resultados_lote)} resultados enviados!")
        else:
            print("❌ Não foi possível conectar ao Google Sheets")
    except Exception as e:
        print(f"⚠️ Erro ao enviar para Sheets: {e}")
    
    return resultados_lote

def processar_lote_alunos(diretorio=".", usar_gemini=True, debug_mode=False, num_questoes=52):
    """
    Processa múltiplos cartões de alunos em lote
    
    Args:
        diretorio: Diretório contendo os arquivos
        usar_gemini: Se deve usar Gemini para cabeçalho
        debug_mode: Se deve mostrar debug detalhado
        num_questoes: Tipo de cartão (44 ou 52 questões)
        
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
        
        # Detectar respostas do gabarito usando o tipo específico (44 ou 52 questões)
        if "page_" in gabarito_img and (gabarito_img.endswith(".png") or gabarito_img.endswith(".jpg")):
            print("🔍 Usando detecção especializada para PDF...")
            respostas_gabarito = detectar_respostas_pdf(gabarito_img, debug=debug_mode)
        else:
            respostas_gabarito = detectar_respostas_por_tipo(gabarito_img, num_questoes=num_questoes, debug=debug_mode, eh_gabarito=True)
        
        questoes_gabarito = sum(1 for r in respostas_gabarito if r != '?')
        num_questoes_detectadas = len(respostas_gabarito)
        print(f"✅ Gabarito processado: {questoes_gabarito}/{num_questoes_detectadas} questões detectadas")
        
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
                    dados_extraidos = extrair_cabecalho_com_fallback(model_gemini, aluno_img, numero_aluno=i)
                    if dados_extraidos:
                        # Mapear chaves minúsculas do Gemini para maiúsculas do sistema
                        mapeamento = {
                            "escola": "Escola",
                            "aluno": "Aluno", 
                            "turma": "Turma",
                            "nascimento": "Nascimento"
                        }
                        
                        for chave_gemini, chave_sistema in mapeamento.items():
                            if chave_gemini in dados_extraidos and dados_extraidos[chave_gemini]:
                                dados_aluno[chave_sistema] = dados_extraidos[chave_gemini]
                        
                        print("✅ Dados extraídos pelo Gemini:")
                        for campo, valor in dados_aluno.items():
                            print(f"   📝 {campo}: {valor}")
                except Exception as e:
                    print(f"⚠️ Erro no Gemini para {aluno_file}: {e}")
                    dados_aluno["Aluno"] = f"Aluno {i}"  # Usar numeração automática
            else:
                dados_aluno["Aluno"] = f"Aluno {i}"  # Usar numeração automática
            
            # Detectar respostas do aluno usando o tipo específico (44 ou 52 questões)
            respostas_aluno = detectar_respostas_por_tipo(aluno_img, num_questoes=num_questoes, debug=debug_mode)
            
            questoes_aluno = sum(1 for r in respostas_aluno if r != '?')
            num_questoes_aluno = len(respostas_aluno)
            print(f"✅ Respostas processadas: {questoes_aluno}/{num_questoes_aluno} questões detectadas")
            
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
            
            print(f"📊 Resultado: ✓ {resultado.get('acertos_portugues', 0)}PT/{resultado.get('acertos_matematica', 0)}MT | ✗ {resultado.get('erros_portugues', 0)}PT/{resultado.get('erros_matematica', 0)}MT | Total {resultado['acertos']}/{resultado['total']} ({resultado['percentual']:.1f}%)")
            alunos_processados += 1
            
            # Delay de 12 segundos após processar cada cartão
            if i < len(arquivos_alunos):
                print(f"⏳ Aguardando 12 segundos antes do próximo cartão...")
                time.sleep(12)
            
        except Exception as e:
            print(f"❌ ERRO ao processar {aluno_file}: {e}")
            alunos_com_erro += 1
            # Adicionar resultado de erro
            resultado_erro = {
                "arquivo": aluno_file,
                "dados": {"Aluno": f"Aluno {i}", "Erro": str(e)},
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
        
        # Mostrar ranking (ordenado alfabeticamente)
        print(f"\n=== LISTA DE ALUNOS (ORDEM ALFABÉTICA) ===")
        resultados_validos = [r for r in resultados_lote if "Erro" not in r["dados"]]
        resultados_ordenados = sorted(resultados_validos, key=lambda x: x["dados"].get("Aluno", "").lower())
        
        for i, r in enumerate(resultados_ordenados[:10], 1):  # Top 10
            nome = r["dados"].get("Aluno", "N/A")
            acertos = r["resultado"]["acertos"]
            percentual = r["resultado"]["percentual"]
            print(f"   {i:02d}. {nome:<25} | {acertos:02d}/52 | {percentual:5.1f}%")
        
        if len(resultados_ordenados) > 10:
            print(f"   ... e mais {len(resultados_ordenados) - 10} alunos")
    
    return resultados_lote

def processar_pasta_gabaritos_sem_sheets(diretorio: str = "./gabaritos", usar_gemini: bool = True, debug_mode: bool = False, num_questoes: int = 52):
    """
    Versão da função que NÃO tenta enviar para Google Sheets
    (evita problema de cota do Drive)
    
    Args:
        diretorio: Caminho da pasta contendo gabarito e cartões dos alunos
        usar_gemini: Se deve usar Gemini para cabeçalho
        debug_mode: Se deve mostrar debug detalhado
        num_questoes: Tipo de cartão (44 ou 52 questões)
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
        
        # Detectar respostas do gabarito usando o tipo específico (44 ou 52 questões)
        respostas_gabarito = detectar_respostas_por_tipo(gabarito_img, num_questoes=num_questoes, debug=debug_mode, eh_gabarito=True)
        
        questoes_gabarito = sum(1 for r in respostas_gabarito if r != '?')
        num_questoes_detectadas = len(respostas_gabarito)
        print(f"✅ Gabarito processado: {questoes_gabarito}/{num_questoes_detectadas} questões detectadas")
        
        # Exibir gabarito em formato simples
        exibir_gabarito_simples(respostas_gabarito)
        
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
                "Aluno": f"Aluno {i}",
                "Escola": "N/A",
                "Nascimento": "N/A", 
                "Turma": "N/A"
            }
            
            if usar_gemini and model_gemini:
                try:
                    dados_extraidos = extrair_cabecalho_com_fallback(model_gemini, aluno_img, numero_aluno=i)
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
                    print(f"⚠️ Gemini falhou, usando numeração automática")
            
            # Detectar respostas do aluno usando o tipo específico (44 ou 52 questões)
            respostas_aluno = detectar_respostas_por_tipo(aluno_img, num_questoes=num_questoes, debug=debug_mode)
            
            questoes_aluno = sum(1 for r in respostas_aluno if r != '?')
            num_questoes_aluno = len(respostas_aluno)
            print(f"✅ Respostas processadas: {questoes_aluno}/{num_questoes_aluno} questões detectadas")
            
            # Calcular resultado
            resultado = comparar_respostas(respostas_gabarito, respostas_aluno)
            
            # Exibir resumo formatado
            print(f"\n{'─'*60}")
            print(f"👤 {dados_aluno['Aluno']}")
            print(f"📚 Turma: {dados_aluno['Turma']} | Escola: {dados_aluno['Escola']}")
            print(f"✅ Acertos: {resultado['acertos']}")
            print(f"❌ Erros: {resultado['erros']}")
            print(f"📊 Percentual: {resultado['percentual']:.1f}%")
            
            # Exibir respostas do aluno
            print(f"\n📝 Respostas:")
            exibir_gabarito_simples(respostas_aluno)
            print(f"{'─'*60}")
            
            # Armazenar resultado com dados completos
            resultado_completo = {
                "arquivo": aluno_file,
                "dados_completos": dados_aluno,  # Dados completos do cabeçalho
                "acertos": resultado['acertos'],
                "acertos_portugues": resultado.get('acertos_portugues', 0),
                "acertos_matematica": resultado.get('acertos_matematica', 0),
                "total": resultado['total'],
                "percentual": resultado['percentual'],
                "questoes_detectadas": questoes_aluno
            }
            resultados_lote.append(resultado_completo)
            
        except Exception as e:
            print(f"❌ ERRO ao processar {aluno_file}: {e}")
            resultado_erro = {
                "arquivo": aluno_file,
                "dados_completos": {
                    "Aluno": f"Aluno {i}",
                    "Escola": "N/A",
                    "Nascimento": "N/A",
                    "Turma": "N/A"
                },
                "acertos": 0,
                "acertos_portugues": 0,
                "acertos_matematica": 0,
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
        print(f"\n=== RESULTADOS DETALHADOS (ORDEM ALFABÉTICA) ===")
        
        # Ordenar por nome do aluno (ordem alfabética)
        resultados_ordenados = sorted(resultados_lote, key=lambda x: x["dados_completos"]["Aluno"].lower())
        
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
    debug_mode: bool = False,
    num_questoes: int = 52
):
    """
    Versão da função que ENVIA para Google Sheets com controle de rate limiting
    
    Args:
        diretorio: Caminho da pasta contendo gabarito e cartões dos alunos
        usar_gemini: Se deve usar Gemini para cabeçalho
        debug_mode: Se deve mostrar debug detalhado
        num_questoes: Tipo de cartão (44 ou 52 questões)
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
        
        # Detectar respostas do gabarito usando o tipo específico (44 ou 52 questões)
        respostas_gabarito = detectar_respostas_por_tipo(gabarito_img, num_questoes=num_questoes, debug=debug_mode, eh_gabarito=True)
        
        questoes_gabarito = sum(1 for r in respostas_gabarito if r != '?')
        num_questoes_detectadas = len(respostas_gabarito)
        print(f"✅ Gabarito processado: {questoes_gabarito}/{num_questoes_detectadas} questões detectadas")
        
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
                "Aluno": f"Aluno {i}",
                "Escola": "N/A",
                "Nascimento": "N/A", 
                "Turma": "N/A"
            }
            
            if usar_gemini and model_gemini:
                try:
                    dados_extraidos = extrair_cabecalho_com_fallback(model_gemini, aluno_img, numero_aluno=i)
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
                    print(f"⚠️ Gemini falhou, usando numeração automática")
            
            # Detectar respostas do aluno usando o tipo específico (44 ou 52 questões)
            respostas_aluno = detectar_respostas_por_tipo(aluno_img, num_questoes=num_questoes, debug=debug_mode)
            
            questoes_aluno = sum(1 for r in respostas_aluno if r != '?')
            num_questoes_aluno = len(respostas_aluno)
            print(f"✅ Respostas processadas: {questoes_aluno}/{num_questoes_aluno} questões detectadas")
            
            # Calcular resultado
            resultado = comparar_respostas(respostas_gabarito, respostas_aluno)
            
            # Exibir resumo formatado
            print(f"\n{'─'*60}")
            print(f"👤 {dados_aluno['Aluno']}")
            print(f"📚 Turma: {dados_aluno['Turma']} | Escola: {dados_aluno['Escola']}")
            print(f"✅ Acertos: {resultado['acertos']}")
            print(f"❌ Erros: {resultado['erros']}")
            print(f"📊 Percentual: {resultado['percentual']:.1f}%")
            
            # Exibir respostas do aluno
            print(f"\n📝 Respostas:")
            exibir_gabarito_simples(respostas_aluno)
            print(f"{'─'*60}")
            
            # Armazenar resultado com dados completos
            resultado_completo = {
                "arquivo": aluno_file,
                "dados_completos": dados_aluno,  # Dados completos do cabeçalho
                "acertos": resultado['acertos'],
                "acertos_portugues": resultado.get('acertos_portugues', 0),
                "acertos_matematica": resultado.get('acertos_matematica', 0),
                "total": resultado['total'],
                "percentual": resultado['percentual'],
                "questoes_detectadas": questoes_aluno
            }
            resultados_lote.append(resultado_completo)
            
            # ===========================================
            # ENVIAR PARA GOOGLE SHEETS COM RATE LIMITING
            # ===========================================
            
            if client:
                try:
                    print(f"📤 Enviando para Google Sheets (aluno {i}/{len(arquivos_alunos)})...")
                    
                    # RATE LIMITING: Aguardar entre envios para evitar quota
                    if i > 1:  # Não aguardar no primeiro
                        time.sleep(2)
                    
                    if enviar_para_planilha(client, dados_aluno, resultado, planilha_id=PLANILHA_ID, questoes_detectadas=questoes_aluno):
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
                    "Aluno": f"Aluno {i}",
                    "Escola": "N/A",
                    "Nascimento": "N/A",
                    "Turma": "N/A"
                },
                "acertos": 0,
                "acertos_portugues": 0,
                "acertos_matematica": 0,
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
        print(f"\n=== RESULTADOS DETALHADOS (ORDEM ALFABÉTICA) ===")
        
        # Ordenar por nome do aluno (ordem alfabética)
        resultados_ordenados = sorted(resultados_lote, key=lambda x: x["dados_completos"]["Aluno"].lower())
        
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


def processar_pdf_multiplas_paginas(
    pdf_path: str,
    num_questoes: int = 52,
    usar_gemini: bool = True,
    debug_mode: bool = False,
    enviar_para_sheets: bool = True,
    mover_para_drive: bool = True,
    pasta_destino_id: str = None
):
    """
    🆕 NOVA FUNÇÃO: Processa PDF com MÚLTIPLAS PÁGINAS de cartões resposta
    
    Workflow:
    1. Converte TODAS as páginas do PDF para PNG
    2. Processa CADA página como um cartão individual
    3. Envia resultados para Google Sheets
    4. Move arquivo processado para pasta do Drive
    
    Args:
        pdf_path: Caminho do arquivo PDF (pode ter múltiplas páginas)
        num_questoes: Tipo de cartão (44 ou 52 questões)
        usar_gemini: Se deve usar Gemini para extrair cabeçalho
        debug_mode: Se deve mostrar informações de debug
        enviar_para_sheets: Se deve enviar para Google Sheets
        mover_para_drive: Se deve mover arquivo para pasta processada
        pasta_destino_id: ID da pasta de destino no Drive (5º ou 9º ano)
        
    Returns:
        Lista de resultados de todos os cartões processados
        
    Exemplo de uso:
        >>> resultados = processar_pdf_multiplas_paginas(
        ...     pdf_path="cartoes_turma_a.pdf",
        ...     num_questoes=52,
        ...     enviar_para_sheets=True
        ... )
        >>> print(f"Processados {len(resultados)} cartões do PDF!")
    """
    from pdf_processor_simple import process_pdf_all_pages
    
    print("=" * 80)
    print("🚀 PROCESSAMENTO DE PDF COM MÚLTIPLAS PÁGINAS")
    print("=" * 80)
    
    # Validar arquivo
    if not os.path.exists(pdf_path):
        print(f"❌ Arquivo não encontrado: {pdf_path}")
        return []
    
    if not pdf_path.lower().endswith('.pdf'):
        print(f"❌ Arquivo não é PDF: {pdf_path}")
        return []
    
    # Configurar Gemini se necessário
    model_gemini = None
    if usar_gemini:
        try:
            model_gemini = configurar_gemini()
            print("✅ Gemini configurado!")
        except Exception as e:
            print(f"⚠️ Erro ao configurar Gemini: {e}")
            usar_gemini = False
    
    # Configurar Google Sheets se necessário
    client = None
    if enviar_para_sheets:
        try:
            client = configurar_google_sheets()
            print("✅ Google Sheets configurado!")
        except Exception as e:
            print(f"⚠️ Erro ao configurar Google Sheets: {e}")
            enviar_para_sheets = False
    
    try:
        # 1️⃣ CONVERTER TODAS AS PÁGINAS PARA PNG
        print(f"\n📄 Convertendo TODAS as páginas do PDF para PNG...")
        imagens_paginas = process_pdf_all_pages(pdf_path, keep_temp_files=True)
        
        if not imagens_paginas:
            print("❌ Nenhuma imagem foi gerada do PDF")
            return []
        
        print(f"✅ {len(imagens_paginas)} páginas convertidas!")
        
        # 🆕 1.5️⃣ CONVERTER TODAS AS IMAGENS PARA PRETO E BRANCO
        print(f"\n🎨 Convertendo imagens para Preto e Branco...")
        imagens_pb = []
        
        for i, img_path in enumerate(imagens_paginas, 1):
            try:
                print(f"   [{i}/{len(imagens_paginas)}] Convertendo {os.path.basename(img_path)}...", end='')
                
                # Converter para P&B
                img_pb_path = converter_para_preto_e_branco(
                    img_path,
                    threshold=180,  # Threshold padrão
                    salvar=True
                )
                
                if img_pb_path and os.path.exists(img_pb_path):
                    # Substituir original pela versão P&B
                    os.remove(img_path)
                    os.rename(img_pb_path, img_path)
                    imagens_pb.append(img_path)
                    print(" ✅")
                else:
                    # Se falhar, usar original
                    imagens_pb.append(img_path)
                    print(" ⚠️ (usando original)")
                    
            except Exception as e:
                print(f" ❌ Erro: {e}")
                imagens_pb.append(img_path)  # Usar original se falhar
        
        print(f"✅ {len(imagens_pb)} imagens prontas para processamento")
        
        # Usar imagens P&B daqui em diante
        imagens_paginas = imagens_pb
        
        # 2️⃣ BUSCAR GABARITO EM ARQUIVO SEPARADO (não no PDF)
        print(f"\n📋 Buscando gabarito em arquivo separado...")
        
        # O gabarito deve estar na mesma pasta do PDF (arquivo PNG/JPG com "gabarito" no nome)
        pasta_pdf = os.path.dirname(pdf_path)
        gabarito_img = None
        
        # Buscar arquivo de gabarito na pasta
        for arquivo in os.listdir(pasta_pdf):
            if 'gabarito' in arquivo.lower() and arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                gabarito_path = os.path.join(pasta_pdf, arquivo)
                gabarito_img = gabarito_path
                print(f"✅ Gabarito encontrado: {arquivo}")
                break
        
        if not gabarito_img:
            print("❌ ERRO: Arquivo de gabarito não encontrado na pasta!")
            print(f"   Procurei por arquivo PNG/JPG com 'gabarito' no nome em: {pasta_pdf}")
            return []
        
        # 🆕 TODAS as páginas do PDF são cartões de alunos
        cartoes_alunos = imagens_paginas  # Todas as páginas são alunos!
        
        print(f"✅ Cartões de alunos no PDF: {len(cartoes_alunos)} páginas")
        
        # 3️⃣ PROCESSAR GABARITO
        print(f"\n{'='*80}")
        print("📋 PROCESSANDO GABARITO (Arquivo separado)")
        print(f"{'='*80}")
        
        respostas_gabarito = detectar_respostas_por_tipo(
            gabarito_img, 
            num_questoes=num_questoes, 
            debug=True,  # 🆕 Sempre ativar debug para gabarito
            eh_gabarito=True
        )
        
        questoes_gabarito = sum(1 for r in respostas_gabarito if r != '?')
        print(f"\n✅ Gabarito processado: {questoes_gabarito}/{num_questoes} questões detectadas")
        
        if questoes_gabarito < num_questoes * 0.8:  # Menos de 80% detectado
            print(f"⚠️ ATENÇÃO: Poucas questões detectadas no gabarito ({questoes_gabarito}/{num_questoes})")
            print("   Isso pode afetar a correção dos cartões dos alunos.")
        
        # Exibir gabarito
        print(f"\n{'='*60}")
        print("📋 GABARITO:")
        print(f"{'='*60}")
        exibir_gabarito_simples(respostas_gabarito)
        
        # 4️⃣ PROCESSAR CADA CARTÃO DE ALUNO
        resultados_todos = []
        
        print(f"\n{'='*80}")
        print(f"👥 PROCESSANDO {len(cartoes_alunos)} CARTÕES DE ALUNOS")
        print(f"{'='*80}")
        
        for i, cartao_img in enumerate(cartoes_alunos, 1):
            pagina_num = i  # 🆕 Agora todas as páginas são alunos (1, 2, 3...)
            print(f"\n🔄 [{i:02d}/{len(cartoes_alunos)}] Processando Página {pagina_num}")
            print("-" * 60)
            
            try:
                # Extrair dados do cabeçalho
                dados_aluno = {
                    "Aluno": f"Página_{pagina_num}",
                    "Escola": "N/A",
                    "Nascimento": "N/A",
                    "Turma": "N/A"
                }
                
                if usar_gemini and model_gemini:
                    try:
                        dados_extraidos = extrair_cabecalho_com_fallback(model_gemini, cartao_img)
                        if dados_extraidos and dados_extraidos.get("aluno"):
                            dados_aluno["Aluno"] = dados_extraidos.get("aluno", f"Página_{pagina_num}")
                            dados_aluno["Escola"] = dados_extraidos.get("escola", "N/A")
                            dados_aluno["Turma"] = dados_extraidos.get("turma", "N/A")
                            dados_aluno["Nascimento"] = dados_extraidos.get("nascimento", "N/A")
                    except Exception as e:
                        pass  # Silenciar erros do Gemini
                
                # Detectar respostas do aluno (COM debug para análise)
                respostas_aluno = detectar_respostas_por_tipo(
                    cartao_img, 
                    num_questoes=num_questoes, 
                    debug=True  # 🆕 Ativar debug para ver detecção das bolhas
                )
                
                questoes_detectadas = sum(1 for r in respostas_aluno if r != '?')
                
                # Verificar se detectou questões suficientes
                if questoes_detectadas < num_questoes * 0.5:  # Menos de 50%
                    print(f"❌ Página {pagina_num}: Poucas questões ({questoes_detectadas}/{num_questoes}) - IGNORADO")
                    continue
                
                # Comparar com gabarito
                resultado = comparar_respostas(respostas_gabarito, respostas_aluno)
                
                # 🆕 MOSTRAR APENAS RESUMO COMPACTO
                print(f"\n{'─'*60}")
                print(f"� {dados_aluno['Aluno']}")
                print(f"📚 Turma: {dados_aluno['Turma']} | Escola: {dados_aluno['Escola']}")
                print(f"✅ Acertos: {resultado['acertos']}")
                print(f"❌ Erros: {resultado['erros']}")
                print(f"📊 Percentual: {resultado['percentual']:.1f}%")
                
                # 🆕 MOSTRAR GABARITO DE RESPOSTAS DO ALUNO
                print(f"\n📝 Respostas:")
                exibir_gabarito_simples(respostas_aluno)
                
                print(f"{'─'*60}")
                
                # Armazenar resultado
                resultados_todos.append({
                    "pagina": pagina_num,
                    "arquivo": os.path.basename(pdf_path),
                    "dados_aluno": dados_aluno,
                    "resultado": resultado,
                    "questoes_detectadas": questoes_detectadas
                })
                
                # Enviar para Google Sheets (silencioso)
                if enviar_para_sheets and client:
                    try:
                        enviar_para_planilha(client, dados_aluno, resultado, questoes_detectadas=questoes_detectadas)
                    except Exception as e:
                        print(f"⚠️ Erro ao enviar para Sheets: {e}")
                
            except Exception as e:
                print(f"❌ ERRO ao processar página {pagina_num}: {e}")
                continue
        
        # 5️⃣ RESUMO FINAL
        print(f"\n{'='*80}")
        print("📊 RESUMO DO PROCESSAMENTO")
        # 🆕 RESUMO FINAL COMPACTO
        print(f"\n{'='*80}")
        print(f"📄 PDF: {os.path.basename(pdf_path)}")
        print(f"{'='*80}")
        print(f"📋 Total de páginas: {len(imagens_paginas)}")
        print(f"✅ Cartões processados: {len(resultados_todos)}/{len(cartoes_alunos)}")
        
        if len(resultados_todos) > 0:
            media_acertos = sum(r['resultado']['acertos'] for r in resultados_todos) / len(resultados_todos)
            media_erros = sum(r['resultado']['erros'] for r in resultados_todos) / len(resultados_todos)
            media_percentual = sum(r['resultado']['percentual'] for r in resultados_todos) / len(resultados_todos)
            
            print(f"\n📊 ESTATÍSTICAS:")
            print(f"   Média de acertos: {media_acertos:.1f}/{num_questoes}")
            print(f"   Média de erros: {media_erros:.1f}/{num_questoes}")
            print(f"   Média geral: {media_percentual:.1f}%")
        
        print(f"{'='*80}")
        
        # 6️⃣ LIMPAR ARQUIVOS TEMPORÁRIOS
        print(f"\n🧹 Limpando arquivos temporários...")
        for img in imagens_paginas:
            try:
                if os.path.exists(img):
                    os.remove(img)
            except:
                pass
        
        print("✅ Processamento concluído!")
        return resultados_todos
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        return []


# ===========================================
# EXECUÇÃO PRINCIPAL
# ===========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sistema automatizado de correção de cartões resposta com Google Drive e Google Sheets."
    )

    parser.add_argument(
        "--drive-folder",
        dest="drive_folder_custom",
        default=None,
        help="ID CUSTOMIZADO da pasta do Google Drive (opcional - sobrescreve as pastas padrão)"
    )
    parser.add_argument(
        "--gabarito",
        action="store_true",
        help="Exibe apenas o gabarito das questões em formato simples (1-A, 2-B, 3-C)"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Inicia monitoramento contínuo da pasta (verifica novos arquivos automaticamente)"
    )
    parser.add_argument(
        "--intervalo",
        type=int,
        default=5,
        help="Intervalo de verificação em minutos para modo monitor (padrão: 5)"
    )
    parser.add_argument(
        "--converter-pb",
        action="store_true",
        default=True,
        help="Converte imagens para preto e branco automaticamente (padrão: ativado)"
    )
    parser.add_argument(
        "--no-converter-pb",
        dest="converter_pb",
        action="store_false",
        help="Desativa conversão automática para preto e branco"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=180,
        help="Threshold para conversão P&B, 0-255 (padrão: 180, menor=mais preto)"
    )
    parser.add_argument(
        "--questoes",
        type=int,
        choices=[44, 52],
        default=52,
        help="Número de questões no cartão (44 ou 52, padrão: 52)"
    )
    parser.add_argument(
        "--pdf-multiplo",
        type=str,
        default=None,
        metavar="ARQUIVO.PDF",
        help="🆕 Processa PDF com múltiplas páginas (cada página = 1 cartão). Ex: --pdf-multiplo cartoes_turma.pdf"
    )

    args = parser.parse_args()


    if PDF_PROCESSOR_AVAILABLE:
        pdf_ok = setup_pdf_support()
        if not pdf_ok:
            print("⚠️ Suporte a PDF limitado - apenas imagens serão processadas")

    # Configurações fixas para automação total
    usar_gemini = True
    enviar_para_sheets = True
    debug_mode = True
    mover_processados = True
    manter_temp = False
    
    # Configurações de conversão P&B
    converter_pb = args.converter_pb
    threshold_pb = args.threshold
    
    # 🆕 SISTEMA AUTOMATIZADO - Não há mais seleção manual de questões!
    # Os gabaritos serão carregados automaticamente (44 e 52 questões)
    # E o sistema escolherá o gabarito correto baseado na turma detectada no cartão
    print("\n" + "=" * 80)
    print("🤖 SISTEMA DE CORREÇÃO AUTOMATIZADO")
    print("=" * 80)
    print("✨ Detecção automática de ano baseada na turma do aluno:")
    print("   • 5° ano → 44 questões")
    print("   • 9° ano → 52 questões")
    print("\n💡 Certifique-se de ter 2 arquivos de gabarito (IMAGENS):")
    print("   🖼️  gabarito_44.png (ou .jpg) → Para alunos do 5° ano")
    print("   🖼️  gabarito_52.png (ou .jpg) → Para alunos do 9° ano")
    print("=" * 80)

    # 👉 Carregar IDs das pastas do Google Drive do arquivo .env
    DRIVER_FOLDER_UPLOAD = os.getenv("DRIVER_FOLDER_ID")  # Pasta de UPLOAD (origem)
    DRIVER_FOLDER_5ANO = os.getenv("DRIVER_FOLDER_5ANO")   # Pasta 5º ano (destino 44 questões)
    DRIVER_FOLDER_9ANO = os.getenv("DRIVER_FOLDER_9ANO")   # Pasta 9º ano (destino 52 questões)
    
    # Validar se as variáveis foram carregadas
    if not all([DRIVER_FOLDER_UPLOAD, DRIVER_FOLDER_5ANO, DRIVER_FOLDER_9ANO]):
        print("❌ ERRO: Variáveis de ambiente não configuradas no .env!")
        print("   Verifique se DRIVER_FOLDER_ID, DRIVER_FOLDER_5ANO e DRIVER_FOLDER_9ANO estão definidos.")
        exit(1)
    
    # Sempre usa a pasta de UPLOAD como origem
    if args.drive_folder_custom:
        # Usar pasta customizada se fornecida
        pasta_drive_id = args.drive_folder_custom
        print(f"📁 Usando pasta customizada do Drive: {pasta_drive_id}")
    else:
        pasta_drive_id = DRIVER_FOLDER_UPLOAD
    
    # 🆕 Pastas de destino serão escolhidas AUTOMATICAMENTE por aluno
    # Com base no ano detectado na turma
    print(f"\n📁 Pastas de destino configuradas:")
    print(f"   • 5° ano → {DRIVER_FOLDER_5ANO}")
    print(f"   • 9° ano → {DRIVER_FOLDER_9ANO}")
    print("=" * 80)

    # 🆕 MODO ESPECIAL: PDF COM MÚLTIPLAS PÁGINAS
    if args.pdf_multiplo:
        print("\n" + "=" * 80)
        print("🆕 MODO: PROCESSAMENTO DE PDF COM MÚLTIPLAS PÁGINAS")
        print("=" * 80)
        
        if not os.path.exists(args.pdf_multiplo):
            print(f"❌ ERRO: Arquivo não encontrado: {args.pdf_multiplo}")
            exit(1)
        
        if not args.pdf_multiplo.lower().endswith('.pdf'):
            print(f"❌ ERRO: Arquivo não é PDF: {args.pdf_multiplo}")
            print("   Use --pdf-multiplo apenas com arquivos .pdf")
            exit(1)
        
        print(f"\n📄 Processando: {args.pdf_multiplo}")
        print(f"🤖 Gemini: {'ATIVADO' if usar_gemini else 'DESATIVADO'}")
        print(f"📊 Google Sheets: {'ATIVADO' if enviar_para_sheets else 'DESATIVADO'}")
        print(f"🔍 Debug: {'ATIVADO' if debug_mode else 'DESATIVADO'}")
        print(f"\n⚠️ NOTA: Modo PDF múltiplo com sistema automatizado")
        print(f"   O sistema detectará automaticamente 44 ou 52 questões por aluno")
        
        # ⚠️ Modo PDF múltiplo precisa ser atualizado para sistema automatizado
        print(f"\n❌ ERRO: Modo --pdf-multiplo ainda não foi adaptado para o sistema automatizado")
        print(f"   Use o modo --monitor para processar PDFs automaticamente")
        print(f"   Coloque o PDF na pasta de upload do Google Drive")
        exit(1)
        
        if resultados:
            print(f"\n✅ SUCESSO! {len(resultados)} cartões processados do PDF")
        else:
            print(f"\n❌ FALHA! Nenhum cartão foi processado")
        
        exit(0)

    # Modo especial: apenas exibir gabarito
    if args.gabarito:
        print(f"\n❌ ERRO: Modo --gabarito desabilitado no sistema automatizado")
        print(f"   O sistema agora carrega automaticamente ambos os gabaritos")
        print(f"   Use o modo --monitor para processar cartões")
        exit(1)

    # Modo especial: monitoramento contínuo
    if args.monitor:
        print("=" * 60)
        print("🤖 MODO MONITORAMENTO CONTÍNUO ATIVADO - SISTEMA AUTOMATIZADO")
        print(f"⏰ Intervalo: {args.intervalo} minutos")
        print(f"📂 Pasta de ORIGEM (upload): {pasta_drive_id}")
        print(f"📁 Pastas de DESTINO:")
        print(f"   • 5° ano (44 questões) → {DRIVER_FOLDER_5ANO}")
        print(f"   • 9° ano (52 questões) → {DRIVER_FOLDER_9ANO}")
        print("✨ Sistema detectará automaticamente o ano de cada cartão")
        print("💡 Pressione Ctrl+C para parar")
        print("=" * 60)
        
        import time
        import json
        from datetime import datetime
        
        # Arquivo para rastrear arquivos já processados por ID e NOME
        historico_file = "historico_monitoramento.json"
        
        def carregar_historico():
            """Carrega IDs e nomes (sem extensão) dos arquivos já processados"""
            try:
                if os.path.exists(historico_file):
                    with open(historico_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Migração: se formato antigo (lista), converter para novo formato
                        arquivos = data.get('arquivos_processados', [])
                        if arquivos and isinstance(arquivos[0], str):
                            # Formato antigo: apenas IDs
                            return {'ids': set(arquivos), 'nomes': set()}
                        else:
                            # Formato novo: dicionário com ID e nome
                            ids = set(item['id'] for item in arquivos)
                            nomes = set(item['nome_sem_ext'] for item in arquivos)
                            return {'ids': ids, 'nomes': nomes}
            except Exception as e:
                print(f"⚠️ Erro ao carregar histórico: {e}")
            return {'ids': set(), 'nomes': set()}
        
        def salvar_historico(arquivos_processados):
            """Salva IDs e nomes dos arquivos processados"""
            try:
                data = {
                    'ultima_verificacao': datetime.now().isoformat(),
                    'total_processados': len(arquivos_processados),
                    'arquivos_processados': arquivos_processados
                }
                with open(historico_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"⚠️ Erro ao salvar histórico: {e}")
        
        def verificar_novos_arquivos():
            """Verifica se há NOVOS arquivos para processar (por ID e NOME)"""
            try:
                # Configurar Google Drive
                drive_service = configurar_google_drive_service_completo()
                if not drive_service:
                    print("❌ Erro ao conectar com Google Drive")
                    return [], {'ids': set(), 'nomes': set()}
                
                # Listar arquivos na pasta
                query = f"'{pasta_drive_id}' in parents and trashed = false"
                results = drive_service.files().list(
                    q=query,
                    fields="files(id, name, mimeType, modifiedTime)",
                    pageSize=100
                ).execute()
                
                arquivos = results.get('files', [])
                historico = carregar_historico()
                ids_processados = historico['ids']
                nomes_processados = historico['nomes']
                
                # 🆕 DEBUG: Mostrar TODOS os arquivos encontrados
                print(f"\n📂 Arquivos na pasta do Drive: {len(arquivos)}")
                for arq in arquivos:
                    nome = arq['name']
                    arquivo_id = arq['id']
                    nome_sem_ext = os.path.splitext(nome)[0].lower()
                    
                    # Verificar duplicata por ID ou NOME
                    ja_processado_id = arquivo_id in ids_processados
                    ja_processado_nome = nome_sem_ext in nomes_processados
                    
                    if ja_processado_id or ja_processado_nome:
                        motivo = "ID" if ja_processado_id else "NOME"
                        print(f"   ✅ PROCESSADO ({motivo}) | {nome}")
                    else:
                        print(f"   🆕 NOVO | {nome}")
                
                novos_cartoes = []
                tem_gabarito = False
                
                for arquivo in arquivos:
                    arquivo_id = arquivo['id']
                    nome = arquivo['name']
                    nome_lower = nome.lower()
                    nome_sem_ext = os.path.splitext(nome)[0].lower()
                    
                    # Verificar se é o gabarito (nunca marcar como processado)
                    if 'gabarito' in nome_lower and any(ext in nome_lower for ext in ['.pdf', '.png', '.jpg', '.jpeg']):
                        tem_gabarito = True
                        print(f"📋 Gabarito detectado: {nome}")
                        continue
                    
                    # Verificar se é um cartão de aluno NOVO (por ID E NOME)
                    if any(ext in nome_lower for ext in ['.pdf', '.png', '.jpg', '.jpeg']):
                        # Duplicata por ID?
                        if arquivo_id in ids_processados:
                            print(f"   ⏭️ Ignorado (ID duplicado): {nome}")
                            continue
                        
                        # Duplicata por NOME?
                        if nome_sem_ext in nomes_processados:
                            print(f"   ⏭️ Ignorado (NOME duplicado): {nome}")
                            print(f"      ⚠️ Arquivo com mesmo nome já foi processado (mesmo com extensão diferente)")
                            continue
                        
                        # É NOVO!
                        tipo = "📄 PDF" if nome_lower.endswith('.pdf') else "🖼️ Imagem"
                        print(f"   {tipo} NOVO detectado: {nome}")
                        novos_cartoes.append(arquivo)
                
                if not tem_gabarito and novos_cartoes:
                    print("⚠️ Novos cartões encontrados mas GABARITO não está na pasta!")
                    return [], historico
                
                print(f"\n📊 Resumo:")
                print(f"   Total na pasta: {len(arquivos)}")
                print(f"   Já processados (IDs): {len(ids_processados)}")
                print(f"   Já processados (Nomes): {len(nomes_processados)}")
                print(f"   Gabarito: {'✅ Encontrado' if tem_gabarito else '❌ Não encontrado'}")
                print(f"   Novos a processar: {len(novos_cartoes)}")
                
                return novos_cartoes, historico
                
            except Exception as e:
                print(f"❌ Erro ao verificar arquivos: {e}")
                import traceback
                traceback.print_exc()
                return [], {'ids': set(), 'nomes': set()}
        
        # Loop de monitoramento
        contador_verificacoes = 0
        try:
            while True:
                try:
                    contador_verificacoes += 1
                    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    print(f"\n🔍 Verificação #{contador_verificacoes} - {timestamp}")
                    
                    # Verificar NOVOS cartões (por ID e NOME)
                    novos_cartoes, historico = verificar_novos_arquivos()
                    
                    if novos_cartoes:
                        print(f"🆕 Encontrados {len(novos_cartoes)} NOVOS cartões!")
                        for arquivo in novos_cartoes:
                            print(f"   -> {arquivo['name']} ")
                        
                        # Processar APENAS os novos cartões
                        print("🚀 Processando APENAS os novos cartões...")
                        try:
                            # Imports necessários
                            import tempfile
                            import shutil
                            from googleapiclient.http import MediaIoBaseDownload
                            
                            # Configurar serviços
                            if usar_gemini:
                                model_gemini = configurar_gemini()
                            else:
                                model_gemini = None
                            
                            if enviar_para_sheets:
                                client = configurar_google_sheets()
                                PLANILHA_ID = os.getenv('GOOGLE_SHEETS_9ANO')
                            else:
                                client = None
                                PLANILHA_ID = None
                            
                            drive_service = configurar_google_drive_service_completo()
                            
                            # Pasta temporária
                            pasta_temp = tempfile.mkdtemp(prefix="cartoes_novos_")
                            print(f"📁 Pasta temporária: {pasta_temp}")
                            
                            # 🆕 1. CARREGAR AMBOS OS GABARITOS AUTOMATICAMENTE
                            print(f"\n{'='*80}")
                            print("📚 CARREGANDO GABARITOS AUTOMATICAMENTE")
                            print(f"{'='*80}")
                            
                            # Baixar gabaritos do Drive
                            gabaritos_dict = {}
                            
                            # Buscar gabarito de 44 questões
                            query_gab44 = f"'{pasta_drive_id}' in parents and (name contains 'gabarito_44' or name contains 'gabarito44') and trashed = false"
                            results_44 = drive_service.files().list(
                                q=query_gab44,
                                fields="files(id, name, mimeType)",
                                pageSize=1
                            ).execute()
                            
                            gab44_files = results_44.get('files', [])
                            if gab44_files:
                                gab44_info = gab44_files[0]
                                print(f"✅ Gabarito 44 questões: {gab44_info['name']}")
                                
                                # Baixar
                                request = drive_service.files().get_media(fileId=gab44_info['id'])
                                gab44_path = os.path.join(pasta_temp, gab44_info['name'])
                                with open(gab44_path, 'wb') as f:
                                    downloader = MediaIoBaseDownload(f, request)
                                    done = False
                                    while not done:
                                        status, done = downloader.next_chunk()
                                
                                # Processar
                                gab44_img = preprocessar_arquivo(gab44_path, "gabarito_44")
                                respostas_44 = detectar_respostas_por_tipo(gab44_img, num_questoes=44, debug=False, eh_gabarito=True)
                                gabaritos_dict[44] = respostas_44
                                print(f"   ✓ 44 questões processadas: {sum(1 for r in respostas_44 if r != '?')}/44")
                            else:
                                print("❌ Gabarito de 44 questões não encontrado!")
                                
                            # Buscar gabarito de 52 questões
                            query_gab52 = f"'{pasta_drive_id}' in parents and (name contains 'gabarito_52' or name contains 'gabarito52') and trashed = false"
                            results_52 = drive_service.files().list(
                                q=query_gab52,
                                fields="files(id, name, mimeType)",
                                pageSize=1
                            ).execute()
                            
                            gab52_files = results_52.get('files', [])
                            if gab52_files:
                                gab52_info = gab52_files[0]
                                print(f"✅ Gabarito 52 questões: {gab52_info['name']}")
                                
                                # Baixar
                                request = drive_service.files().get_media(fileId=gab52_info['id'])
                                gab52_path = os.path.join(pasta_temp, gab52_info['name'])
                                with open(gab52_path, 'wb') as f:
                                    downloader = MediaIoBaseDownload(f, request)
                                    done = False
                                    while not done:
                                        status, done = downloader.next_chunk()
                                
                                # Processar
                                gab52_img = preprocessar_arquivo(gab52_path, "gabarito_52")
                                respostas_52 = detectar_respostas_por_tipo(gab52_img, num_questoes=52, debug=False, eh_gabarito=True)
                                gabaritos_dict[52] = respostas_52
                                print(f"   ✓ 52 questões processadas: {sum(1 for r in respostas_52 if r != '?')}/52")
                            else:
                                print("❌ Gabarito de 52 questões não encontrado!")
                            
                            # Validar se ambos foram carregados
                            if 44 not in gabaritos_dict or 52 not in gabaritos_dict:
                                print("\n❌ ERRO: Ambos os gabaritos são necessários!")
                                print("   Certifique-se de ter gabarito_44.png e gabarito_52.png no Drive")
                                continue
                            
                            print(f"\n✅ AMBOS OS GABARITOS CARREGADOS COM SUCESSO!")
                            print(f"{'='*80}")
                            
                            # 2. Processar cada cartão NOVO (separar PDFs de imagens)
                            arquivos_processados_agora = []  # Lista de {id, nome_sem_ext}
                            pdfs_para_processar = []
                            imagens_para_processar = []
                            
                            # Separar PDFs de imagens
                            for cartao_info in novos_cartoes:
                                nome = cartao_info['name'].lower()
                                if nome.endswith('.pdf'):
                                    pdfs_para_processar.append(cartao_info)
                                elif nome.endswith(('.png', '.jpg', '.jpeg')):
                                    imagens_para_processar.append(cartao_info)
                            
                            print(f"\n📊 Arquivos detectados:")
                            print(f"   📄 PDFs: {len(pdfs_para_processar)}")
                            print(f"   🖼️ Imagens: {len(imagens_para_processar)}")
                            
                            # ═══════════════════════════════════════════════════
                            # PROCESSAR PDFs (MÚLTIPLAS PÁGINAS)
                            # ═══════════════════════════════════════════════════
                            if pdfs_para_processar:
                                print(f"\n{'='*80}")
                                print(f"📄 PROCESSANDO {len(pdfs_para_processar)} PDF(s) COM MÚLTIPLAS PÁGINAS")
                                print(f"{'='*80}")
                                
                                from pdf_processor_simple import process_pdf_all_pages
                                
                                for pdf_idx, pdf_info in enumerate(pdfs_para_processar, 1):
                                    try:
                                        print(f"\n📄 [{pdf_idx}/{len(pdfs_para_processar)}] {pdf_info['name']}")
                                        
                                        # Baixar PDF
                                        request = drive_service.files().get_media(fileId=pdf_info['id'])
                                        pdf_path = os.path.join(pasta_temp, pdf_info['name'])
                                        with open(pdf_path, 'wb') as f:
                                            downloader = MediaIoBaseDownload(f, request)
                                            done = False
                                            while not done:
                                                status, done = downloader.next_chunk()
                                        
                                        print(f"✅ PDF baixado: {pdf_info['name']}")
                                        
                                        # Converter TODAS as páginas para PNG
                                        print(f"🔄 Convertendo páginas do PDF para PNG...")
                                        imagens_paginas = process_pdf_all_pages(pdf_path, keep_temp_files=True)
                                        
                                        if not imagens_paginas:
                                            print(f"❌ Nenhuma página convertida do PDF")
                                            continue
                                        
                                        print(f"✅ {len(imagens_paginas)} páginas convertidas!")
                                        
                                        # Converter para P&B
                                        print(f"🎨 Convertendo para P&B...")
                                        for img_idx, img_path in enumerate(imagens_paginas, 1):
                                            try:
                                                img_pb_path = converter_para_preto_e_branco(
                                                    img_path,
                                                    threshold=threshold_pb,
                                                    salvar=True
                                                )
                                                if img_pb_path and os.path.exists(img_pb_path):
                                                    os.remove(img_path)
                                                    os.rename(img_pb_path, img_path)
                                            except Exception as e:
                                                print(f"   ⚠️ Erro ao converter página {img_idx}: {e}")
                                        
                                        print(f"✅ Todas as páginas prontas!")
                                        
                                        # 🆕 Variável para rastrear a pasta destino do PDF
                                        # Se todas as páginas forem do mesmo ano, vai para aquela pasta
                                        # Se houver páginas mistas, vai para a pasta do 9° ano
                                        pastas_detectadas = []
                                        
                                        # Processar CADA página como um aluno
                                        print(f"\n{'─'*60}")
                                        print(f"👥 Processando {len(imagens_paginas)} alunos do PDF")
                                        print(f"{'─'*60}")
                                        
                                        for pagina_idx, pagina_img in enumerate(imagens_paginas, 1):
                                            try:
                                                print(f"\n🔄 Página {pagina_idx}/{len(imagens_paginas)}")
                                                
                                                # 🆕 USAR EXTRAÇÃO OTIMIZADA (1 chamada única ao Gemini)
                                                num_questoes_pagina = None
                                                dados_aluno = None
                                                
                                                # PRIORIDADE 1: Tentar Gemini primeiro
                                                if model_gemini:
                                                    dados_completos = extrair_dados_completos_com_gemini(
                                                        model_gemini, 
                                                        pagina_img,
                                                        nome_arquivo=pdf_info['name']
                                                    )
                                                    if dados_completos:
                                                        dados_aluno = dados_completos
                                                        num_questoes_pagina = dados_completos.get('num_questoes')
                                                        if num_questoes_pagina:
                                                            print(f"   ✅ Gemini detectou com sucesso: {num_questoes_pagina} questões")
                                                
                                                # FALLBACK: Se Gemini falhar, usar OCR direto
                                                if not num_questoes_pagina:
                                                    print(f"   ⚠️ Gemini falhou - usando OCR como fallback")
                                                    num_questoes_pagina = detectar_ano_com_ocr_direto(pagina_img, debug=False)
                                                    print(f"   📊 OCR (fallback) detectou: {num_questoes_pagina} questões")
                                                
                                                # Se dados do aluno não foram extraídos, usar OCR
                                                if not dados_aluno:
                                                    dados_aluno = extrair_cabecalho_com_ocr_fallback(pagina_img)
                                                
                                                if not dados_aluno or dados_aluno.get("aluno") == "N/A":
                                                    dados_aluno = {
                                                        "escola": "N/A",
                                                        "aluno": f"{os.path.splitext(pdf_info['name'])[0]}_pag{pagina_idx}",
                                                        "turma": "N/A",
                                                        "nascimento": "N/A"
                                                    }
                                                
                                                print(f"   🔍 DEBUG - Dados extraídos: Escola={dados_aluno.get('escola')}, Aluno={dados_aluno.get('aluno')}, Turma={dados_aluno.get('turma')}, Nasc={dados_aluno.get('nascimento')}")
                                                
                                                # 🆕 SELECIONAR PASTA DE DESTINO BASEADA NO ANO DETECTADO
                                                if num_questoes_pagina == 44:
                                                    pasta_destino_pagina = DRIVER_FOLDER_5ANO
                                                    print(f"   📁 Destino: Pasta 5° ano")
                                                else:  # 52 questões
                                                    pasta_destino_pagina = DRIVER_FOLDER_9ANO
                                                    print(f"   📁 Destino: Pasta 9° ano")
                                                
                                                # 🆕 Registrar pasta detectada para esta página
                                                pastas_detectadas.append(pasta_destino_pagina)
                                                
                                                # 🆕 SELECIONAR GABARITO CORRETO PARA ESTA PÁGINA
                                                respostas_gabarito_correto = gabaritos_dict.get(num_questoes_pagina)
                                                if not respostas_gabarito_correto:
                                                    print(f"   ❌ Gabarito de {num_questoes_pagina} questões não disponível!")
                                                    continue
                                                
                                                # Detectar respostas (usando número detectado para esta página)
                                                respostas_aluno = detectar_respostas_por_tipo(
                                                    pagina_img, 
                                                    num_questoes=num_questoes_pagina, 
                                                    debug=False
                                                )
                                                
                                                questoes_detectadas = sum(1 for r in respostas_aluno if r != '?')
                                                
                                                # Verificar detecção mínima
                                                if questoes_detectadas < num_questoes_pagina * 0.5:
                                                    print(f"   ⚠️ Poucas questões detectadas ({questoes_detectadas}/{num_questoes_pagina}) - IGNORADO")
                                                    continue
                                                
                                                # Comparar com gabarito correto
                                                resultado = comparar_respostas(respostas_gabarito_correto, respostas_aluno)
                                                
                                                # Exibir resumo formatado com respostas do aluno
                                                print(f"\n{'─'*60}")
                                                print(f"👤 {dados_aluno.get('aluno', 'N/A')}")
                                                print(f"📚 Turma: {dados_aluno.get('turma', 'N/A')} | Escola: {dados_aluno.get('escola', 'N/A')}")
                                                print(f"✅ Acertos: {resultado['acertos']}")
                                                print(f"❌ Erros: {resultado['erros']}")
                                                print(f"📊 Percentual: {resultado['percentual']:.1f}%")
                                                
                                                # Exibir respostas do aluno
                                                print(f"\n📝 Respostas:")
                                                exibir_gabarito_simples(respostas_aluno)
                                                print(f"{'─'*60}")
                                                
                                                # 🆕 SELECIONAR PASTA DE DESTINO BASEADA NO ANO DETECTADO
                                                if num_questoes_pagina == 44:
                                                    pasta_destino_pagina = DRIVER_FOLDER_5ANO
                                                    print(f"   📁 Destino: Pasta 5° ano")
                                                else:  # 52 questões
                                                    pasta_destino_pagina = DRIVER_FOLDER_9ANO
                                                    print(f"   📁 Destino: Pasta 9° ano")
                                                
                                                # Enviar para Sheets (já escolhe planilha correta automaticamente)
                                                if client and PLANILHA_ID:
                                                    dados_envio = {
                                                        "Escola": dados_aluno.get("escola", "N/A"),
                                                        "Aluno": dados_aluno.get("aluno", "N/A"),
                                                        "Nascimento": dados_aluno.get("nascimento", "N/A"),
                                                        "Turma": dados_aluno.get("turma", "N/A")
                                                    }
                                                    enviar_para_planilha(client, dados_envio, resultado, PLANILHA_ID, questoes_detectadas=questoes_detectadas)
                                                
                                            except Exception as e:
                                                print(f"   ❌ Erro na página {pagina_idx}: {e}")
                                        
                                        # Limpar imagens temporárias do PDF
                                        
                                        # Se todas as páginas forem do mesmo ano, vai para aquela pasta
                                        # Se houver mix, vai para pasta do 9° ano (prioridade)
                                        if not pastas_detectadas:
                                            pasta_destino_pdf = DRIVER_FOLDER_9ANO  # Padrão
                                            num_questoes_pdf = 52
                                        elif len(set(pastas_detectadas)) == 1:
                                            # Todas as páginas do mesmo ano
                                            pasta_destino_pdf = pastas_detectadas[0]
                                            num_questoes_pdf = 44 if pasta_destino_pdf == DRIVER_FOLDER_5ANO else 52
                                            ano_str = "5° ano" if num_questoes_pdf == 44 else "9° ano"
                                            print(f"\n📁 PDF será movido para: {ano_str} (todas as páginas são do mesmo ano)")
                                        else:
                                            # Mix de anos - vai para 9° ano
                                            pasta_destino_pdf = DRIVER_FOLDER_9ANO
                                            num_questoes_pdf = 52
                                            print(f"\n📁 PDF será movido para: 9° ano (PDF contém páginas de anos diferentes)")
                                        
                                        # Marcar PDF como processado (ID + NOME + PASTA DESTINO)
                                        nome_sem_ext = os.path.splitext(pdf_info['name'])[0].lower()
                                        arquivos_processados_agora.append({
                                            'id': pdf_info['id'],
                                            'nome_sem_ext': nome_sem_ext,
                                            'nome_original': pdf_info['name'],
                                            'pasta_destino': pasta_destino_pdf,  # 🆕 Usa pasta detectada
                                            'num_questoes': num_questoes_pdf
                                        })
                                        print(f"\n✅ PDF processado: {pdf_info['name']}")
                                        
                                    except Exception as e:
                                        print(f"   ❌ Erro ao processar PDF: {e}")
                                        import traceback
                                        traceback.print_exc()
                            
                            # ═══════════════════════════════════════════════════
                            # PROCESSAR IMAGENS (PÁGINA ÚNICA)
                            # ═══════════════════════════════════════════════════
                            if imagens_para_processar:
                                print(f"\n{'='*80}")
                                print(f"🖼️ PROCESSANDO {len(imagens_para_processar)} IMAGEM(NS)")
                                print(f"{'='*80}")
                                
                                for img_idx, cartao_info in enumerate(imagens_para_processar, 1):
                                    try:
                                        print(f"\n🔄 [{img_idx}/{len(imagens_para_processar)}] {cartao_info['name']}")
                                        
                                        # Baixar imagem
                                        request = drive_service.files().get_media(fileId=cartao_info['id'])
                                        cartao_path = os.path.join(pasta_temp, cartao_info['name'])
                                        with open(cartao_path, 'wb') as f:
                                            downloader = MediaIoBaseDownload(f, request)
                                            done = False
                                            while not done:
                                                status, done = downloader.next_chunk()
                                        
                                        # Converter para P&B se habilitado
                                        if converter_pb:
                                            cartao_path = converter_para_preto_e_branco(cartao_path, threshold=threshold_pb, salvar=True)
                                        
                                        # Processar cartão
                                        aluno_img = preprocessar_arquivo(cartao_path, f"aluno_{img_idx}")
                                        
                                        # 🆕 USAR EXTRAÇÃO OTIMIZADA (1 chamada única ao Gemini)
                                        num_questoes_aluno = None
                                        dados_aluno = None
                                        
                                        # PRIORIDADE 1: Tentar Gemini primeiro
                                        if model_gemini:
                                            dados_completos = extrair_dados_completos_com_gemini(
                                                model_gemini, 
                                                aluno_img,
                                                nome_arquivo=cartao_info['name']
                                            )
                                            if dados_completos:
                                                dados_aluno = dados_completos
                                                num_questoes_aluno = dados_completos.get('num_questoes')
                                                if num_questoes_aluno:
                                                    print(f"   ✅ Gemini detectou com sucesso: {num_questoes_aluno} questões")
                                        
                                        # FALLBACK: Se Gemini falhar, usar OCR direto
                                        if not num_questoes_aluno:
                                            print(f"   ⚠️ Gemini falhou - usando OCR como fallback")
                                            num_questoes_aluno = detectar_ano_com_ocr_direto(aluno_img, debug=False)
                                            print(f"   📊 OCR (fallback) detectou: {num_questoes_aluno} questões")
                                        
                                        # Se dados do aluno não foram extraídos, usar OCR
                                        if not dados_aluno:
                                            dados_aluno = extrair_cabecalho_com_ocr_fallback(aluno_img)
                                        
                                        if not dados_aluno or dados_aluno.get("aluno") == "N/A":
                                            dados_aluno = {
                                                "escola": "N/A",
                                                "aluno": os.path.splitext(cartao_info['name'])[0],
                                                "turma": "N/A",
                                                "nascimento": "N/A"
                                            }
                                        
                                        print(f"   🔍 DEBUG - Dados extraídos: Escola={dados_aluno.get('escola')}, Aluno={dados_aluno.get('aluno')}, Turma={dados_aluno.get('turma')}, Nasc={dados_aluno.get('nascimento')}")
                                        
                                        # 🆕 SELECIONAR PASTA DE DESTINO BASEADA NO ANO DETECTADO
                                        if num_questoes_aluno == 44:
                                            pasta_destino_atual = DRIVER_FOLDER_5ANO
                                            print(f"   📁 Destino: Pasta 5° ano (44 questões)")
                                        else:  # 52 questões
                                            pasta_destino_atual = DRIVER_FOLDER_9ANO
                                            print(f"   📁 Destino: Pasta 9° ano (52 questões)")
                                        
                                        # Detectar respostas (usando número detectado)
                                        respostas_aluno = detectar_respostas_por_tipo(aluno_img, num_questoes=num_questoes_aluno, debug=False)
                                        questoes_detectadas = sum(1 for r in respostas_aluno if r != '?')
                                        
                                        # 🆕 COMPARAR COM O GABARITO CORRETO
                                        respostas_gabarito_correto = gabaritos_dict.get(num_questoes_aluno)
                                        if not respostas_gabarito_correto:
                                            print(f"   ❌ Gabarito de {num_questoes_aluno} questões não disponível!")
                                            continue
                                        
                                        resultado = comparar_respostas(respostas_gabarito_correto, respostas_aluno)
                                        
                                        # Exibir resumo formatado
                                        print(f"\n{'─'*60}")
                                        print(f"👤 {dados_aluno.get('aluno', 'N/A')}")
                                        print(f"📚 Turma: {dados_aluno.get('turma', 'N/A')} | Escola: {dados_aluno.get('escola', 'N/A')}")
                                        print(f"✅ Acertos: {resultado['acertos']}")
                                        print(f"❌ Erros: {resultado['erros']}")
                                        print(f"📊 Percentual: {resultado['percentual']:.1f}%")
                                        
                                        # Exibir respostas do aluno
                                        print(f"\n📝 Respostas:")
                                        exibir_gabarito_simples(respostas_aluno)
                                        print(f"{'─'*60}")
                                        
                                        # Enviar para Google Sheets
                                        if client and PLANILHA_ID:
                                            dados_envio = {
                                                "Escola": dados_aluno.get("escola", "N/A"),
                                                "Aluno": dados_aluno.get("aluno", "N/A"),
                                                "Nascimento": dados_aluno.get("nascimento", "N/A"),
                                                "Turma": dados_aluno.get("turma", "N/A")
                                            }
                                            enviar_para_planilha(client, dados_envio, resultado, PLANILHA_ID, questoes_detectadas=questoes_detectadas)
                                        
                                        # Marcar como processado (ID + NOME + PASTA DESTINO)
                                        nome_sem_ext = os.path.splitext(cartao_info['name'])[0].lower()
                                        arquivos_processados_agora.append({
                                            'id': cartao_info['id'],
                                            'nome_sem_ext': nome_sem_ext,
                                            'nome_original': cartao_info['name'],
                                            'pasta_destino': pasta_destino_atual,  # 🆕 Guardar pasta específica
                                            'num_questoes': num_questoes_aluno  # 🆕 Guardar número de questões
                                        })
                                        
                                    except Exception as e:
                                        print(f"   ❌ Erro: {e}")
                            
                            # 3. Mover arquivos processados no Drive para pasta correta (5º ou 9º ano)
                            if mover_processados and arquivos_processados_agora:
                                print(f"\n📦 Movendo {len(arquivos_processados_agora)} cartões para pastas de destino...")
                                
                                # 🆕 MOVER CADA ARQUIVO PARA SUA PASTA ESPECÍFICA
                                for arquivo_proc in arquivos_processados_agora:
                                    pasta_destino_arquivo = arquivo_proc.get('pasta_destino')
                                    num_questoes_arquivo = arquivo_proc.get('num_questoes', 52)
                                    
                                    if pasta_destino_arquivo:
                                        ano_str = "5° ano" if num_questoes_arquivo == 44 else "9° ano"
                                        print(f"   📁 {arquivo_proc['nome_original']} → {ano_str}")
                                        
                                        mover_arquivo_no_drive(
                                            drive_service,
                                            arquivo_proc['id'],
                                            pasta_drive_id,
                                            pasta_destino_arquivo,  # 🆕 Usa pasta específica do aluno
                                            arquivo_proc['nome_original']
                                        )
                            
                            # 4. Atualizar histórico com IDs e NOMES processados
                            historico['ids'].update(item['id'] for item in arquivos_processados_agora)
                            historico['nomes'].update(item['nome_sem_ext'] for item in arquivos_processados_agora)
                            
                            # Converter para formato de salvamento
                            lista_para_salvar = [
                                {
                                    'id': arquivo_id,
                                    'nome_sem_ext': nome_sem_ext,
                                    'processado_em': datetime.now().isoformat()
                                }
                                for arquivo_id, nome_sem_ext in zip(
                                    sorted(historico['ids']),
                                    sorted(historico['nomes'])
                                )
                            ]
                            
                            # Reconstruir lista correta mantendo correspondência ID-Nome
                            lista_correta = []
                            for item in arquivos_processados_agora:
                                lista_correta.append({
                                    'id': item['id'],
                                    'nome_sem_ext': item['nome_sem_ext'],
                                    'processado_em': datetime.now().isoformat()
                                })
                            
                            # Adicionar itens antigos do histórico
                            if os.path.exists(historico_file):
                                try:
                                    with open(historico_file, 'r', encoding='utf-8') as f:
                                        data_antiga = json.load(f)
                                        arquivos_antigos = data_antiga.get('arquivos_processados', [])
                                        if arquivos_antigos and isinstance(arquivos_antigos[0], dict):
                                            for item_antigo in arquivos_antigos:
                                                if item_antigo['id'] not in [x['id'] for x in lista_correta]:
                                                    lista_correta.append(item_antigo)
                                except:
                                    pass
                            
                            salvar_historico(lista_correta)
                            
                            # Limpar pasta temporária
                            shutil.rmtree(pasta_temp, ignore_errors=True)
                            
                            print(f"\n✅ Processamento concluído!")
                            print(f"📊 Novos processados: {len(arquivos_processados_agora)}")
                            print(f"📝 Total no histórico: {len(historico['ids'])} IDs / {len(historico['nomes'])} Nomes")
                                
                        except Exception as e:
                            print(f"❌ Erro durante processamento: {e}")
                            import traceback
                            traceback.print_exc()
                            print("🔄 Continuando monitoramento...")
                    else:
                        print("� Nenhum cartão para processar")
                    
                    print(f"⏰ Próxima verificação em {args.intervalo} minuto(s)...")
                    time.sleep(args.intervalo * 60)
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"❌ Erro na verificação #{contador_verificacoes}: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"🔄 Continuando... próxima verificação em {args.intervalo} minutos")
                    time.sleep(args.intervalo * 60)
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoramento interrompido pelo usuário")
            print(f"Total de verificações realizadas: {contador_verificacoes}")
        
        exit(0)

    # 🆕 Sistema agora é totalmente automatizado via modo --monitor
    # Para usar o sistema, execute: python script.py --monitor
    print("\n" + "=" * 80)
    print("ℹ️  MODO DE USO")
    print("=" * 80)
    print("O sistema agora funciona em modo de monitoramento automatizado.")
    print("\nPara iniciar o sistema, use:")
    print("  python script.py --monitor")
    print("\nO sistema irá:")
    print("  • Carregar automaticamente os 2 gabaritos (44 e 52 questões)")
    print("  • Detectar via IA e OCR o ano de cada aluno")
    print("  • Corrigir usando o gabarito correto")
    print("  • Enviar para a planilha correta (5° ou 9° ano)")
    print("  • Mover para a pasta correta no Drive")
    print("=" * 80)
    
