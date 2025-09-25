# ===========================================
# SISTEMA DE PROCESSAMENTO DE PDF PARA CARTAO RESPOSTA
# ===========================================
# 
# Este modulo adiciona suporte para processar arquivos PDF
# convertendo-os em imagens para serem processados pelo sistema principal
# ===========================================

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFPageCountError, PDFPopplerTimeoutError
    PDF_SUPPORT_AVAILABLE = True
    print("OK - Suporte a PDF disponivel")
except ImportError:
    PDF_SUPPORT_AVAILABLE = False
    print("AVISO - pdf2image nao disponivel. Instale com: pip install pdf2image")

from PIL import Image
import cv2
import numpy as np

# ===========================================
# CONFIGURACOES PARA PROCESSAMENTO DE PDF
# ===========================================

# Qualidade DPI para conversao (maior = melhor qualidade, mas arquivos maiores)
DEFAULT_DPI = 300  # 300 DPI e ideal para OCR/OMR

# Formato de saida das imagens convertidas
DEFAULT_FORMAT = 'PNG'  # PNG mantem qualidade, JPEG e menor

def is_pdf_file(file_path: str) -> bool:
    """
    Verifica se o arquivo e um PDF
    """
    return Path(file_path).suffix.lower() == '.pdf'

def convert_pdf_to_images(pdf_path: str, dpi: int = DEFAULT_DPI, 
                         output_format: str = DEFAULT_FORMAT) -> List[str]:
    """
    Converte PDF em lista de imagens
    """
    if not PDF_SUPPORT_AVAILABLE:
        raise Exception("pdf2image nao esta instalado. Execute: pip install pdf2image")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Arquivo PDF nao encontrado: {pdf_path}")
    
    print(f"Convertendo PDF para imagens: {pdf_path}")
    print(f"   DPI: {dpi}")
    print(f"   Formato: {output_format}")
    
    try:
        # Detectar se poppler esta disponivel no sistema
        poppler_path = None
        
        # Tentar localizar poppler no Windows
        possible_poppler_paths = [
            r"C:\Program Files\poppler\bin",
            r"C:\Program Files (x86)\poppler\bin", 
            r"C:\poppler\bin",
            os.path.join(os.getcwd(), "poppler", "bin")
        ]
        
        for path in possible_poppler_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "pdftoppm.exe")):
                poppler_path = path
                print(f"OK - Poppler encontrado em: {poppler_path}")
                break
        
        if not poppler_path:
            print("AVISO - Poppler nao encontrado, tentando sem especificar caminho...")
        
        # Converter PDF para imagens
        try:
            if poppler_path:
                images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
            else:
                images = convert_from_path(pdf_path, dpi=dpi)
        except Exception as e:
            if "poppler" in str(e).lower():
                raise Exception(
                    f"ERRO relacionado ao Poppler: {e}\n\n"
                    "SOLUCAO:\n"
                    "1. Baixe poppler para Windows em: https://github.com/oschwartz10612/poppler-windows/releases\n"
                    "2. Extraia para C:\\poppler\n"
                    "3. Ou instale via Chocolatey: choco install poppler (como administrador)\n"
                    "4. Ou adicione poppler/bin ao PATH do sistema"
                )
            else:
                raise e
        
        print(f"OK - PDF convertido! Total de paginas: {len(images)}")
        
        # Salvar imagens temporarias
        temp_files = []
        base_name = Path(pdf_path).stem
        
        for i, image in enumerate(images):
            # Nome do arquivo temporario
            temp_filename = f"{base_name}_page_{i+1}.{output_format.lower()}"
            temp_path = os.path.join(os.path.dirname(pdf_path), temp_filename)
            
            # Salvar imagem
            image.save(temp_path, format=output_format, quality=95, dpi=(dpi, dpi))
            temp_files.append(temp_path)
            print(f"   Pagina {i+1} salva como: {temp_filename}")
        
        return temp_files
        
    except Exception as e:
        print(f"ERRO ao converter PDF: {e}")
        raise

def get_best_page_for_processing(image_paths: List[str]) -> str:
    """
    Seleciona a melhor pagina para processamento baseado no conteudo
    """
    if len(image_paths) == 1:
        return image_paths[0]
    
    print(f"Analisando {len(image_paths)} paginas para encontrar a melhor...")
    
    best_page = None
    best_score = 0
    
    for i, img_path in enumerate(image_paths):
        try:
            # Carregar imagem
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Converter para escala de cinza
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calcular score baseado no numero de circulos detectados
            circles_score = count_circular_elements(gray)
            
            # Calcular score baseado na quantidade de texto
            text_score = estimate_text_density(gray)
            
            # Score combinado (prioriza circulos para cartoes resposta)
            combined_score = circles_score * 2 + text_score
            
            print(f"   Pagina {i+1}: {circles_score} circulos, {text_score} texto, score: {combined_score}")
            
            if combined_score > best_score:
                best_score = combined_score
                best_page = img_path
                
        except Exception as e:
            print(f"   ERRO ao analisar pagina {i+1}: {e}")
            continue
    
    if best_page:
        page_num = image_paths.index(best_page) + 1
        print(f"OK - Melhor pagina selecionada: Pagina {page_num}")
        return best_page
    else:
        print("AVISO - Nao foi possivel determinar a melhor pagina, usando a primeira")
        return image_paths[0]

def count_circular_elements(gray_image) -> int:
    """
    Conta elementos circulares na imagem (indicativo de cartao resposta)
    """
    try:
        # Aplicar threshold para detectar elementos escuros
        _, thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circular_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 1000:  # Tamanho tipico de bolhas
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # Razoavelmente circular
                        circular_count += 1
        
        return circular_count
        
    except Exception:
        return 0

def estimate_text_density(gray_image) -> int:
    """
    Estima densidade de texto na imagem
    """
    try:
        # Detectar bordas para contar elementos de texto
        edges = cv2.Canny(gray_image, 50, 150)
        text_elements = cv2.countNonZero(edges)
        
        # Normalizar baseado no tamanho da imagem
        height, width = gray_image.shape
        density = text_elements / (height * width) * 10000
        
        return int(density)
        
    except Exception:
        return 0

def cleanup_temp_files(file_paths: List[str]) -> None:
    """
    Remove arquivos temporarios criados durante a conversao
    """
    print("Limpando arquivos temporarios...")
    
    cleaned = 0
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleaned += 1
                print(f"   Removido: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"   AVISO - Erro ao remover {file_path}: {e}")
    
    print(f"OK - {cleaned}/{len(file_paths)} arquivos temporarios removidos")

def process_pdf_file(pdf_path: str, keep_temp_files: bool = False) -> Tuple[str, Optional[List[str]]]:
    """
    Processa arquivo PDF e retorna o caminho da melhor imagem
    """
    if not is_pdf_file(pdf_path):
        # Se nao e PDF, retorna o proprio arquivo
        return pdf_path, None
    
    print(f"\nPROCESSANDO PDF: {os.path.basename(pdf_path)}")
    
    try:
        # Converter PDF para imagens
        temp_images = convert_pdf_to_images(pdf_path)
        
        if not temp_images:
            raise Exception("Nenhuma imagem foi gerada do PDF")
        
        # Selecionar melhor pagina
        best_image = get_best_page_for_processing(temp_images)
        
        # Limpar arquivos temporarios se solicitado
        temp_files_to_return = temp_images.copy() if keep_temp_files else None
        
        if not keep_temp_files:
            # Manter apenas a melhor imagem, remover as outras
            files_to_remove = [f for f in temp_images if f != best_image]
            if files_to_remove:
                cleanup_temp_files(files_to_remove)
        
        print(f"OK - PDF processado com sucesso!")
        print(f"   Melhor imagem: {os.path.basename(best_image)}")
        
        return best_image, temp_files_to_return
        
    except Exception as e:
        print(f"ERRO ao processar PDF: {e}")
        raise

def setup_pdf_support() -> bool:
    """
    Configura e valida suporte a PDF
    """
    print("\nCONFIGURANDO SUPORTE A PDF...")
    
    # Verificar se pdf2image esta disponivel
    if not PDF_SUPPORT_AVAILABLE:
        print("ERRO - pdf2image nao esta instalado")
        print("Para instalar: pip install pdf2image")
        return False
    
    print("OK - pdf2image instalado")
    
    # Verificar se poppler esta disponivel
    try:
        # Tentar converter um PDF de teste (arquivo vazio)
        test_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n174\n%%EOF"
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(test_pdf_content)
            temp_pdf_path = temp_pdf.name
        
        try:
            # Tentar converter
            test_images = convert_from_path(temp_pdf_path, dpi=72)
            print("OK - Poppler funcionando corretamente")
            success = True
        except Exception as e:
            if "poppler" in str(e).lower():
                print("AVISO - Poppler nao encontrado no sistema")
                print("Solucoes:")
                print("   1. Baixar de: https://github.com/oschwartz10612/poppler-windows/releases")
                print("   2. Extrair para C:\\poppler")
                print("   3. Ou executar como admin: choco install poppler")
                success = False
            else:
                print(f"ERRO inesperado: {e}")
                success = False
        finally:
            # Limpar arquivo de teste
            try:
                os.unlink(temp_pdf_path)
            except:
                pass
        
    except Exception as e:
        print(f"ERRO ao testar PDF: {e}")
        success = False
    
    if success:
        print("OK - Suporte a PDF configurado com sucesso!")
    else:
        print("ERRO - Suporte a PDF nao esta funcionando completamente")
    
    return success

# ===========================================
# EXEMPLO DE USO
# ===========================================

if __name__ == "__main__":
    print("TESTANDO PROCESSAMENTO DE PDF...")
    
    # Configurar suporte
    if not setup_pdf_support():
        print("ERRO - Nao foi possivel configurar suporte a PDF")
        exit(1)
    
    # Testar com arquivos do projeto se existirem
    test_files = [
        "resposta_aluno.pdf",
        "resposta_gabarito_teste.pdf"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nTestando com: {test_file}")
            try:
                best_image, temp_files = process_pdf_file(test_file, keep_temp_files=True)
                print(f"OK - Sucesso! Melhor imagem: {best_image}")
                
                if temp_files:
                    print(f"Arquivos temporarios criados: {len(temp_files)}")
                    # Limpar apos teste
                    cleanup_temp_files(temp_files)
                    
            except Exception as e:
                print(f"ERRO: {e}")
        else:
            print(f"AVISO - Arquivo nao encontrado: {test_file}")