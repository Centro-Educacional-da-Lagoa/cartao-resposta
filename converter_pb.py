"""
🎨 CONVERSOR DE IMAGENS PARA PRETO E BRANCO

Este script converte imagens coloridas em preto e branco puro (binarizado)
para testar a detecção de bolhas no cartão resposta.

USO:
    python converter_pb.py <caminho_da_imagem> [threshold]
    
EXEMPLOS:
    python converter_pb.py cartao.jpg
    python converter_pb.py cartao.jpg 180
    python converter_pb.py cartao.jpg 150  # Mais preto
    python converter_pb.py cartao.jpg 200  # Mais branco

THRESHOLD:
    - Valor entre 0-255 (padrão: 180)
    - Menor valor = Mais pixels ficam pretos
    - Maior valor = Mais pixels ficam brancos
    - 180 = Recomendado (baseado na imagem de referência)
"""

import cv2
import numpy as np
import os
import sys
from PIL import Image

def converter_para_pb(image_path, threshold=180, mostrar_preview=True):
    """
    Converte uma imagem para preto e branco puro
    """
    print(f"\n{'='*60}")
    print(f"🎨 CONVERSOR PARA PRETO E BRANCO")
    print(f"{'='*60}")
    print(f"📁 Arquivo: {os.path.basename(image_path)}")
    print(f"🎚️ Threshold: {threshold}")
    print(f"{'='*60}\n")
    
    try:
        # Verificar se arquivo existe
        if not os.path.exists(image_path):
            print(f"❌ Arquivo não encontrado: {image_path}")
            return None
        
        # Carregar imagem
        print("⏳ Carregando imagem...")
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Não foi possível carregar a imagem")
            return None
        
        altura, largura = img.shape[:2]
        print(f"✅ Imagem carregada: {largura}x{altura} pixels")
        
        # Converter para escala de cinza
        print("⏳ Convertendo para escala de cinza...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold para preto e branco puro
        print(f"⏳ Aplicando threshold ({threshold})...")
        _, img_pb = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Calcular estatísticas
        pixels_pretos = np.sum(img_pb == 0)
        pixels_brancos = np.sum(img_pb == 255)
        total_pixels = img_pb.size
        percentual_preto = (pixels_pretos / total_pixels) * 100
        percentual_branco = (pixels_brancos / total_pixels) * 100
        
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   ⚫ Pixels pretos: {pixels_pretos:,} ({percentual_preto:.1f}%)")
        print(f"   ⚪ Pixels brancos: {pixels_brancos:,} ({percentual_branco:.1f}%)")
        
        # Criar nome do arquivo de saída
        nome_base = os.path.splitext(image_path)[0]
        extensao = os.path.splitext(image_path)[1]
        output_path = f"{nome_base}_pb_t{threshold}{extensao}"
        
        # Salvar imagem convertida
        print(f"\n💾 Salvando imagem convertida...")
        cv2.imwrite(output_path, img_pb)
        print(f"✅ Salvo como: {os.path.basename(output_path)}")
        
        # Mostrar preview comparativo
        if mostrar_preview:
            print(f"\n👁️ Abrindo preview comparativo...")
            
            # Criar preview lado a lado
            # Redimensionar se muito grande
            max_width = 800
            if largura > max_width:
                scale = max_width / largura
                new_width = max_width
                new_height = int(altura * scale)
                gray_resized = cv2.resize(gray, (new_width, new_height))
                pb_resized = cv2.resize(img_pb, (new_width, new_height))
            else:
                gray_resized = gray
                pb_resized = img_pb
            
            # Juntar lado a lado
            preview = np.hstack([gray_resized, pb_resized])
            
            # Adicionar legendas
            cv2.putText(preview, "ORIGINAL (Grayscale)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 128, 2)
            cv2.putText(preview, f"PRETO E BRANCO (Threshold={threshold})", 
                       (gray_resized.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 128, 2)
            
            # Mostrar preview
            cv2.imshow("Preview - Pressione qualquer tecla para fechar", preview)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"✅ CONVERSÃO CONCLUÍDA COM SUCESSO!")
        print(f"{'='*60}\n")
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        return None

def main():
    """
    Função principal
    """
    # Verificar argumentos
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n❌ ERRO: Forneça o caminho da imagem!")
        print("\nEXEMPLO:")
        print("  python converter_pb.py minha_imagem.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Threshold opcional (padrão: 180)
    threshold = 180
    if len(sys.argv) >= 3:
        try:
            threshold = int(sys.argv[2])
            if threshold < 0 or threshold > 255:
                print("⚠️ Threshold deve estar entre 0 e 255. Usando 180.")
                threshold = 180
        except ValueError:
            print("⚠️ Threshold inválido. Usando 180.")
            threshold = 180
    
    # Converter
    resultado = converter_para_pb(image_path, threshold)
    
    if resultado:
        print(f"✅ Arquivo convertido salvo em: {resultado}")
        print(f"\n💡 DICA: Para ajustar o contraste, teste diferentes valores de threshold:")
        print(f"   - Mais preto:  python converter_pb.py {image_path} 150")
        print(f"   - Mais branco: python converter_pb.py {image_path} 200")
    else:
        print(f"❌ Falha na conversão")
        sys.exit(1)

if __name__ == "__main__":
    main()
