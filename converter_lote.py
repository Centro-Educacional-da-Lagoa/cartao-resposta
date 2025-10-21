"""
🎨 CONVERSOR EM LOTE - PRETO E BRANCO

Converte todas as imagens de uma pasta para preto e branco
"""

import cv2
import os
import sys

def converter_pasta(pasta, threshold=180):
    """
    Converte todas as imagens de uma pasta para P&B
    """
    print(f"\n{'='*60}")
    print(f"🎨 CONVERSOR EM LOTE - PRETO E BRANCO")
    print(f"{'='*60}")
    print(f"📁 Pasta: {pasta}")
    print(f"🎚️ Threshold: {threshold}")
    print(f"{'='*60}\n")
    
    # Extensões suportadas
    extensoes = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    
    # Listar arquivos
    arquivos = [f for f in os.listdir(pasta) 
                if f.lower().endswith(extensoes) and '_pb' not in f.lower()]
    
    if not arquivos:
        print("❌ Nenhuma imagem encontrada na pasta!")
        return
    
    print(f"📋 Encontradas {len(arquivos)} imagens para converter:\n")
    
    convertidos = 0
    erros = 0
    
    for i, arquivo in enumerate(arquivos, 1):
        caminho_original = os.path.join(pasta, arquivo)
        
        print(f"[{i}/{len(arquivos)}] Processando: {arquivo}...", end=' ')
        
        try:
            # Carregar imagem
            img = cv2.imread(caminho_original)
            if img is None:
                print("❌ Erro ao carregar")
                erros += 1
                continue
            
            # Converter para escala de cinza
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Aplicar threshold
            _, img_pb = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Salvar
            nome_base = os.path.splitext(arquivo)[0]
            extensao = os.path.splitext(arquivo)[1]
            output_path = os.path.join(pasta, f"{nome_base}_pb{extensao}")
            
            cv2.imwrite(output_path, img_pb)
            print(f"✅ Convertido")
            convertidos += 1
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            erros += 1
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTADO:")
    print(f"   ✅ Convertidos: {convertidos}")
    print(f"   ❌ Erros: {erros}")
    print(f"{'='*60}\n")

def main():
    if len(sys.argv) < 2:
        print("\n🎨 CONVERSOR EM LOTE - PRETO E BRANCO")
        print("\nUSO:")
        print("  python converter_lote.py <pasta> [threshold]")
        print("\nEXEMPLOS:")
        print("  python converter_lote.py ./gabaritos")
        print("  python converter_lote.py ./gabaritos 150")
        print("\nTHRESHOLD:")
        print("  - Valor entre 0-255 (padrão: 180)")
        print("  - Menor = mais preto, Maior = mais branco")
        sys.exit(1)
    
    pasta = sys.argv[1]
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
    
    if not os.path.exists(pasta):
        print(f"❌ Pasta não encontrada: {pasta}")
        sys.exit(1)
    
    converter_pasta(pasta, threshold)

if __name__ == "__main__":
    main()
