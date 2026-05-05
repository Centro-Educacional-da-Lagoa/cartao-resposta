"""
Teste visual de perspectiva para cartões/folhas.

Uso:
    python teste_visual_perspectiva.py --input <pasta_imagens> --output <pasta_debug>
"""

import argparse
import os
from typing import List, Dict

import cv2

from script import normalizar_documento_para_omr


EXTENSOES_IMAGEM = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def listar_imagens(diretorio: str) -> List[str]:
    arquivos = []
    for nome in sorted(os.listdir(diretorio)):
        caminho = os.path.join(diretorio, nome)
        if os.path.isfile(caminho) and nome.lower().endswith(EXTENSOES_IMAGEM):
            arquivos.append(nome)
    return arquivos


def salvar_imagem_final(caminho_origem: str, caminho_destino: str) -> bool:
    imagem = cv2.imread(caminho_origem)
    if imagem is None:
        return False
    cv2.imwrite(caminho_destino, imagem)
    return True


def escrever_resumo(caminho_saida: str, linhas: List[Dict[str, str]]) -> None:
    with open(caminho_saida, "w", encoding="utf-8") as f:
        f.write("RESUMO - TESTE VISUAL DE PERSPECTIVA\n")
        f.write("=" * 80 + "\n")
        f.write("arquivo | perspectiva | deskew | motivo | saida_final\n")
        f.write("-" * 80 + "\n")
        for linha in linhas:
            f.write(
                f"{linha['arquivo']} | {linha['perspectiva']} | "
                f"{linha['deskew']} | {linha['motivo']} | {linha['saida_final']}\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera teste visual de perspectiva (antes/depois) para um lote de imagens."
    )
    parser.add_argument("--input", required=True, help="Pasta com imagens de entrada")
    parser.add_argument("--output", required=True, help="Pasta para saída dos artefatos")
    parser.add_argument(
        "--no-perspectiva",
        action="store_true",
        help="Executa apenas deskew (sem retificação de perspectiva)"
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.isdir(input_dir):
        print(f"❌ Pasta de entrada não encontrada: {input_dir}")
        return 1

    os.makedirs(output_dir, exist_ok=True)
    arquivos = listar_imagens(input_dir)

    if not arquivos:
        print(f"❌ Nenhuma imagem suportada encontrada em: {input_dir}")
        print("   Formatos: PNG, JPG, JPEG, BMP, TIFF, WEBP")
        return 1

    print("=" * 80)
    print("🧪 TESTE VISUAL DE PERSPECTIVA")
    print("=" * 80)
    print(f"📁 Entrada: {input_dir}")
    print(f"📁 Saída:   {output_dir}")
    print(f"🧭 Perspectiva: {'DESATIVADA' if args.no_perspectiva else 'ATIVADA'}")
    print(f"🖼️ Imagens encontradas: {len(arquivos)}")
    print("=" * 80)

    resumo = []

    for indice, nome_arquivo in enumerate(arquivos, 1):
        caminho_entrada = os.path.join(input_dir, nome_arquivo)
        nome_base = os.path.splitext(nome_arquivo)[0]
        pasta_debug_arquivo = os.path.join(output_dir, nome_base)
        os.makedirs(pasta_debug_arquivo, exist_ok=True)

        print(f"\n🔄 [{indice:02d}/{len(arquivos):02d}] {nome_arquivo}")

        resultado = normalizar_documento_para_omr(
            image_path=caminho_entrada,
            aplicar_perspectiva=not args.no_perspectiva,
            debug=True,
            debug_dir=pasta_debug_arquivo,
        )

        caminho_final = str(resultado.get("output_path", caminho_entrada))
        caminho_final_debug = os.path.join(pasta_debug_arquivo, f"{nome_base}_05_normalizada_final.png")
        final_salvo = salvar_imagem_final(caminho_final, caminho_final_debug)

        status_perspectiva = str(resultado.get("status_visual", "ignored"))
        deskew_aplicado = "sim" if bool(resultado.get("deskew_aplicado", False)) else "não"
        motivo = str(resultado.get("motivo", "")).strip() or "-"

        print(
            f"   ✅ perspectiva={status_perspectiva} | "
            f"deskew={deskew_aplicado} | final={'ok' if final_salvo else 'erro'}"
        )
        if motivo != "-":
            print(f"   ℹ️ motivo: {motivo}")

        resumo.append(
            {
                "arquivo": nome_arquivo,
                "perspectiva": status_perspectiva,
                "deskew": deskew_aplicado,
                "motivo": motivo.replace("\n", " "),
                "saida_final": caminho_final_debug if final_salvo else caminho_final,
            }
        )

    resumo_path = os.path.join(output_dir, "resumo_processamento.txt")
    escrever_resumo(resumo_path, resumo)

    print("\n" + "=" * 80)
    print("✅ TESTE VISUAL CONCLUÍDO")
    print(f"📄 Resumo: {resumo_path}")
    print("📌 Artefatos por imagem:")
    print("   01_original, 02_cantos_detectados, 03_retificada, 04_comparativo, 05_normalizada_final")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
