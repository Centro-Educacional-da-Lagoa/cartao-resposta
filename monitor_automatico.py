#!/usr/bin/env python3
"""Entrada de compatibilidade para o monitor principal baseado em Vultr S3."""

import argparse
import os
import sys

from dotenv import load_dotenv

from storage_google_drive import GoogleDriveStorage
from storage_vultr import VultrS3Storage


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Monitor automático de cartões-resposta no Vultr S3"
    )
    parser.add_argument(
        "--intervalo",
        type=int,
        default=5,
        help="Intervalo de verificação em minutos (padrão: 5)",
    )
    parser.add_argument(
        "--testar",
        action="store_true",
        help="Testa a conexão e lista os arquivos aguardando processamento",
    )
    args = parser.parse_args()

    if args.testar:
        storage_s3 = VultrS3Storage.from_env()
        uploads_s3 = storage_s3.listar_uploads()
        gabaritos = storage_s3.listar_gabaritos()
        print(f"Bucket: {storage_s3.config.bucket}")
        print(f"Arquivos na entrada S3: {len(uploads_s3)}")
        print(f"Gabaritos encontrados: {len(gabaritos)}")

        try:
            storage_drive = GoogleDriveStorage.from_env()
            uploads_drive = storage_drive.listar_uploads()
            cartoes_drive = [
                arquivo
                for arquivo in uploads_drive
                if arquivo["name"].lower().endswith((".pdf", ".png", ".jpg", ".jpeg"))
                and "gabarito" not in arquivo["name"].lower()
            ]
            print(f"Cartões na entrada Google Drive: {len(cartoes_drive)}")
        except Exception as e:
            print(f"Google Drive indisponível: {e}")
        return

    script_path = os.path.join(os.path.dirname(__file__), "script.py")
    os.execv(
        sys.executable,
        [
            sys.executable,
            script_path,
            "--monitor",
            "--intervalo",
            str(args.intervalo),
        ],
    )


if __name__ == "__main__":
    main()
