#!/usr/bin/env python3
"""Migra uma única vez os quatro gabaritos do Google Drive para o Vultr S3."""

from __future__ import annotations

import argparse
import io
import json
import os
import posixpath

from dotenv import load_dotenv
from botocore.exceptions import ClientError
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from anos_escolares import ANOS_ESCOLARES, nome_gabarito
from storage_vultr import VultrS3Storage


EXTENSOES_GABARITO = (".png", ".jpg", ".jpeg")


def configurar_drive():
    credentials_json = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()
    if credentials_json:
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(credentials_json),
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
    else:
        credentials = service_account.Credentials.from_service_account_file(
            "credenciais_google.json",
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )

    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def listar_gabaritos_drive(drive, pasta_id: str) -> dict:
    resposta = drive.files().list(
        q=f"'{pasta_id}' in parents and trashed = false",
        fields="files(id, name, mimeType)",
        pageSize=1000,
    ).execute()
    arquivos = resposta.get("files", [])
    por_nome = {arquivo.get("name", "").lower(): arquivo for arquivo in arquivos}
    encontrados = {}

    for ano_escolar in ANOS_ESCOLARES:
        nome_base = nome_gabarito(ano_escolar)
        arquivo = next(
            (
                por_nome.get(f"{nome_base}{extensao}")
                for extensao in EXTENSOES_GABARITO
                if por_nome.get(f"{nome_base}{extensao}")
            ),
            None,
        )
        if arquivo:
            encontrados[ano_escolar] = arquivo

    return encontrados


def baixar_drive(drive, arquivo_id: str) -> bytes:
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(
        buffer,
        drive.files().get_media(fileId=arquivo_id),
    )
    concluido = False
    while not concluido:
        _, concluido = downloader.next_chunk()
    return buffer.getvalue()


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Migra os gabaritos atuais do Google Drive para o Vultr S3"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas lista os gabaritos encontrados, sem enviar ao S3",
    )
    parser.add_argument(
        "--sobrescrever",
        action="store_true",
        help="Sobrescreve gabaritos que já existam no S3",
    )
    args = parser.parse_args()

    pasta_id = os.getenv("DRIVER_FOLDER_ID") or os.getenv("DRIVE_FOLDER_ID")
    if not pasta_id:
        raise RuntimeError("DRIVER_FOLDER_ID não configurado para a migração")

    drive = configurar_drive()
    storage = VultrS3Storage.from_env()
    encontrados = listar_gabaritos_drive(drive, pasta_id)

    for ano_escolar in ANOS_ESCOLARES:
        arquivo = encontrados.get(ano_escolar)
        if not arquivo:
            print(f"FALTANDO: {nome_gabarito(ano_escolar)}")
            continue

        key = posixpath.join(
            storage.config.prefixo_gabaritos,
            arquivo["name"],
        )
        print(f"ENCONTRADO: {arquivo['name']} -> s3://{storage.config.bucket}/{key}")

        if args.dry_run:
            continue

        if not args.sobrescrever:
            try:
                storage.client.head_object(Bucket=storage.config.bucket, Key=key)
                print("  IGNORADO: objeto já existe")
                continue
            except ClientError as error:
                codigo = str(error.response.get("Error", {}).get("Code", ""))
                if codigo not in {"404", "NoSuchKey", "NotFound"}:
                    raise

        conteudo = baixar_drive(drive, arquivo["id"])
        storage.client.put_object(
            Bucket=storage.config.bucket,
            Key=key,
            Body=conteudo,
            ContentType=arquivo.get("mimeType") or "application/octet-stream",
        )
        print("  MIGRADO")

    faltantes = set(ANOS_ESCOLARES) - set(encontrados)
    if faltantes:
        raise RuntimeError(
            "Migração incompleta; faltam: "
            + ", ".join(nome_gabarito(ano) for ano in sorted(faltantes))
        )


if __name__ == "__main__":
    main()
