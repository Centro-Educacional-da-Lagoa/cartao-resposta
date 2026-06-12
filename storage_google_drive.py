"""Camada de entrada e arquivamento de cartões no Google Drive."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from anos_escolares import ANOS_ESCOLARES, NUMERO_POR_ANO


@dataclass(frozen=True)
class GoogleDriveConfig:
    pasta_upload_id: str
    pastas_processados: Dict[str, str]

    @classmethod
    def from_env(cls) -> "GoogleDriveConfig":
        pasta_upload_id = (
            os.getenv("DRIVER_FOLDER_ID") or os.getenv("DRIVE_FOLDER_ID") or ""
        ).strip()
        if not pasta_upload_id:
            raise RuntimeError("DRIVER_FOLDER_ID não configurado")

        pastas_processados = {}
        for ano_escolar in ANOS_ESCOLARES:
            numero = NUMERO_POR_ANO[ano_escolar]
            pasta_id = (
                os.getenv(f"DRIVER_FOLDER_{numero}ANO")
                or os.getenv(f"DRIVE_FOLDER_{numero}ANO")
                or ""
            ).strip()
            if not pasta_id:
                raise RuntimeError(
                    f"DRIVER_FOLDER_{numero}ANO não configurado"
                )
            pastas_processados[ano_escolar] = pasta_id

        return cls(
            pasta_upload_id=pasta_upload_id,
            pastas_processados=pastas_processados,
        )


class GoogleDriveStorage:
    source_name = "google_drive"

    def __init__(self, config: GoogleDriveConfig, service):
        self.config = config
        self.service = service

    @classmethod
    def from_env(cls) -> "GoogleDriveStorage":
        config = GoogleDriveConfig.from_env()
        credentials_json = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()

        if credentials_json:
            credentials = Credentials.from_service_account_info(
                json.loads(credentials_json),
                scopes=["https://www.googleapis.com/auth/drive"],
            )
        else:
            credentials = Credentials.from_service_account_file(
                "credenciais_google.json",
                scopes=["https://www.googleapis.com/auth/drive"],
            )

        service = build(
            "drive",
            "v3",
            credentials=credentials,
            cache_discovery=False,
        )
        return cls(config, service)

    def listar_uploads(self) -> List[Dict]:
        arquivos = []
        page_token = None

        while True:
            resposta = self.service.files().list(
                q=f"'{self.config.pasta_upload_id}' in parents and trashed = false",
                fields=(
                    "nextPageToken, "
                    "files(id, name, mimeType, modifiedTime, size)"
                ),
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()

            for item in resposta.get("files", []):
                if item.get("mimeType") == "application/vnd.google-apps.folder":
                    continue
                arquivos.append(self._normalizar_arquivo(item))

            page_token = resposta.get("nextPageToken")
            if not page_token:
                break

        return arquivos

    def _normalizar_arquivo(self, item: Dict) -> Dict:
        arquivo_id = item.get("id", "")
        return {
            "id": f"drive:{arquivo_id}",
            "storage_id": arquivo_id,
            "source": self.source_name,
            "name": item.get("name", ""),
            "mimeType": item.get("mimeType", ""),
            "modifiedTime": item.get("modifiedTime", ""),
            "size": int(item.get("size", 0) or 0),
        }

    def baixar(self, arquivo_id: str, caminho_destino: str) -> str:
        request = self.service.files().get_media(fileId=arquivo_id)
        with open(caminho_destino, "wb") as arquivo_local:
            downloader = MediaIoBaseDownload(arquivo_local, request)
            concluido = False
            while not concluido:
                _, concluido = downloader.next_chunk()
        return caminho_destino

    def mover_para_processados(self, arquivo_id: str, ano_escolar: str) -> str:
        pasta_destino = self.config.pastas_processados[ano_escolar]
        metadata = self.service.files().get(
            fileId=arquivo_id,
            fields="parents",
            supportsAllDrives=True,
        ).execute()
        pais_atuais = ",".join(metadata.get("parents", []))

        self.service.files().update(
            fileId=arquivo_id,
            addParents=pasta_destino,
            removeParents=pais_atuais,
            fields="id, parents",
            supportsAllDrives=True,
        ).execute()
        return pasta_destino

    def destino_label(self, ano_escolar: str) -> str:
        return f"Google Drive/{ano_escolar}"
