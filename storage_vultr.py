"""Camada de acesso ao Vultr Object Storage compatível com S3."""

from __future__ import annotations

import mimetypes
import os
import posixpath
from dataclasses import dataclass
from datetime import timezone
from typing import Dict, List

import boto3
from botocore.config import Config


def _normalizar_prefixo(prefixo: str) -> str:
    normalizado = str(prefixo or "").strip().strip("/")
    return f"{normalizado}/" if normalizado else ""


def _normalizar_endpoint(host: str) -> str:
    host = host.strip()
    if host.startswith(("http://", "https://")):
        return host
    return f"https://{host}"


@dataclass
class VultrS3Config:
    access_key_id: str
    secret_access_key: str
    endpoint_url: str
    bucket: str
    region: str
    prefixo_upload: str
    prefixo_gabaritos: str
    prefixo_processados: str

    @classmethod
    def from_env(cls) -> "VultrS3Config":
        access_key_id = os.getenv("VULTR_S3_ACCESS_KEY_ID", "").strip()
        secret_access_key = os.getenv("VULTR_S3_SECRET_ACCESS_KEY", "").strip()
        host = os.getenv("VULTR_S3_HOST", "").strip()
        bucket = os.getenv("VULTR_S3_BUCKET", "").strip()

        faltantes = [
            nome
            for nome, valor in (
                ("VULTR_S3_ACCESS_KEY_ID", access_key_id),
                ("VULTR_S3_SECRET_ACCESS_KEY", secret_access_key),
                ("VULTR_S3_HOST", host),
                ("VULTR_S3_BUCKET", bucket),
            )
            if not valor
        ]
        if faltantes:
            raise RuntimeError(
                "Configuração do Vultr S3 incompleta: " + ", ".join(faltantes)
            )

        endpoint_url = _normalizar_endpoint(host)
        hostname = endpoint_url.split("://", 1)[-1].split("/", 1)[0]
        region = os.getenv("VULTR_S3_REGION", "").strip() or hostname.split(".", 1)[0]

        return cls(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            endpoint_url=endpoint_url,
            bucket=bucket,
            region=region or "us-east-1",
            prefixo_upload=_normalizar_prefixo(
                os.getenv("VULTR_S3_PREFIX_UPLOAD", "entrada")
            ),
            prefixo_gabaritos=_normalizar_prefixo(
                os.getenv("VULTR_S3_PREFIX_GABARITOS", "gabaritos")
            ),
            prefixo_processados=_normalizar_prefixo(
                os.getenv("VULTR_S3_PREFIX_PROCESSADOS", "processados")
            ),
        )


class VultrS3Storage:
    source_name = "vultr_s3"

    def __init__(self, config: VultrS3Config):
        self.config = config
        addressing_style = (
            "path"
            if os.getenv("VULTR_S3_FORCE_PATH_STYLE", "false").strip().lower()
            in {"1", "true", "yes", "on"}
            else "virtual"
        )
        self.client = boto3.client(
            "s3",
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            endpoint_url=config.endpoint_url,
            region_name=config.region,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": addressing_style},
            ),
        )

    @classmethod
    def from_env(cls) -> "VultrS3Storage":
        return cls(VultrS3Config.from_env())

    def listar_uploads(self) -> List[Dict]:
        return self._listar_objetos(self.config.prefixo_upload)

    def listar_gabaritos(self) -> List[Dict]:
        candidatos: Dict[str, Dict] = {}
        prefixos = [self.config.prefixo_gabaritos, self.config.prefixo_upload]

        for prefixo in dict.fromkeys(prefixos):
            for arquivo in self._listar_objetos(prefixo):
                if posixpath.basename(arquivo["key"]).lower().startswith("gabarito"):
                    candidatos[arquivo["key"]] = arquivo

        if not candidatos:
            for arquivo in self._listar_objetos(""):
                if posixpath.basename(arquivo["key"]).lower().startswith("gabarito"):
                    candidatos[arquivo["key"]] = arquivo

        return list(candidatos.values())

    def baixar(self, key: str, caminho_destino: str) -> str:
        os.makedirs(os.path.dirname(caminho_destino) or ".", exist_ok=True)
        self.client.download_file(self.config.bucket, key, caminho_destino)
        return caminho_destino

    def mover_para_processados(self, key: str, ano_escolar: str) -> str:
        relativo = key
        if self.config.prefixo_upload and key.startswith(self.config.prefixo_upload):
            relativo = key[len(self.config.prefixo_upload) :]

        destino = posixpath.join(
            self.config.prefixo_processados,
            ano_escolar,
            relativo.lstrip("/"),
        )
        self.client.copy_object(
            Bucket=self.config.bucket,
            CopySource={"Bucket": self.config.bucket, "Key": key},
            Key=destino,
            ContentType=mimetypes.guess_type(destino)[0] or "application/octet-stream",
            MetadataDirective="REPLACE",
        )
        self.client.delete_object(Bucket=self.config.bucket, Key=key)
        return destino

    def _listar_objetos(self, prefixo: str) -> List[Dict]:
        objetos: List[Dict] = []
        parametros = {
            "Bucket": self.config.bucket,
            "Prefix": prefixo,
        }

        while True:
            resposta = self.client.list_objects_v2(**parametros)
            for item in resposta.get("Contents", []):
                key = item.get("Key", "")
                if not key or key.endswith("/"):
                    continue

                modificado = item.get("LastModified")
                if modificado and modificado.tzinfo is None:
                    modificado = modificado.replace(tzinfo=timezone.utc)

                objetos.append(
                    {
                        "id": f"s3:{key}",
                        "storage_id": key,
                        "source": self.source_name,
                        "key": key,
                        "name": posixpath.basename(key),
                        "mimeType": mimetypes.guess_type(key)[0]
                        or "application/octet-stream",
                        "modifiedTime": modificado.isoformat() if modificado else "",
                        "size": int(item.get("Size", 0)),
                        "etag": str(item.get("ETag", "")).strip('"'),
                    }
                )

            if not resposta.get("IsTruncated"):
                break

            parametros["ContinuationToken"] = resposta.get("NextContinuationToken")

        return objetos

    def destino_label(self, ano_escolar: str) -> str:
        return posixpath.join(
            self.config.prefixo_processados,
            ano_escolar,
        )
