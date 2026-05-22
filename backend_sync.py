"""Integração do worker Python com o backend NestJS."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


class BackendSyncClient:
    def __init__(
        self,
        base_url: str,
        email: str = "",
        password: str = "",
        timeout_seconds: int = 20,
        auth_enabled: bool = False,
        auth_path: str = "/api/auth/login",
        resultado_path: str = "/api/aluno",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self.timeout_seconds = timeout_seconds
        self.auth_enabled = auth_enabled
        self.auth_path = auth_path
        self.resultado_path = resultado_path
        self.session = requests.Session()
        self._is_authenticated = False
        self._access_token = ""

    @property
    def auth_login_url(self) -> str:
        return self._url(self.auth_path)

    @property
    def create_leitura_url(self) -> str:
        return self._url(self.resultado_path)

    def _url(self, path_or_url: str) -> str:
        if path_or_url.startswith(("http://", "https://")):
            return path_or_url

        return f"{self.base_url}/{path_or_url.lstrip('/')}"

    def _login(self) -> None:
        if not self.email or not self.password:
            raise RuntimeError("Autenticação do backend habilitada, mas email/senha não foram configurados")

        response = self.session.post(
            self.auth_login_url,
            json={"email": self.email, "password": self.password},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        self._store_access_token(response)
        self._is_authenticated = True

    def send_leitura(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.auth_enabled and not self._is_authenticated:
            self._login()

        response = self.session.post(
            self.create_leitura_url,
            json=payload,
            timeout=self.timeout_seconds,
        )

        # token expirou/cookie inválido: reautentica uma vez
        if self.auth_enabled and response.status_code == 401:
            self._login()
            response = self.session.post(
                self.create_leitura_url,
                json=payload,
                timeout=self.timeout_seconds,
            )

        self._raise_with_context(response)
        data = response.json()
        if not isinstance(data, dict):
            return {"ok": True}
        return data

    def _raise_with_context(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detalhes = response.text.strip()
            if len(detalhes) > 500:
                detalhes = f"{detalhes[:500]}..."
            mensagem = (
                f"HTTP {response.status_code} ao enviar leitura para backend "
                f"({response.request.method} {response.url})"
            )
            if detalhes:
                mensagem = f"{mensagem} - resposta: {detalhes}"
            raise RuntimeError(mensagem) from exc

    def _store_access_token(self, response: requests.Response) -> None:
        try:
            data = response.json()
        except ValueError:
            return

        if not isinstance(data, dict):
            return

        access_token = data.get("accessToken") or data.get("access_token") or data.get("token")
        if not isinstance(access_token, str) or not access_token:
            return

        self._access_token = access_token
        self.session.headers.update({"Authorization": f"Bearer {access_token}"})


def create_backend_sync_client_from_env() -> Optional[BackendSyncClient]:
    enabled = os.getenv("BACKEND_SYNC_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if not enabled:
        return None

    base_url = os.getenv("BACKEND_BASE_URL", "").strip()
    email = os.getenv("BACKEND_USER_EMAIL", "").strip()
    password = os.getenv("BACKEND_USER_PASSWORD", "").strip()
    auth_enabled = os.getenv("BACKEND_AUTH_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    auth_path = os.getenv("BACKEND_AUTH_PATH", "/api/auth/login").strip()
    resultado_path = os.getenv("BACKEND_RESULTADO_PATH", "/api/aluno").strip()

    if not base_url:
        print("⚠️ Backend sync desativado: configure BACKEND_BASE_URL")
        return None

    if auth_enabled and (not email or not password):
        print("⚠️ Backend sync desativado: BACKEND_AUTH_ENABLED exige BACKEND_USER_EMAIL e BACKEND_USER_PASSWORD")
        return None

    timeout = int(os.getenv("BACKEND_SYNC_TIMEOUT_SECONDS", "20"))
    return BackendSyncClient(
        base_url=base_url,
        email=email,
        password=password,
        timeout_seconds=timeout,
        auth_enabled=auth_enabled,
        auth_path=auth_path,
        resultado_path=resultado_path,
    )
