"""Integração do worker Python com o backend NestJS."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


class BackendSyncClient:
    def __init__(
        self,
        base_url: str,
        email: str,
        password: str,
        timeout_seconds: int = 20,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self._is_authenticated = False

    @property
    def auth_login_url(self) -> str:
        return f"{self.base_url}/api/v1/auth/login"

    @property
    def create_leitura_url(self) -> str:
        return f"{self.base_url}/api/v1/cartao-resposta/leituras"

    def _login(self) -> None:
        response = self.session.post(
            self.auth_login_url,
            json={"email": self.email, "password": self.password},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        self._is_authenticated = True

    def send_leitura(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_authenticated:
            self._login()

        response = self.session.post(
            self.create_leitura_url,
            json=payload,
            timeout=self.timeout_seconds,
        )

        # token expirou/cookie inválido: reautentica uma vez
        if response.status_code == 401:
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

    if not base_url or not email or not password:
        print(
            "⚠️ Backend sync desativado: configure BACKEND_BASE_URL, BACKEND_USER_EMAIL e BACKEND_USER_PASSWORD"
        )
        return None

    timeout = int(os.getenv("BACKEND_SYNC_TIMEOUT_SECONDS", "20"))
    return BackendSyncClient(
        base_url=base_url,
        email=email,
        password=password,
        timeout_seconds=timeout,
    )
