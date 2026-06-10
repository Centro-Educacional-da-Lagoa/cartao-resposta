import re
import unicodedata
from typing import Optional


ANOS_ESCOLARES = ("4ano", "5ano", "8ano", "9ano")

QUESTOES_POR_ANO = {
    "4ano": 44,
    "5ano": 44,
    "8ano": 52,
    "9ano": 52,
}

NUMERO_POR_ANO = {
    "4ano": 4,
    "5ano": 5,
    "8ano": 8,
    "9ano": 9,
}


def _normalizar_texto(valor: object) -> str:
    texto = str(valor or "").strip().lower()
    texto = texto.replace("º", " ").replace("°", " ").replace("ª", " ")
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(char for char in texto if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", texto)


def detectar_ano_escolar(valor: object) -> Optional[str]:
    texto_original = str(valor or "").strip().lower()
    match_grau = re.search(r"(?<!\d)(4|5|8|9)\s*[º°ª]", texto_original)
    if match_grau:
        return f"{match_grau.group(1)}ano"

    texto = _normalizar_texto(valor)
    if not texto or texto in {"n/a", "na", "none", "null", "-"}:
        return None

    match = re.search(r"(?<!\d)(4|5|8|9)\s*ano(?![a-z])", texto)
    if match:
        return f"{match.group(1)}ano"

    palavras = (
        ("4ano", r"\bquart[oa]?\s+ano\b"),
        ("5ano", r"\bquint[oa]?\s+ano\b"),
        ("8ano", r"\boitav[oa]?\s+ano\b"),
        ("9ano", r"\bnon[oa]?\s+ano\b"),
    )
    for ano_escolar, padrao in palavras:
        if re.search(padrao, texto):
            return ano_escolar

    return None


def detectar_ano_por_turma(valor: object) -> Optional[str]:
    ano_escolar = detectar_ano_escolar(valor)
    if ano_escolar:
        return ano_escolar

    texto = str(valor or "").strip()
    match_turma = re.search(r"(?:^|\b)(4|5|8|9)\s*[A-Za-z]\b", texto)
    if match_turma:
        return f"{match_turma.group(1)}ano"

    return None


def numero_questoes_por_ano(ano_escolar: object) -> Optional[int]:
    ano_normalizado = detectar_ano_escolar(ano_escolar)
    if not ano_normalizado:
        return None
    return QUESTOES_POR_ANO[ano_normalizado]


def rotulo_ano(ano_escolar: object) -> str:
    ano_normalizado = detectar_ano_escolar(ano_escolar)
    if not ano_normalizado:
        return "ano não identificado"
    return f"{NUMERO_POR_ANO[ano_normalizado]}º ano"


def nome_gabarito(ano_escolar: object) -> Optional[str]:
    ano_normalizado = detectar_ano_escolar(ano_escolar)
    if not ano_normalizado:
        return None
    return f"gabarito_{ano_normalizado}"
