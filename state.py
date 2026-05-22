import copy
import json
import os
import tempfile
from datetime import datetime
from typing import Optional


STATE_FILE = os.getenv(
    "BOT_STATE_FILE",
    os.path.join(os.path.dirname(__file__), "bot_state_runtime.json")
)


def _default_state() -> dict:
    return {
        "status": "idle",        # idle | running | error
        "current_file": None,     # arquivo sendo processado agora
        "progress": 0,            # 0 a 100
        "total_corrected": 0,     # total desde que o bot subiu
        "last_correction": None,  # timestamp da ultima correcao
        "logs": []                # lista dos ultimos 50 eventos
    }


def _normalize_state(data: dict) -> dict:
    state = _default_state()
    if isinstance(data, dict):
        for key in state:
            if key in data:
                state[key] = data[key]
    if not isinstance(state.get("logs"), list):
        state["logs"] = []
    return state


def _read_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return _default_state()

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return _normalize_state(json.load(f))
    except Exception:
        return _default_state()


def _write_state(state: dict) -> None:
    dir_name = os.path.dirname(STATE_FILE) or "."
    os.makedirs(dir_name, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(prefix="bot_state_", suffix=".json", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, STATE_FILE)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _trim_logs(state: dict) -> None:
    if len(state["logs"]) > 50:
        state["logs"] = state["logs"][:50]


def _normalize_progress(progress: Optional[int]) -> Optional[int]:
    if progress is None:
        return None

    try:
        return max(0, min(100, int(progress)))
    except (TypeError, ValueError):
        return 0


bot_state = _read_state()
_write_state(bot_state)


def get_state_snapshot() -> dict:
    global bot_state
    bot_state = _read_state()
    return copy.deepcopy(bot_state)


def reset_session():
    global bot_state
    state = _default_state()
    state["logs"].insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "msg": "Sessao do bot iniciada"
    })
    bot_state = state
    _write_state(state)


def update_status(status: str, file: str = None, progress: Optional[int] = 0):
    global bot_state
    state = _read_state()
    state["status"] = status
    state["current_file"] = file
    progress_normalized = _normalize_progress(progress)
    if progress_normalized is not None:
        state["progress"] = progress_normalized
    bot_state = state
    _write_state(state)


def record_correction(filename: str, file: str = None, progress: Optional[int] = None):
    global bot_state
    state = _read_state()
    state["status"] = "running"
    state["current_file"] = file if file is not None else filename
    progress_normalized = _normalize_progress(progress)
    if progress_normalized is not None:
        state["progress"] = progress_normalized
    state["total_corrected"] = int(state.get("total_corrected") or 0) + 1
    state["last_correction"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    state["logs"].insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "msg": f"Correcao concluida: {filename}"
    })
    _trim_logs(state)
    bot_state = state
    _write_state(state)


def finish_correction(filename: str):
    record_correction(filename=filename, file=None, progress=100)
    update_status("idle", None, 100)


def log(msg: str):
    global bot_state
    state = _read_state()
    state["logs"].insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "msg": msg
    })
    _trim_logs(state)
    bot_state = state
    _write_state(state)
