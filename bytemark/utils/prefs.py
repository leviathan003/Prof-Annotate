"""
bytemark/utils/prefs.py
Tiny JSON preferences store at PREFS_FILE — used for first-run flags
and other small bits of app state that aren't worth a config module.
"""

from __future__ import annotations

import json
from typing import Any

from bytemark.config.constants import APP_CACHE_DIR, PREFS_FILE


def load_prefs() -> dict[str, Any]:
    if not PREFS_FILE.exists():
        return {}
    try:
        return json.loads(PREFS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_prefs(data: dict[str, Any]) -> None:
    APP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        PREFS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def get_pref(key: str, default: Any = None) -> Any:
    return load_prefs().get(key, default)


def set_pref(key: str, value: Any) -> None:
    data = load_prefs()
    data[key] = value
    save_prefs(data)
