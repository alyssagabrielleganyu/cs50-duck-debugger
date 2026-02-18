from __future__ import annotations
import json
import os
import re
import time
from typing import Any, Dict, List

FERPA_MODE = os.getenv("FERPA_MODE", "true").lower() == "true"
FERPA_REDACT_PII = os.getenv("FERPA_REDACT_PII", "true").lower() == "true"
FERPA_LOG_CONTENT = os.getenv("FERPA_LOG_CONTENT", "false").lower() == "true"

# Basic PII patterns (demo-grade; add more as needed)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
STUDENT_ID_RE = re.compile(r"\b\d{7,12}\b")  # common student ID lengths

def redact_pii(text: str) -> str:
    """Redact likely PII in free-form text."""
    if not FERPA_REDACT_PII:
        return text
    t = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    t = PHONE_RE.sub("[REDACTED_PHONE]", t)
    t = STUDENT_ID_RE.sub("[REDACTED_STUDENT_ID]", t)
    return t

def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Return messages with PII redacted (content only)."""
    sanitized: List[Dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))
        sanitized.append({"role": role, "content": redact_pii(content)})
    return sanitized

def audit_event(event: str, meta: Dict[str, Any]) -> None:
    """
    Demo audit logger: write metadata-only events to audit.log (no prompt content).
    """
    safe_meta = dict(meta)
    safe_meta["ts"] = int(time.time())

    line = json.dumps({"event": event, "meta": safe_meta}, ensure_ascii=False)

    # Write to file in the project directory (same folder as server.py)
    with open("audit.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

    # Also print to stdout so you can see it in uvicorn logs
    print(line)

