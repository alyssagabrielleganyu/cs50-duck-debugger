from __future__ import annotations

import os
import time
import uuid
from typing import Dict, List, Literal

import jwt
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

DEV_JWT_SECRET = os.getenv("DEV_JWT_SECRET", "dev-local-secret-change-me")
DEV_JWT_ALGORITHM = "HS256"
DEV_TOKEN_TTL_SECONDS = 60 * 60 * 12


def issue_dev_token(email: str, role: Literal["student", "teacher"], groups: List[str]) -> str:
    now = int(time.time())
    payload = {
        "sub": str(uuid.uuid4()),
        "email": email,
        "role": role,
        "groups": groups,
        "iat": now,
        "exp": now + DEV_TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, DEV_JWT_SECRET, algorithm=DEV_JWT_ALGORITHM)


def verify_token(authorization_header: str | None) -> Dict[str, object]:
    if not authorization_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")

    parts = authorization_header.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format.")

    token = parts[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    try:
        claims = jwt.decode(token, DEV_JWT_SECRET, algorithms=[DEV_JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

    email = claims.get("email")
    role = claims.get("role")
    groups = claims.get("groups", [])
    sub = claims.get("sub")

    if not isinstance(email, str) or not email:
        raise HTTPException(status_code=401, detail="Token missing email claim.")
    if role not in {"student", "teacher"}:
        raise HTTPException(status_code=401, detail="Token missing valid role claim.")
    if not isinstance(groups, list):
        raise HTTPException(status_code=401, detail="Token groups claim must be a list.")
    if not isinstance(sub, str) or not sub:
        raise HTTPException(status_code=401, detail="Token missing sub claim.")

    return {
        "email": email,
        "role": role,
        "groups": [str(group) for group in groups],
        "sub": sub,
    }
