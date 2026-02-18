from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path("identity.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                role TEXT NOT NULL DEFAULT 'student'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memberships (
                user_id INTEGER NOT NULL,
                group_id INTEGER NOT NULL,
                PRIMARY KEY (user_id, group_id),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE
            )
            """
        )


def _serialize_user(row: sqlite3.Row) -> Dict[str, object]:
    return {
        "id": str(row["id"]),
        "userName": row["email"],
        "active": bool(row["active"]),
        "role": row["role"],
    }


def _group_members(conn: sqlite3.Connection, group_id: int) -> List[Dict[str, str]]:
    rows = conn.execute(
        """
        SELECT u.id, u.email
        FROM memberships m
        JOIN users u ON u.id = m.user_id
        WHERE m.group_id = ?
        ORDER BY u.id
        """,
        (group_id,),
    ).fetchall()
    return [{"value": str(row["id"]), "display": row["email"]} for row in rows]


def _serialize_group(conn: sqlite3.Connection, row: sqlite3.Row) -> Dict[str, object]:
    group_id = int(row["id"])
    return {
        "id": str(group_id),
        "displayName": row["name"],
        "members": _group_members(conn, group_id),
    }


def create_user(email: str, active: bool = True, role: str = "student") -> Dict[str, object]:
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO users(email, active, role) VALUES (?, ?, ?)",
            (email, 1 if active else 0, role),
        )
        user_id = int(cursor.lastrowid)
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            raise ValueError("Failed to create user.")
        return _serialize_user(row)


def get_user(user_id: int) -> Optional[Dict[str, object]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        return _serialize_user(row)


def get_user_by_email(email: str) -> Optional[Dict[str, object]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            return None
        return _serialize_user(row)


def set_user_active(user_id: int, active: bool) -> Optional[Dict[str, object]]:
    with _connect() as conn:
        conn.execute("UPDATE users SET active = ? WHERE id = ?", (1 if active else 0, user_id))
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        return _serialize_user(row)


def create_group(name: str) -> Dict[str, object]:
    with _connect() as conn:
        cursor = conn.execute("INSERT INTO groups(name) VALUES (?)", (name,))
        group_id = int(cursor.lastrowid)
        row = conn.execute("SELECT * FROM groups WHERE id = ?", (group_id,)).fetchone()
        if row is None:
            raise ValueError("Failed to create group.")
        return _serialize_group(conn, row)


def get_group(group_id: int) -> Optional[Dict[str, object]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM groups WHERE id = ?", (group_id,)).fetchone()
        if row is None:
            return None
        return _serialize_group(conn, row)


def add_group_members(group_id: int, user_ids: List[int]) -> Optional[Dict[str, object]]:
    with _connect() as conn:
        group_row = conn.execute("SELECT * FROM groups WHERE id = ?", (group_id,)).fetchone()
        if group_row is None:
            return None

        for user_id in user_ids:
            user_row = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
            if user_row is None:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO memberships(user_id, group_id) VALUES (?, ?)",
                (user_id, group_id),
            )

        return _serialize_group(conn, group_row)


def remove_group_members(group_id: int, user_ids: List[int]) -> Optional[Dict[str, object]]:
    with _connect() as conn:
        group_row = conn.execute("SELECT * FROM groups WHERE id = ?", (group_id,)).fetchone()
        if group_row is None:
            return None

        for user_id in user_ids:
            conn.execute(
                "DELETE FROM memberships WHERE user_id = ? AND group_id = ?",
                (user_id, group_id),
            )

        return _serialize_group(conn, group_row)
