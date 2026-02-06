import json
import os
import sqlite3
import uuid
from datetime import datetime

DB_PATH = "db/skillsync_local.db"


def init_sql_db():
    """Létrehozza a helyi SQL adatbázist a perzisztenciához és audithoz."""
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 30. pont: Chat szekciók tárolása
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 34. pont: Audit Log & Üzenetek tárolása
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT NOT NULL, -- 'user' vagy 'assistant'
            content TEXT NOT NULL,
            sources TEXT, -- JSON-ként tárolt forráslista (fájlnevek, oldalak)
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
        )
    """)
    conn.commit()
    conn.close()


def save_message_to_db(session_id, tenant_id, role, content, sources=None):
    """Elmenti az üzenetet és szükésg esetén létrehozza a sessiont (Audit Trail)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ha új session, regisztráljuk
    cursor.execute(
        "INSERT OR IGNORE INTO chat_sessions (session_id, tenant_id, title) VALUES (?, ?, ?)",
        (session_id, tenant_id, content[:50] + "..."),
    )

    # Üzenet beszúrása
    cursor.execute(
        """
        INSERT INTO messages (session_id, role, content, sources)
        VALUES (?, ?, ?, ?)
    """,
        (session_id, role, content, json.dumps(sources) if sources else None),
    )

    conn.commit()
    conn.close()


def get_chat_history(session_id, limit=20):
    """Visszatölti a korábbi üzeneteket a kontextushoz."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, content FROM messages 
        WHERE session_id = ? 
        ORDER BY timestamp ASC LIMIT ?
    """,
        (session_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]
