"""Thread-safe conversation memory manager for multi-turn chat."""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional


class ChatTurn:
    def __init__(self, role: str, content: str, timestamp: float | None = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()


class ChatMemoryManager:
    """Stores conversation turns per session with automatic trimming and TTL expiry."""

    def __init__(self, max_turns: int = 6, ttl_seconds: int = 7200):
        self.max_turns = max_turns  # ~6 turns (3 user + 3 assistant)
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, List[ChatTurn]] = {}
        self._last_accessed: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _cleanup_expired(self):
        now = time.time()
        expired = [
            sid for sid, ts in self._last_accessed.items()
            if now - ts > self.ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_accessed.pop(sid, None)

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[dict]:
        with self._lock:
            self._cleanup_expired()
            self._last_accessed[session_id] = time.time()
            turns = self._sessions.get(session_id, [])
            limit = limit or self.max_turns
            recent = turns[-limit:]
            return [{"role": t.role, "content": t.content} for t in recent]

    def add_turn(self, session_id: str, user_message: str, assistant_message: str):
        with self._lock:
            self._cleanup_expired()
            self._last_accessed[session_id] = time.time()
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            
            self._sessions[session_id].append(ChatTurn("user", user_message))
            self._sessions[session_id].append(ChatTurn("assistant", assistant_message))
            
            # Keep only the most recent max_turns
            if len(self._sessions[session_id]) > self.max_turns:
                self._sessions[session_id] = self._sessions[session_id][-self.max_turns:]

    def format_history_for_prompt(self, session_id: str, limit: Optional[int] = None) -> str:
        history = self.get_history(session_id, limit=limit)
        if not history:
            return "No previous conversation."
        
        lines = []
        for turn in history:
            speaker = "Patient" if turn["role"] == "user" else "Doctor / Assistant"
            lines.append(f"{speaker}: {turn['content']}")
        return "\n".join(lines)

    def clear_history(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)
            self._last_accessed.pop(session_id, None)


# Global singleton instance
memory_manager = ChatMemoryManager(max_turns=6, ttl_seconds=7200)
