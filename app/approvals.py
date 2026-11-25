"""审批单持久化：不依赖内存状态，重启后可按 thread_id 对账。"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ApprovalStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _write(self, records: list[dict[str, Any]]) -> None:
        self.path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
            encoding="utf-8",
        )

    def create(self, approval_id: str, thread_id: str, tool: str, args: dict[str, Any], reason: str) -> dict[str, Any]:
        with self._lock:
            existing = self.get(approval_id)
            if existing:
                return existing
            record = {
                "id": approval_id,
                "thread_id": thread_id,
                "tool": tool,
                "args": args,
                "reason": reason,
                "status": "pending",
                "approver": "",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "decided_at": None,
            }
            records = self._read()
            records.append(record)
            self._write(records)
            return record

    def get(self, approval_id: str) -> dict[str, Any] | None:
        for record in self._read():
            if record.get("id") == approval_id:
                return record
        return None

    def decide(self, approval_id: str, approved: bool, approver: str = "admin") -> dict[str, Any] | None:
        with self._lock:
            records = self._read()
            for record in records:
                if record.get("id") == approval_id:
                    record.update(
                        {
                            "status": "approved" if approved else "rejected",
                            "approver": approver or "admin",
                            "decided_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    self._write(records)
                    return record
            return None

    def list_pending(self, thread_id: str | None = None) -> list[dict[str, Any]]:
        return [
            record
            for record in self._read()
            if record.get("status") == "pending"
            and (thread_id is None or record.get("thread_id") == thread_id)
        ]

    def list_all(self) -> list[dict[str, Any]]:
        return self._read()
