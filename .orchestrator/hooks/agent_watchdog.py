#!/usr/bin/env python3
"""
Watchdog that monitors an agent's state file and halts the agent on:
  - successful completion (status == "done")
  - repeated failure (failure_count >= MAX_FAILURES)
  - stalled heartbeat (no heartbeat update in STALL_SECONDS)

Reads:    .orchestrator/state/<agent>.json
Writes:   .orchestrator/incidents/<timestamp>-<agent>.json (on escalation)
Side-effect: signals the agent's tmux pane to stop.

State file shape:
    {
      "agent": "C",
      "status": "running" | "done" | "failed",
      "current_task": "PR-3",
      "failure_count": 0,
      "heartbeat_at": "2026-05-27T10:00:00Z",
      "started_at": "2026-05-27T09:30:00Z",
      "reason": "<optional, e.g. 'kill-criterion'>"
    }

Usage:
    python agent_watchdog.py --agent C [--poll 5] [--stall 600] [--max-failures 3]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = REPO_ROOT / ".orchestrator" / "state"
INCIDENT_DIR = REPO_ROOT / ".orchestrator" / "incidents"
TMUX_SESSION = "claude-dev"


@dataclass
class WatchdogConfig:
    agent: str
    poll_seconds: int = 5
    stall_seconds: int = 600
    max_failures: int = 3

    @property
    def state_file(self) -> Path:
        return STATE_DIR / f"agent-{self.agent}.json"

    @property
    def tmux_window(self) -> str:
        suffix = {"A": "frontend", "B": "backend", "C": "ml", "D": "qa"}[self.agent]
        return f"{TMUX_SESSION}:agent-{self.agent}-{suffix}"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def parse_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def write_incident(cfg: WatchdogConfig, kind: str, state: dict | None) -> Path:
    INCIDENT_DIR.mkdir(parents=True, exist_ok=True)
    ts = now_iso().replace(":", "").replace("-", "")
    path = INCIDENT_DIR / f"{ts}-agent-{cfg.agent}-{kind}.json"
    path.write_text(
        json.dumps(
            {
                "kind": kind,
                "agent": cfg.agent,
                "at": now_iso(),
                "state_snapshot": state,
            },
            indent=2,
        )
    )
    return path


def signal_agent_stop(cfg: WatchdogConfig, reason: str) -> None:
    """Send a soft stop signal into the agent's tmux pane."""
    msg = f"echo '[watchdog] halting agent {cfg.agent}: {reason}'"
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", cfg.tmux_window, msg, "C-m"],
            check=False,
            timeout=5,
        )
        subprocess.run(
            ["tmux", "send-keys", "-t", cfg.tmux_window, "C-c"],
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def page_orchestrator(cfg: WatchdogConfig, incident_path: Path) -> None:
    msg = f"echo '[watchdog] AGENT {cfg.agent} ESCALATION: see {incident_path}'"
    subprocess.run(
        ["tmux", "send-keys", "-t", f"{TMUX_SESSION}:orchestrator", msg, "C-m"],
        check=False,
        timeout=5,
    )


def evaluate(cfg: WatchdogConfig, state: dict | None) -> tuple[str, str] | None:
    """Return (kind, reason) if action needed, else None."""
    if state is None:
        return None

    status = state.get("status")
    if status == "done":
        return ("completion", "task complete")

    if state.get("failure_count", 0) >= cfg.max_failures:
        return ("repeated-failure", f"failure_count >= {cfg.max_failures}")

    if state.get("reason") == "kill-criterion":
        return ("kill-criterion", "agent reported kill-criterion miss")

    hb = state.get("heartbeat_at")
    if hb:
        age = (datetime.now(timezone.utc) - parse_iso(hb)).total_seconds()
        if age > cfg.stall_seconds:
            return ("stall", f"no heartbeat for {int(age)}s")

    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--poll", type=int, default=5)
    ap.add_argument("--stall", type=int, default=600)
    ap.add_argument("--max-failures", type=int, default=3)
    ap.add_argument("--once", action="store_true", help="evaluate once and exit")
    args = ap.parse_args()

    cfg = WatchdogConfig(
        agent=args.agent,
        poll_seconds=args.poll,
        stall_seconds=args.stall,
        max_failures=args.max_failures,
    )

    print(f"[watchdog] watching agent-{cfg.agent} at {cfg.state_file}")

    while True:
        state = read_state(cfg.state_file)
        action = evaluate(cfg, state)
        if action is not None:
            kind, reason = action
            incident = write_incident(cfg, kind, state)
            print(f"[watchdog] {kind}: {reason} -> {incident}")
            if kind == "completion":
                signal_agent_stop(cfg, "task done")
            else:
                signal_agent_stop(cfg, reason)
                page_orchestrator(cfg, incident)
            return 0
        if args.once:
            return 0
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
