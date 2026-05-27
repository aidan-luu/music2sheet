#!/usr/bin/env python3
"""
Watchdog that monitors an agent's state file and halts the agent on:
  - successful completion (status == "done")
  - repeated failure (failure_count >= MAX_FAILURES)
  - stalled heartbeat (no heartbeat update in STALL_SECONDS)
  - panic button: presence of .orchestrator/PAUSE_ALL or .orchestrator/PAUSE_<X>
    sends SIGINT to the pane but keeps the watchdog running; on file removal
    the watchdog logs the resume transition. Pause is non-terminal.

Reads:    .orchestrator/state/<agent>.json
          .orchestrator/PAUSE_ALL, .orchestrator/PAUSE_<X>
Writes:   .orchestrator/incidents/<timestamp>-<agent>.json (on escalation/pause/resume)
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
    python agent_watchdog.py --all   # multi-agent monitor (one process watches A/B/C/D)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCH_DIR = REPO_ROOT / ".orchestrator"
STATE_DIR = ORCH_DIR / "state"
INCIDENT_DIR = ORCH_DIR / "incidents"
PAUSE_ALL_FILE = ORCH_DIR / "PAUSE_ALL"
TMUX_SESSION = "claude-dev"

# Pane layout matches launch_agents.sh: window "agents", panes 0..3 = A/B/C/D
PANE_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


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
    def pause_file(self) -> Path:
        return ORCH_DIR / f"PAUSE_{self.agent}"

    @property
    def tmux_target(self) -> str:
        return f"{TMUX_SESSION}:agents.{PANE_INDEX[self.agent]}"


@dataclass
class WatchdogState:
    """Runtime state — not persisted; lives only in the watchdog process."""
    paused: bool = False
    pause_kind: str = ""  # "ALL" or letter
    last_action: tuple[str, str] | None = None


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


def write_incident(cfg: WatchdogConfig, kind: str, state: dict | None, extra: dict | None = None) -> Path:
    INCIDENT_DIR.mkdir(parents=True, exist_ok=True)
    ts = now_iso().replace(":", "").replace("-", "")
    path = INCIDENT_DIR / f"{ts}-agent-{cfg.agent}-{kind}.json"
    payload: dict = {
        "kind": kind,
        "agent": cfg.agent,
        "at": now_iso(),
        "state_snapshot": state,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))
    return path


def tmux_send(target: str, message: str, submit: bool = True) -> None:
    args = ["tmux", "send-keys", "-t", target, "--", message]
    if submit:
        args.append("C-m")
    try:
        subprocess.run(args, check=False, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def tmux_signal(target: str, signal: str) -> None:
    """Send a control character (e.g. C-c) to a tmux target."""
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", target, signal],
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def signal_agent_stop(cfg: WatchdogConfig, reason: str) -> None:
    """Send a soft stop signal into the agent's tmux pane."""
    tmux_send(cfg.tmux_target, f"echo '[watchdog] halting agent {cfg.agent}: {reason}'")
    tmux_signal(cfg.tmux_target, "C-c")


def signal_agent_pause(cfg: WatchdogConfig, scope: str) -> None:
    """SIGINT the pane but leave it alive — agent re-enters polling on resume."""
    tmux_send(cfg.tmux_target, f"echo '[watchdog] PAUSED agent {cfg.agent} ({scope}). Remove pause file to resume.'")
    tmux_signal(cfg.tmux_target, "C-c")


def signal_agent_resume(cfg: WatchdogConfig) -> None:
    tmux_send(
        cfg.tmux_target,
        f"echo '[watchdog] RESUMED agent {cfg.agent}. Resume polling .orchestrator/tickets/.'",
    )


def page_orchestrator(cfg: WatchdogConfig, incident_path: Path) -> None:
    # Find the orchestrator pane — convention: window "orchestrator" if it exists,
    # else fall back to writing to incidents/ only.
    target = f"{TMUX_SESSION}:orchestrator"
    tmux_send(target, f"echo '[watchdog] AGENT {cfg.agent} ESCALATION: see {incident_path}'")


def read_pause_reason(path: Path) -> str:
    try:
        text = path.read_text().strip()
    except OSError:
        text = ""
    return text or "(no reason given)"


def evaluate(cfg: WatchdogConfig, state: dict | None) -> tuple[str, str] | None:
    """Return (kind, reason) if a TERMINAL action is needed, else None.

    Pause handling is done outside this function because it is non-terminal.
    """
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


def check_pause(cfg: WatchdogConfig) -> str | None:
    """Return 'ALL' or the agent letter if a pause file is present, else None."""
    if PAUSE_ALL_FILE.exists():
        return "ALL"
    if cfg.pause_file.exists():
        return cfg.agent
    return None


def step(cfg: WatchdogConfig, rt: WatchdogState) -> bool:
    """Run one watchdog cycle. Returns True if the watchdog should keep running."""
    pause_scope = check_pause(cfg)

    # Handle pause/resume transitions first (non-terminal).
    if pause_scope and not rt.paused:
        # Newly paused.
        reason = (
            read_pause_reason(PAUSE_ALL_FILE) if pause_scope == "ALL" else read_pause_reason(cfg.pause_file)
        )
        rt.paused = True
        rt.pause_kind = pause_scope
        state = read_state(cfg.state_file)
        incident = write_incident(cfg, "pause", state, extra={"scope": pause_scope, "reason": reason})
        print(f"[watchdog] PAUSE ({pause_scope}) agent-{cfg.agent}: {reason} -> {incident}")
        signal_agent_pause(cfg, pause_scope)
        return True  # keep watching for resume
    if not pause_scope and rt.paused:
        # Resuming.
        state = read_state(cfg.state_file)
        incident = write_incident(cfg, "resume", state, extra={"prior_scope": rt.pause_kind})
        print(f"[watchdog] RESUME agent-{cfg.agent} -> {incident}")
        signal_agent_resume(cfg)
        rt.paused = False
        rt.pause_kind = ""
        return True
    if rt.paused:
        # Still paused; skip terminal evaluation so we don't escalate stall/done while paused.
        return True

    # Normal terminal evaluation
    state = read_state(cfg.state_file)
    action = evaluate(cfg, state)
    if action is None:
        return True
    kind, reason = action
    incident = write_incident(cfg, kind, state)
    print(f"[watchdog] {kind}: agent-{cfg.agent}: {reason} -> {incident}")
    if kind == "completion":
        signal_agent_stop(cfg, "task done")
    else:
        signal_agent_stop(cfg, reason)
        page_orchestrator(cfg, incident)
    return False  # terminal: stop watching this agent


def watch_single(cfg: WatchdogConfig, once: bool = False) -> int:
    rt = WatchdogState()
    print(f"[watchdog] watching agent-{cfg.agent} (state={cfg.state_file}, target={cfg.tmux_target})")
    while True:
        keep_going = step(cfg, rt)
        if not keep_going:
            return 0
        if once:
            return 0
        time.sleep(cfg.poll_seconds)


def watch_all(poll_seconds: int, stall_seconds: int, max_failures: int) -> int:
    configs = [
        WatchdogConfig(agent=a, poll_seconds=poll_seconds, stall_seconds=stall_seconds, max_failures=max_failures)
        for a in PANE_INDEX
    ]
    runtimes = {c.agent: WatchdogState() for c in configs}
    live = {c.agent for c in configs}
    print(f"[watchdog] multi-agent mode: monitoring {sorted(live)}")
    while live:
        for cfg in configs:
            if cfg.agent not in live:
                continue
            keep_going = step(cfg, runtimes[cfg.agent])
            if not keep_going:
                live.discard(cfg.agent)
                print(f"[watchdog] agent-{cfg.agent} retired from monitoring")
        if not live:
            break
        time.sleep(poll_seconds)
    print("[watchdog] all agents reached terminal state; exiting")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--agent", choices=["A", "B", "C", "D"], help="watch a single agent")
    group.add_argument("--all", action="store_true", help="watch all four agents in one process")
    ap.add_argument("--poll", type=int, default=5)
    ap.add_argument("--stall", type=int, default=600)
    ap.add_argument("--max-failures", type=int, default=3)
    ap.add_argument("--once", action="store_true", help="single-agent mode: evaluate once and exit")
    args = ap.parse_args()

    if args.all:
        return watch_all(args.poll, args.stall, args.max_failures)

    cfg = WatchdogConfig(
        agent=args.agent,
        poll_seconds=args.poll,
        stall_seconds=args.stall,
        max_failures=args.max_failures,
    )
    return watch_single(cfg, once=args.once)


if __name__ == "__main__":
    sys.exit(main())
