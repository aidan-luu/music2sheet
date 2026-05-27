#!/usr/bin/env bash
# launch_agents.sh — start a 4-pane tmux session with one Claude agent per pane.
#
# Layout (2x2 tiled):
#   +--------------+--------------+
#   | Agent A      | Agent B      |
#   | Frontend     | Backend      |
#   +--------------+--------------+
#   | Agent C      | Agent D      |
#   | ML           | QA           |
#   +--------------+--------------+
#
# Each pane:
#   1. cd's into the agent's git worktree (created on demand if missing).
#   2. exports CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1.
#   3. launches `claude` (interactive) with a one-shot system-style prompt typed
#      via tmux send-keys, establishing the agent's role and ticket-polling loop.
#   4. The agent then watches .orchestrator/tickets/agent-<X>-*.json for work and
#      writes progress to .orchestrator/state/agent-<X>.json.
#
# The orchestrator (running separately) creates tickets in
# .orchestrator/tickets/, which each pane's agent picks up and executes.
#
# Usage:
#   bash .orchestrator/hooks/launch_agents.sh         # creates + attaches
#   bash .orchestrator/hooks/launch_agents.sh --no-attach  # creates only
#
# Stop with: tmux kill-session -t claude-dev

set -euo pipefail

SESSION="claude-dev"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ATTACH=1

for arg in "$@"; do
  case "$arg" in
    --no-attach) ATTACH=0 ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0 ;;
    *)
      echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not installed; aborting" >&2; exit 1
fi
if ! command -v claude >/dev/null 2>&1; then
  echo "claude CLI not on PATH; aborting" >&2; exit 1
fi

# Pick a Python interpreter for the watchdog. Prefer the project's venv, fall
# back to python3 on PATH.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  WATCHDOG_PY="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  WATCHDOG_PY="$(command -v python3)"
else
  echo "no python3 found; aborting (the watchdog needs it)" >&2; exit 1
fi

# Tear down any existing session AND any stale watchdog so we start clean
tmux kill-session -t "$SESSION" 2>/dev/null || true
pkill -f "agent_watchdog.py --all" 2>/dev/null || true

# Make sure each agent has a worktree to live in
mkdir -p "$REPO_ROOT/.orchestrator/worktrees"
for letter in A B C D; do
  wt="$REPO_ROOT/.orchestrator/worktrees/agent-${letter}"
  if [[ ! -d "$wt" ]]; then
    (cd "$REPO_ROOT" && git worktree add "$wt" -b "agent-${letter}/live" >/dev/null 2>&1 \
      || git worktree add "$wt" "agent-${letter}/live" >/dev/null 2>&1 \
      || git worktree add "$wt" >/dev/null 2>&1)
  fi
done

# Create the session with one window starting in the repo root
tmux new-session -ds "$SESSION" -x 240 -y 64 -c "$REPO_ROOT" -n agents

# Split into a 2x2 grid:
#   start: pane 0 is the whole window
#   split horizontal: pane 0 (left) | pane 1 (right)
#   split pane 0 vertical: pane 0 (top-left) / pane 2 (bottom-left)
#   split pane 1 vertical: pane 1 (top-right) / pane 3 (bottom-right)
tmux split-window -h -t "$SESSION:agents.0" -c "$REPO_ROOT"
tmux split-window -v -t "$SESSION:agents.0" -c "$REPO_ROOT"
tmux split-window -v -t "$SESSION:agents.1" -c "$REPO_ROOT"
tmux select-layout -t "$SESSION:agents" tiled >/dev/null

# Quote helper for send-keys (avoids shell-interpreting the prompt text)
sk() { tmux send-keys -t "$SESSION:agents.$1" -- "$2" C-m; }

# Per-pane agent map: pane_index : letter : role
agents=(
  "0:A:Frontend"
  "1:B:Backend"
  "2:C:ML"
  "3:D:QA"
)

for entry in "${agents[@]}"; do
  IFS=":" read -r pane letter role <<< "$entry"
  wt="$REPO_ROOT/.orchestrator/worktrees/agent-${letter}"

  # Banner + env + cd into worktree
  sk "$pane" "clear"
  sk "$pane" "printf '\\n=== Agent %s — %s ===\\nWorktree: %s\\n\\n' '$letter' '$role' '$wt'"
  sk "$pane" "export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1"
  sk "$pane" "cd '$wt'"

  # Launch claude interactively. --dangerously-skip-permissions makes the agent
  # autonomous within its worktree (still constrained by skills.md boundaries).
  sk "$pane" "claude --dangerously-skip-permissions"

  # Give claude a moment to come up before injecting the role prompt
  sleep 1
done

# Now inject the role prompt into each pane. Two-step (sleep between) so the
# claude UI is ready to accept input before we send the role description.
sleep 2

prompt_for() {
  local letter="$1" role="$2"
  cat <<EOF
You are Agent ${letter} — ${role} on a 4-agent team building an audio-to-lead-sheet ML system (a SheetSage successor). The orchestrator runs in a separate session.

Your role spec is in /Users/nhan-tuanaidanluu/sheet-sage-2/skills.md — read your section first, then read .orchestrator/dependency_graph.yaml. Your owned paths and boundaries are non-negotiable.

Workflow:
1. Poll /Users/nhan-tuanaidanluu/sheet-sage-2/.orchestrator/tickets/ every 30 seconds for files matching agent-${letter}-*.json with status "open".
2. When you find one, atomically claim it by renaming to .in-progress.json and updating your state file at /Users/nhan-tuanaidanluu/sheet-sage-2/.orchestrator/state/agent-${letter}.json.
3. Execute the task as described in the ticket body. Stay inside your owned paths.
4. On success: mark the ticket .done.json, commit your work on a feature branch named agent-${letter}/<pr-id>-<slug>, tag it ready-for-qa-<pr-id>, update your state to status: "done".
5. On failure: increment failure_count in state, write an incident under .orchestrator/incidents/, and resume polling. The watchdog escalates at failure_count >= 3.
6. Heartbeat: update state.heartbeat_at every minute even when idle.

Start by reading skills.md and reporting (under 60 words) which paths you own and which you do not. Then begin polling.
EOF
}

for entry in "${agents[@]}"; do
  IFS=":" read -r pane letter role <<< "$entry"
  prompt="$(prompt_for "$letter" "$role")"
  # Send the prompt as a single literal block via tmux paste-buffer
  tmux load-buffer -b "agent-${letter}-prompt" - <<< "$prompt"
  tmux paste-buffer -t "$SESSION:agents.$pane" -b "agent-${letter}-prompt" -d
  # Submit
  tmux send-keys -t "$SESSION:agents.$pane" Enter
done

# Start the multi-agent watchdog in the background. nohup'd so it survives
# the orchestrator shell exiting; logs land at .orchestrator/watchdog.log.
WATCHDOG_LOG="$REPO_ROOT/.orchestrator/watchdog.log"
nohup "$WATCHDOG_PY" "$REPO_ROOT/.orchestrator/hooks/agent_watchdog.py" --all \
  > "$WATCHDOG_LOG" 2>&1 &
WATCHDOG_PID=$!
disown $WATCHDOG_PID 2>/dev/null || true

echo "Session '$SESSION' is up with 4 agent panes."
echo "Layout: A=top-left, B=top-right, C=bottom-left, D=bottom-right."
echo "Watchdog: PID $WATCHDOG_PID (logs: $WATCHDOG_LOG)"
echo "  Stop watchdog: pkill -f 'agent_watchdog.py --all'"
echo "  Stop everything: tmux kill-session -t $SESSION && pkill -f 'agent_watchdog.py --all'"
echo
if [[ $ATTACH -eq 1 ]]; then
  echo "Attaching... (detach with Ctrl-b d)"
  exec tmux attach -t "$SESSION"
else
  echo "Attach later with: tmux attach -t $SESSION"
fi
