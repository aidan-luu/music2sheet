#!/usr/bin/env bash
# Idempotent tmux session setup for the agent team.
#
# Re-running this script is safe: if `claude-dev` exists, panes are refreshed
# rather than re-created. Each window gets CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
# in its environment and starts an agent_watchdog.py process in the background.
#
# Usage: bash .orchestrator/hooks/tmux_session.sh
#
# Attach with: tmux attach -t claude-dev

set -euo pipefail

SESSION="claude-dev"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WATCHDOG="$REPO_ROOT/.orchestrator/hooks/agent_watchdog.py"

agents=(
  "A:frontend"
  "B:backend"
  "C:ml"
  "D:qa"
)

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not installed; aborting" >&2
  exit 1
fi

# Create session if it doesn't already exist
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -ds "$SESSION" -n orchestrator -x 220 -y 50
  tmux send-keys -t "$SESSION:orchestrator" \
    "export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 && cd '$REPO_ROOT' && echo '[orchestrator ready]'" C-m
fi

# Ensure one window per agent + watchdog running
for entry in "${agents[@]}"; do
  letter="${entry%%:*}"
  role="${entry##*:}"
  window="agent-${letter}-${role}"

  if ! tmux list-windows -t "$SESSION" -F '#W' | grep -qx "$window"; then
    tmux new-window -t "$SESSION" -n "$window"
  fi

  # Seed each window: set env var, cd into repo, launch watchdog.
  tmux send-keys -t "$SESSION:$window" \
    "export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 && cd '$REPO_ROOT'" C-m
  tmux send-keys -t "$SESSION:$window" \
    "python3 '$WATCHDOG' --agent $letter &" C-m
  tmux send-keys -t "$SESSION:$window" \
    "echo '[$window ready — watchdog PID '\$!]'" C-m
done

echo "tmux session '$SESSION' ready."
echo "Windows: $(tmux list-windows -t "$SESSION" -F '#W' | tr '\n' ' ')"
echo "Attach with: tmux attach -t $SESSION"
