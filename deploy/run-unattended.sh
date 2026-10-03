#!/bin/zsh -l
# Unattended infovore live extraction (best-first) in tmux, Mac kept awake. Resume-safe.
#   INFOVORE_PROMPT_VERSION=vN ~/infovore/run-unattended.sh start --i-approved
#   ~/infovore/run-unattended.sh stop    # end it (work in progress is kept)
set -u
SELF=$HOME/infovore/run-unattended.sh
cd ~/projects/github/unxmaal/infovore
set -a; source ~/.infovore.env; set +a
LOG=~/infovore/logs; mkdir -p $LOG

case "${1:-start}" in
  start)
    [ "${2:-}" = "--i-approved" ] || { echo "refusing: start needs --i-approved (explicit approval)" >&2; exit 1; }
    [ -n "${INFOVORE_PROMPT_VERSION:-}" ] || { echo "refusing: INFOVORE_PROMPT_VERSION is not set" >&2; exit 1; }
    uv run infovore triage 2>&1 | tail -n 5 | tee -a $LOG/setup.log
    uv run infovore promote --prompt-version "$INFOVORE_PROMPT_VERSION" 2>&1 | tee -a $LOG/setup.log
    tmux new-session -d -s infovore -n extract "caffeinate -dimsu $SELF extract"
    echo "started: tmux attach -t infovore  (logs in $LOG)"
    ;;
  stop)
    tmux kill-session -t infovore && echo stopped
    ;;
  extract)
    while true; do
      uv run infovore extract --mode live --order best >>$LOG/extract.log 2>&1
      rc=$?
      if [ $rc -eq 0 ]; then echo "$(date) extract queue drained" >>$LOG/extract.log; break; fi
      echo "$(date) extract exited $rc, restarting in 60s" >>$LOG/extract.log; sleep 60
    done
    ;;
esac
