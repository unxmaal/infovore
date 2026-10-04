#!/bin/zsh -l
# Unattended infovore run loop (ingest, chunk, relevance cascade; no LLM) in tmux, Mac kept awake. Resume-safe.
#   ~/infovore/run-unattended.sh start --i-approved
#   ~/infovore/run-unattended.sh stop    # end it (work in progress is kept)
set -u
SELF=$HOME/infovore/run-unattended.sh
cd ~/projects/github/unxmaal/infovore
set -a; source ~/.infovore.env; set +a
LOG=~/infovore/logs; mkdir -p $LOG

case "${1:-start}" in
  start)
    [ "${2:-}" = "--i-approved" ] || { echo "refusing: start needs --i-approved (explicit approval)" >&2; exit 1; }
    tmux new-session -d -s infovore -n run "caffeinate -dimsu $SELF loop"
    echo "started: tmux attach -t infovore  (logs in $LOG)"
    ;;
  stop)
    tmux kill-session -t infovore && echo stopped
    ;;
  loop)
    while true; do
      uv run infovore run >>$LOG/run.log 2>&1
      rc=$?
      if [ $rc -eq 0 ]; then echo "$(date) run exited cleanly" >>$LOG/run.log; break; fi
      echo "$(date) run exited $rc, restarting in 60s" >>$LOG/run.log; sleep 60
    done
    ;;
  *)
    echo "usage: $SELF start|stop" >&2
    exit 2
    ;;
esac
