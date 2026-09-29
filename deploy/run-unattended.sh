#!/bin/zsh -l
# Unattended infovore run: live extraction (best-first) plus a probe loop,
# side by side in tmux, with the Mac kept awake. Resume-safe: rerun any time.
#   ~/infovore/run-unattended.sh start   # prepare, then launch tmux session "infovore"
#   ~/infovore/run-unattended.sh stop    # end it (work in progress is kept)
set -u
SELF=$HOME/infovore/run-unattended.sh
cd ~/projects/github/unxmaal/infovore
set -a; source ~/.infovore.env; set +a
LOG=~/infovore/logs; mkdir -p $LOG

case "${1:-start}" in
  start)
    uv run infovore triage 2>&1 | tail -n 5 | tee -a $LOG/setup.log
    uv run infovore promote --prompt-version "${INFOVORE_PROMPT_VERSION:-v5}" 2>&1 | tee -a $LOG/setup.log
    tmux new-session -d -s infovore -n extract "caffeinate -dimsu $SELF extract"
    tmux new-window -t infovore -n probe "caffeinate -dimsu $SELF probe"
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
  probe)
    while true; do
      uv run infovore probe >>$LOG/probe.log 2>&1
      echo "$(date) probe pass exited $?" >>$LOG/probe.log
      sleep 600
    done
    ;;
esac
