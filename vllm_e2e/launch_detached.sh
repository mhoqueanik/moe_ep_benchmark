#!/bin/bash
# Detached bench launcher: survives the parent shell/session dying.
#   bash launch_detached.sh <jobid> <name> <container command...>
# Log: logs/<name>.log ; completion marker: logs/<name>.done
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBID_ARG=$1; NAME=$2; shift 2
CMD="$*"
setsid nohup bash -c "
  JOBID=$JOBID_ARG bash '$HERE/in_container.sh' '$CMD' > '$HERE/logs/$NAME.log' 2>&1
  echo \$? > '$HERE/logs/$NAME.done'
" > /dev/null 2>&1 &
disown
echo "launched $NAME (pid $!) -> logs/$NAME.log"
