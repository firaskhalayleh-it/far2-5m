#!/bin/zsh
# Train one arm/seed on the 1,000-update contract, then every metric: taxonomy +
# T curve, slot/entity ordering probes, hint-row encoder probe.  The run is tied
# to a commit: it refuses a dirty research tree unless FAR2_ALLOW_DIRTY=1, and
# that override is recorded in the manifest.
#   diagnostics/measure.sh ARM SEED        (FAR2_UNLOCK=expand,split for operator arms)
set -o pipefail
cd "${0:A:h}/.." || exit 1
A=$1; S=$2; R=seeds/${A}_seed$S; RUN=runs_diag/$R
P=.venv/bin/python; D=diagnostics
DIRTY=$($P $D/run_manifest.py dirty)
if [[ -n "$DIRTY" && "$FAR2_ALLOW_DIRTY" != 1 ]]; then
  echo "refusing: uncommitted research changes (commit first, or FAR2_ALLOW_DIRTY=1)"; echo "$DIRTY"; exit 2
fi
mkdir -p $RUN runs_diag/seeds $D/results
$P $D/run_manifest.py start $RUN -- $D/measure.sh $A $S
LOG=runs_diag/seeds/${A}_seed$S.log
{ $P $D/train_seed.py $A $S && $P $D/analyze_broken.py $R 1000 \
  && $P $D/probe_upstream.py $R && $P $D/probe_hint_row.py $R; } 2>&1 \
  | grep -v "Warning\|sched.step\|Consider\|patch\|warnings.warn\|inactive_drift" > $LOG
STATUS=$?
$P $D/run_manifest.py finish $RUN $LOG
cp $LOG $D/results/${A}_seed$S.log; cp $RUN/run_manifest.json $D/results/${A}_seed$S.manifest.json
echo "exit=$STATUS $A seed $S"; cat $LOG
