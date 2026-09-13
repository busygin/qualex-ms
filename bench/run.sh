#!/bin/bash
# bench/run.sh -- run solver variants over benchmark suites, resumably.
#
# Usage: bench/run.sh [-j WORKERS] [-g GPUS] [-n REGEX] SUITES VARIANTS
#
#   SUITES    comma-separated, from: dimacs (the 70 DIMACS graphs outside the
#             ten big ones), dimacs-big (those ten), ru and rw (the 120 uniform
#             random graphs with n <= 1000, unweighted and weighted), ru2000
#             and rw2000 (the 12 with n = 2000)
#   VARIANTS  comma-separated names from bench/variants
#   -j        parallel workers (default 6)
#   -g        comma-separated GPU ids the workers take turns on (default all)
#   -n        only the graphs whose names match this extended regex
#
# Results go to bench/runs/bin-<sha1 of the solver binary>/ (QMS_BENCH_RUNS
# overrides bench/runs), so the results of one build stay together:
#
#   res/<suite>/<variant>.<worker>.res   "graph value seconds" per finished run
#   log/<suite>/<variant>/<graph>.out    the solver's output, stderr included
#   work/<suite>/<variant>/              links to the graphs, and the .sol files
#   bin/qms                              the binary the results came from
#   SOURCE                               git describe and date when created
#
# A recorded run is skipped, so after a reboot the same command picks up where
# it stopped; a run without a result (a crash, or cut off) is done again.  Every
# QMS_* variable the variant does not set is unset.  DIMACS graphs are read
# from $QMS_DIMACS (default ~/DIMACS); the random graphs are generated into
# bench/graphs/rnd on first use, identically each time.  Do not run two
# invocations with the same suite and variant at once.
set -u
BENCH=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$BENCH")
RUNS=${QMS_BENCH_RUNS:-$BENCH/runs}
DIMACS=${QMS_DIMACS:-$HOME/DIMACS}
RND=$BENCH/graphs/rnd

usage() { sed -n '4,14p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

WORKERS=6 GPUS="" FILTER=""
while getopts "j:g:n:h" opt; do
  case $opt in
    j) WORKERS=$OPTARG ;;
    g) GPUS=$OPTARG ;;
    n) FILTER=$OPTARG ;;
    *) usage ;;
  esac
done
shift $((OPTIND-1))
[ $# -eq 2 ] || usage
IFS=, read -ra SUITES <<< "$1"
IFS=, read -ra VARIANTS <<< "$2"

# settings_of VARIANT prints the environment settings of a variant
settings_of() {
  awk -v v="$1" '{ sub(/#.*/, "") } NF && $1 == v { $1 = ""; print; found = 1 }
                 END { exit !found }' "$BENCH/variants"
}

# random_names SIZE... prints the random graph names, generating the graphs
random_names() {
  local n
  for n in "$@"; do
    if [ ! -s "$RND/n$n.lst" ]; then
      python3 "$REPO/tools/random_graphs.py" --sizes "$n" "$RND" > /dev/null || return 1
    fi
    cat "$RND/n$n.lst"
  done
}

suite_names() {
  case $1 in
    dimacs)        cat "$BENCH/lists/dimacs.lst" ;;
    dimacs-big)    cat "$BENCH/lists/dimacs_big.lst" ;;
    ru|rw)         random_names 200 300 500 1000 ;;
    ru2000|rw2000) random_names 2000 ;;
    *) echo "unknown suite: $1" >&2; return 1 ;;
  esac
}

# recorded SUITE VARIANT GRAPH succeeds when that run has a result
recorded() {
  cat "$RUN/res/$1/$2".*.res 2>/dev/null |
    awk -v g="$3" '$1 == g { found = 1 } END { exit !found }'
}

run_one() {  # SUITE VARIANT GRAPH GPU WORKER
  local suite=$1 v=$2 name=$3 gpu=$4 id=$5
  local work=$RUN/work/$suite/$v log=$RUN/log/$suite/$v/$name.out
  local graph extra=() env_args=() var t0 t1 value
  case $suite in
    dimacs|dimacs-big) graph=$DIMACS/$name.clq.b ;;
    ru|ru2000)         graph=$RND/$name.clq.b ;;
    rw|rw2000)         graph=$RND/$name.clq.b; extra=("-w$RND/$name.w") ;;
  esac
  mkdir -p "$work" "$RUN/log/$suite/$v" "$RUN/res/$suite"
  ln -sf "$graph" "$work/$name.clq.b"
  for var in $(compgen -e | grep '^QMS_'); do env_args+=(-u "$var"); done
  [ -n "$gpu" ] && env_args+=("CUDA_VISIBLE_DEVICES=$gpu")
  t0=$(date +%s.%N)
  env "${env_args[@]}" $(settings_of "$v") "$RUN/bin/qms" "$work/$name.clq.b" \
    "${extra[@]}" > "$log" 2>&1
  t1=$(date +%s.%N)
  value=$(sed -n 's/.*_w >= \([-0-9.e+]*\), time=.*/\1/p' "$log" | tail -1)
  if [ -z "$value" ]; then
    echo "no result for $suite $v $name, see $log" >&2
    return
  fi
  printf "%s %s %.2f\n" "$name" "$value" "$(echo "$t1 - $t0" | bc)" \
    >> "$RUN/res/$suite/$v.$id.res"
}

worker() {  # ID GPU JOBFILE
  local suite name v
  while read -r suite name; do
    for v in "${VARIANTS[@]}"; do
      recorded "$suite" "$v" "$name" || run_one "$suite" "$v" "$name" "$2" "$1"
    done
  done < "$3"
}

for v in "${VARIANTS[@]}"; do
  settings_of "$v" > /dev/null || { echo "unknown variant: $v" >&2; exit 1; }
done

make -s -C "$REPO" > /dev/null || { echo "the solver does not build" >&2; exit 1; }
SHA=$(sha1sum "$REPO/qualex-ms" | cut -d' ' -f1)
RUN=$RUNS/bin-${SHA:0:10}
mkdir -p "$RUN/bin"
if [ ! -x "$RUN/bin/qms" ]; then
  cp "$REPO/qualex-ms" "$RUN/bin/qms"
  printf "binary  %s\nsource  %s\ncreated %s on %s\n" "$SHA" \
    "$(git -C "$REPO" describe --always --dirty 2>/dev/null)" \
    "$(date -Iseconds)" "$(hostname)" > "$RUN/SOURCE"
fi

JOBS=$RUN/jobs.$$
: > "$JOBS"
for s in "${SUITES[@]}"; do
  names=$(suite_names "$s") || exit 1
  for name in $names; do
    if [ -n "$FILTER" ] && ! [[ $name =~ $FILTER ]]; then continue; fi
    echo "$s $name" >> "$JOBS"
  done
done

if [ -z "$GPUS" ]; then
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)
fi
IFS=, read -ra GPU_IDS <<< "$GPUS"

echo "$RUN: $(wc -l < "$JOBS") graphs, variants ${VARIANTS[*]}, $WORKERS workers" >&2
for ((k = 0; k < WORKERS; k++)); do
  awk -v k=$k -v m="$WORKERS" 'NR % m == k' "$JOBS" > "$JOBS.$k"
  gpu=""
  [ ${#GPU_IDS[@]} -gt 0 ] && gpu=${GPU_IDS[$((k % ${#GPU_IDS[@]}))]}
  worker $k "$gpu" "$JOBS.$k" &
done
wait
rm -f "$JOBS" "$JOBS".*

for s in "${SUITES[@]}"; do
  for v in "${VARIANTS[@]}"; do
    printf "%-10s %-8s %4d results\n" "$s" "$v" \
      "$(cat "$RUN/res/$s/$v".*.res 2>/dev/null | wc -l)" >&2
  done
done
