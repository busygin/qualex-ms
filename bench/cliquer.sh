#!/bin/bash
# bench/cliquer.sh -- prove optima of the random graphs with cliquer, resumably.
#
# Usage: bench/cliquer.sh [-a] [-j JOBS] [-l SECONDS] [-n REGEX] SUITE
#
#   SUITE  ru or rw (n <= 1000), ru2000 or rw2000; the ru suites use -u
#   -a     every graph of the suite, not only the cells known to be provable
#   -j     graphs solved at once (default 8)
#   -l     time limit per graph in seconds (default 3600)
#   -n     only the graphs whose names match this extended regex
#
# Appends "graph optimum seconds", or "graph TIMEOUT limit", to
# bench/optima/u.tsv or bench/optima/w.tsv, smallest and sparsest graphs first,
# running cliquer at nice 10.  A graph with an optimum is skipped, and so is a
# TIMEOUT unless -l gives it more time than it had.  cliquer is $CLIQUER
# (default ~/cliquer-1.21/cl); it reads the ASCII DIMACS files that
# tools/random_graphs.py writes with --cliquer, generated into bench/graphs/rnd
# if missing.
#
# Without -a only the cells cliquer 1.21 proved within an hour per graph on
# 2026-09-12 are attempted, the reference set of benchmarks.md: weighted graphs
# with n = 200 and p <= 0.9, n = 300 and p <= 0.8, n = 500 and p <= 0.7, and
# n = 1000 and p <= 0.5 (the slowest took 47 minutes); unweighted ones with
# n = 200 and p <= 0.8, n = 300 and p <= 0.7, n = 500 and p <= 0.5, and
# n = 1000 and p = 0.25.  Nothing with n = 2000 was proved.
set -u
BENCH=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$BENCH")
RND=$BENCH/graphs/rnd
CLIQUER=${CLIQUER:-$HOME/cliquer-1.21/cl}

usage() { sed -n '4,10p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

ALL=0 JOBS=8 LIMIT=3600 FILTER=""
while getopts "aj:l:n:h" opt; do
  case $opt in
    a) ALL=1 ;;
    j) JOBS=$OPTARG ;;
    l) LIMIT=$OPTARG ;;
    n) FILTER=$OPTARG ;;
    *) usage ;;
  esac
done
shift $((OPTIND-1))
[ $# -eq 1 ] || usage
case $1 in
  ru)     MODE=u; SIZES="200 300 500 1000" ;;
  rw)     MODE=w; SIZES="200 300 500 1000" ;;
  ru2000) MODE=u; SIZES=2000 ;;
  rw2000) MODE=w; SIZES=2000 ;;
  *) usage ;;
esac
if [ "$MODE" = w ]; then
  KNOWN='^g(200_(25|50|70|80|90)|300_(25|50|70|80)|500_(25|50|70)|1000_(25|50))_[0-9]+$'
else
  KNOWN='^g(200_(25|50|70|80)|300_(25|50|70)|500_(25|50)|1000_25)_[0-9]+$'
fi
[ -x "$CLIQUER" ] || { echo "no cliquer at $CLIQUER" >&2; exit 1; }
OUT=$BENCH/optima/$MODE.tsv
mkdir -p "$BENCH/optima"
touch "$OUT"

for n in $SIZES; do
  if [ ! -s "$RND/n$n.lst" ] || [ ! -s "$RND/$(head -1 "$RND/n$n.lst").clq" ]; then
    python3 "$REPO/tools/random_graphs.py" --sizes "$n" --cliquer "$RND" > /dev/null || exit 1
  fi
done

solve() {  # GRAPH
  local flag="" out status t0 t1 value
  [ "$MODE" = u ] && flag=-u
  t0=$(date +%s.%N)
  out=$(timeout "$LIMIT" nice -n 10 "$CLIQUER" -q -q $flag "$RND/$1.clq" 2>/dev/null)
  status=$?
  t1=$(date +%s.%N)
  if [ $status -eq 124 ]; then
    printf "%s TIMEOUT %s\n" "$1" "$LIMIT" >> "$OUT"
    return
  fi
  if [ "$MODE" = u ]; then
    value=$(sed -n 's/^size=\([0-9]*\),.*/\1/p' <<< "$out")
  else
    value=$(sed -n 's/.*weight=\([0-9.]*\):.*/\1/p' <<< "$out")
  fi
  if [ -n "$value" ]; then
    printf "%s %s %.1f\n" "$1" "$value" "$(echo "$t1 - $t0" | bc)" >> "$OUT"
  else
    echo "cliquer gave no result for $1" >&2
  fi
}
export -f solve
export MODE LIMIT CLIQUER RND OUT

for n in $SIZES; do cat "$RND/n$n.lst"; done |
  while read -r name; do
    if [ "$ALL" = 0 ] && ! [[ $name =~ $KNOWN ]]; then continue; fi
    if [ -n "$FILTER" ] && ! [[ $name =~ $FILTER ]]; then continue; fi
    awk -v g="$name" -v l="$LIMIT" '$1 == g && ($2 != "TIMEOUT" || $3 + 0 >= l + 0) { found = 1 }
                                    END { exit !found }' "$OUT" && continue
    echo "$name"
  done |
  sort -t_ -k1.2,1n -k2,2n -k3,3n |
  xargs -r -P "$JOBS" -I{} bash -c 'solve "$1"' _ {}
