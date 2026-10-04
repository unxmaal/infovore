#!/usr/bin/env bash
# usage: red_green.sh BASE HEAD ; env RED_GREEN_REFACTOR="substr ..." marks pure-refactor tests
set -euo pipefail
base=${1:?usage: red_green.sh BASE HEAD}
head=${2:?usage: red_green.sh BASE HEAD}
root=$(git rev-parse --show-toplevel)
cd "$root"
tmp=$(mktemp -d)
wt="$tmp/base"
trap 'git worktree remove --force "$wt" >/dev/null 2>&1 || true; rm -rf "$tmp"' EXIT

files=()
while IFS= read -r f; do files+=("$f"); done < <(
  git diff --name-only --diff-filter=AM "$base...$head" | grep -E '^tests/.*\.py$' | grep -v '__init__' || true
)
if [ "${#files[@]}" -eq 0 ]; then echo "no changed test files"; exit 0; fi

run() { (cd "$1" && uv run pytest -p no:cacheprovider --no-cov -q -rA "${files[@]}" 2>&1) | grep -E '^(PASSED|FAILED|ERROR) [^ :]+\.py::' | sed -E 's/ - .*//' | sort -u || true; }

git worktree add --detach "$wt" "$base" >/dev/null
for f in "${files[@]}"; do
  mkdir -p "$wt/$(dirname "$f")"
  git show "$head:$f" >"$wt/$f"
done
run "$wt" >"$tmp/base.txt"

git worktree add --detach "$tmp/head" "$head" >/dev/null
run "$tmp/head" >"$tmp/head.txt"
git worktree remove --force "$tmp/head"

awk -v refactor="${RED_GREEN_REFACTOR:-}" '
  FILENAME == ARGV[1] { if ($1 == "PASSED") basepass[$2] = 1; next }
  {
    id = $2
    if ($1 != "PASSED") { print "HEAD-NOT-GREEN " id; bad++; next }
    seen++
    if (id in basepass) {
      skip = 0; n = split(refactor, r, " ")
      for (i = 1; i <= n; i++) if (r[i] != "" && index(id, r[i])) skip = 1
      if (skip) print "green/green(refactor) " id; else { print "NEVER-RED " id; never++ }
    } else { print "red->green " id; red++ }
  }
  END {
    printf "red->green: %d, never red: %d, head tests: %d\n", red, never, seen
    exit (bad || !seen) ? 1 : 0
  }
' "$tmp/base.txt" "$tmp/head.txt"
