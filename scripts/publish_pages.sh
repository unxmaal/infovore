#!/usr/bin/env bash
# usage: publish_pages.sh SITE_DIR [REMOTE] [BRANCH]
# Scans SITE_DIR, copies its HTML and search index onto the orphan BRANCH, pushes, and
# enables GitHub Pages from that branch if the repo has none.
set -euo pipefail
site=${1:?usage: publish_pages.sh SITE_DIR [REMOTE] [BRANCH]}
remote=${2:-origin}
branch=${3:-gh-pages}
root=$(git rev-parse --show-toplevel)
cd "$root"
[ -f "$site/index.html" ] || { echo "no index.html in $site; build with --html first" >&2; exit 1; }
uv run infovore wiki scan "$site"

wt=$(mktemp -d)/site
trap 'git worktree remove --force "$wt" >/dev/null 2>&1 || true' EXIT
if git ls-remote --exit-code --heads "$remote" "$branch" >/dev/null 2>&1; then
  git fetch -q "$remote" "$branch"
  git worktree add -q "$wt" "$remote/$branch" --detach
else
  git worktree add -q --orphan -b "$branch" "$wt"
fi
find "$wt" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp "$site"/*.html "$site"/*.gif "$site"/search.json "$wt"/
touch "$wt/.nojekyll"
(
  cd "$wt"
  git add -A
  if git diff --cached --quiet; then echo "nothing changed"; exit 0; fi
  git commit -q -m "Publish wiki $(date -u +%Y-%m-%dT%H:%MZ): $(ls ./*.html | wc -l | tr -d ' ') pages, $(ls ./*.gif | wc -l | tr -d ' ') images"
  git push -q "$remote" "HEAD:refs/heads/$branch"
  echo "pushed $(git rev-parse --short HEAD) to $remote/$branch"
)

repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)
if ! gh api "repos/$repo/pages" >/dev/null 2>&1; then
  gh api -X POST "repos/$repo/pages" -f build_type=legacy -f "source[branch]=$branch" -f "source[path]=/" >/dev/null
  echo "enabled Pages from $branch"
fi
gh api "repos/$repo/pages" -q .html_url
