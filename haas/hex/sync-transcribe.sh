#!/usr/bin/env bash
# sync-transcribe.sh <pin>: check out this fork's Haas changes on top of the
# transcribe.cpp commit <pin> (what a Hex release pins) in the build worktree
# $HAAS_HOME/transcribe.cpp-build, and print that commit.
#
# haas/BASE records the upstream commit the Haas commits sit on. When <pin>
# differs, the Haas commits are rebased onto it on a new branch haas-<pin7>
# (the current checkout is left alone). On conflicts the rebase stops in
# $HAAS_HOME/rebase-<pin7> for you (or Claude Code) to resolve and rerun.
set -euo pipefail
. "$(dirname "$0")/common.sh"

pin=${1:?usage: sync-transcribe.sh <transcribe.cpp commit>}
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
head=$(git -C "$repo" rev-parse HEAD)
base=$(git -C "$repo" show "$head:haas/BASE")

if ! git -C "$repo" cat-file -e "$pin^{commit}" 2>/dev/null; then
  say "fetching transcribe.cpp $pin"
  git -C "$repo" fetch --quiet "$TC_UPSTREAM" "$pin"
fi

if [ "$base" = "$pin" ]; then
  target=$head
else
  branch="haas-${pin:0:7}"
  wt="$HAAS_HOME/rebase-${pin:0:7}"
  say "rebasing the Haas commits from ${base:0:7} onto ${pin:0:7} (branch $branch)"
  git -C "$repo" worktree remove --force "$wt" 2>/dev/null || true
  git -C "$repo" branch -f "$branch" "$head"
  git -C "$repo" worktree add --quiet "$wt" "$branch"
  if ! git -C "$wt" rebase --quiet --onto "$pin" "$base"; then
    cat >&2 <<MSG
error: the rebase onto ${pin:0:7} has conflicts, left in
  $wt
Resolve them there (git status; edit; git add; git rebase --continue), then:
  echo $pin > $wt/haas/BASE && git -C $wt commit -qam "haas: base on transcribe.cpp ${pin:0:7}"
and rerun the update from that worktree's haas/hex/update.sh.
MSG
    exit 1
  fi
  echo "$pin" > "$wt/haas/BASE"
  git -C "$wt" commit --quiet -am "haas: base on transcribe.cpp ${pin:0:7}"
  target=$(git -C "$wt" rev-parse HEAD)
  git -C "$repo" worktree remove "$wt"
fi

if [ -e "$TC_BUILD_DIR/.git" ]; then
  git -C "$TC_BUILD_DIR" checkout --quiet --detach "$target"
else
  mkdir -p "$HAAS_HOME"
  git -C "$repo" worktree add --quiet --detach "$TC_BUILD_DIR" "$target"
fi
echo "$target"
