#!/usr/bin/env bash
# Coverage gate: 100% lines, 100% regions and at least 95% branches.
#
# Reads the summary that `cargo llvm-cov report --branch --json --summary-only`
# prints, from the file given as the first argument or from stdin, and exits
# non-zero naming every total that falls short.
#
# Integer counts are compared rather than the rounded `percent` fields, so 99.99%
# can never pass as 100%. Lines and regions must have a non-zero count: a count
# of zero means nothing was measured (for example an ignore pattern that excludes
# every file), which must fail rather than pass vacuously. Branches are the only
# total allowed a count of zero, because a change may contain no branches.

set -euo pipefail

summary="$(cat "${1:-/dev/stdin}")"

total() {
  # Prints "<covered> <count>" for the named total.
  jq -er --arg kind "$1" '.data[0].totals[$kind] | "\(.covered) \(.count)"' <<<"$summary"
}

failed=0

for kind in lines regions; do
  read -r covered count < <(total "$kind")
  if (( count == 0 )); then
    echo "coverage gate: ${kind}: nothing was measured (count is 0)" >&2
    failed=1
  elif (( covered != count )); then
    echo "coverage gate: ${kind}: ${covered}/${count} covered; 100% is required" >&2
    failed=1
  else
    echo "coverage gate: ${kind}: ${covered}/${count} covered"
  fi
done

read -r covered count < <(total branches)
if (( count > 0 && covered * 100 < 95 * count )); then
  echo "coverage gate: branches: ${covered}/${count} covered; at least 95% is required" >&2
  failed=1
else
  echo "coverage gate: branches: ${covered}/${count} covered"
fi

exit "$failed"
