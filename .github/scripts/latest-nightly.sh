#!/usr/bin/env bash
# Prints the newest nightly, as `nightly-YYYY-MM-DD`, that ships both `rustfmt`
# and `llvm-tools` for x86_64-unknown-linux-gnu: the components the pinned
# nightly in `.github/nightly-version` is used for.
#
# It reads the release manifests rather than installing `nightly` and asking
# `rustc -V`, which reports the commit date, usually the day before the
# toolchain's name. The window is 14 dates, today (UTC) and the 13 before it,
# newest first; a date with no manifest is a nightly that was skipped.
#
# Exit codes: 0, a nightly was printed; 1, no date in the window qualifies;
# 2, bad input or environment; 3, a manifest could not be fetched or parsed.
#
# Environment, for testing: LATEST_NIGHTLY_TODAY overrides today's date
# (YYYY-MM-DD); LATEST_NIGHTLY_DIST overrides the dist root URL.

set -euo pipefail

today="${LATEST_NIGHTLY_TODAY:-$(date -u +%F)}"
dist="${LATEST_NIGHTLY_DIST:-https://static.rust-lang.org/dist}"

# `tomllib` arrived in Python 3.11; without this check its absence would read
# as "no component available".
if ! python3 -I -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
  echo "latest-nightly: python3 3.11 or newer is required" >&2
  exit 2
fi

# The shape is checked here and the calendar by Python, which also does the
# date arithmetic that GNU and BSD `date` spell differently.
window='
import datetime
import sys

today = datetime.date.fromisoformat(sys.argv[1])
for days in range(14):
    print(today - datetime.timedelta(days=days))
'
if ! [[ "$today" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] \
  || ! window_dates="$(python3 -I -c "$window" "$today" 2>/dev/null)"; then
  echo "latest-nightly: not a YYYY-MM-DD date: ${today}" >&2
  exit 2
fi
mapfile -t dates <<<"$window_dates"

# curl's own message is shown only when a fetch fails for real.
curl_errors="$(mktemp)"
trap 'rm -f "$curl_errors"' EXIT

# Reads a manifest on stdin and prints whether both components are available.
# A missing key means unavailable; a manifest that is not valid TOML, or is
# dated other than requested, fails.
verdict='
import sys
import tomllib

date = sys.argv[1]
manifest = tomllib.loads(sys.stdin.read())
dated = manifest.get("date")
if dated != date:
    sys.exit(f"the manifest is dated {dated!r}, not {date}")
triple = "x86_64-unknown-linux-gnu"
available = all(
    manifest.get("pkg", {}).get(name, {}).get("target", {}).get(triple, {}).get("available") is True
    for name in ("rustfmt-preview", "llvm-tools-preview")
)
print("available" if available else "unavailable")
'

for date in "${dates[@]}"; do
  status=0
  manifest="$(curl --fail --silent --show-error --location --retry 3 \
    --connect-timeout 10 --max-time 120 \
    "${dist}/${date}/channel-rust-nightly.toml" 2>"$curl_errors")" || status=$?
  case "$status" in
    0) ;;
    # An HTTP error (S3 answers 403 or 404 for a missing manifest), or a
    # missing `file://` manifest: no nightly that day.
    22 | 37) continue ;;
    *)
      cat "$curl_errors" >&2
      echo "latest-nightly: fetching the ${date} manifest failed (curl exit ${status})" >&2
      exit 3
      ;;
  esac
  if ! found="$(python3 -I -c "$verdict" "$date" <<<"$manifest")"; then
    echo "latest-nightly: the ${date} manifest could not be read" >&2
    exit 3
  fi
  if [ "$found" = available ]; then
    echo "nightly-${date}"
    exit 0
  fi
done

echo "latest-nightly: no nightly from ${dates[-1]} to ${today} ships rustfmt and llvm-tools" >&2
exit 1
