#!/usr/bin/env bash
set -eo pipefail
root="/agibot/data/home/agi/x2dev/x2_stair_autonomy"
aimdk_setup="/home/agi/x2dev/aimdk-v1/aimdk-aarch64-a424add7-artifacts/install/setup.bash"
out="${1:-$HOME/x2dev/stair-bags/$(date -u +%Y%m%dT%H%M%SZ)}"
inventory="${2:-$root/validation/pc2_compatibility.json}"
source /opt/ros/humble/setup.bash
source "$HOME/.aima/env/bashrc"
test -f "$aimdk_setup" || { echo "AimDK runtime missing: $aimdk_setup" >&2; exit 22; }
source "$aimdk_setup"
source "$root/ros_ws/install/local_setup.bash"
set -u
export PYTHONPATH="$root/src:${PYTHONPATH:-}"
python3 -m x2_stair_autonomy.compatibility --output "$inventory"
exec python3 -m x2_stair_autonomy.recording "$inventory" --output "$out" --execute
