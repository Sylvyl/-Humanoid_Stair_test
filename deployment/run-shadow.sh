#!/usr/bin/env bash
set -eo pipefail
root="/agibot/data/home/agi/x2dev/x2_stair_autonomy"
aimdk_setup="/home/agi/x2dev/aimdk-v1/aimdk-aarch64-a424add7-artifacts/install/setup.bash"
source /opt/ros/humble/setup.bash
source "$HOME/.aima/env/bashrc"
test -f "$aimdk_setup" || { echo "AimDK v1 runtime is missing: $aimdk_setup" >&2; exit 22; }
source "$aimdk_setup"
source "$root/ros_ws/install/local_setup.bash"
set -u
export X2_STAIR_CORE="$root/src"
exec ros2 run x2_stair_shadow observer
