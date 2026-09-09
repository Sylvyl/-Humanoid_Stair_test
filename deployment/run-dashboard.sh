#!/usr/bin/env bash
set -euo pipefail
root="/agibot/data/home/agi/x2dev/x2_stair_autonomy"
aimdk_setup="/home/agi/x2dev/aimdk-v1/aimdk-aarch64-a424add7-artifacts/install/setup.bash"
cert="$root/config/tls/cert.pem"
key="$root/config/tls/key.pem"
test -f "$cert" || { echo "Generate TLS first with scripts/generate-tls.sh" >&2; exit 20; }
test -f "$root/config/operator-token" || { echo "Operator token is missing" >&2; exit 21; }
set +u
source /opt/ros/humble/setup.bash
source "$HOME/.aima/env/bashrc"
test -f "$aimdk_setup" || { echo "AimDK v1 runtime is missing: $aimdk_setup" >&2; exit 22; }
source "$aimdk_setup"
source "$root/ros_ws/install/local_setup.bash"
set -u
export PYTHONPATH="$root/src:${PYTHONPATH:-}"
exec python3 -m x2_stair_autonomy.server --host 10.0.1.41 --port 8443 \
  --cert "$cert" --key "$key" --token-file "$root/config/operator-token" --ros
