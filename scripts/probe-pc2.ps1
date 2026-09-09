[CmdletBinding()]
param(
    [string]$RobotHost = '10.0.1.41',
    [string]$RobotUser = 'agi',
    [string]$ExpectedSdk = 'aimdk v1.0.0-ga424add'
)
$ErrorActionPreference = 'Stop'
if ($RobotHost -eq '10.0.1.40') { throw 'PC1 is prohibited for secondary-development deployment.' }
$root = Split-Path -Parent $PSScriptRoot
$knownHosts = Join-Path $root '..\x2_sensor_dashboard\deployment\known_hosts'
$sshOptions = @('-o', "UserKnownHostsFile=$knownHosts", '-o', 'StrictHostKeyChecking=accept-new')
if (-not (Test-NetConnection -ComputerName $RobotHost -Port 22 -InformationLevel Quiet)) {
    throw "SSH is unreachable at $RobotHost. Verify Ethernet is 10.0.1.2/24."
}
$probe = @'
set -eo pipefail
root=/agibot/data/home/agi/x2dev/x2_stair_autonomy
aimdk_setup=/home/agi/x2dev/aimdk-v1/aimdk-aarch64-a424add7-artifacts/install/setup.bash
echo '---SYSTEM---'
uname -m
. /etc/os-release
echo "$VERSION_ID"
test -f /opt/ros/humble/setup.bash && echo ROS_HUMBLE_OK
test -f "$HOME/.aima/env/bashrc" && echo AIMA_ENV_OK
source /opt/ros/humble/setup.bash
source "$HOME/.aima/env/bashrc"
test -f "$aimdk_setup" || { echo 'AIMDK_V1_RUNTIME_MISSING'; exit 30; }
source "$aimdk_setup"
python3 -c 'from aimdk_msgs.msg import JointStateArray, PmuState; print("AIMDK_MSGS_OK")'
echo '---TOPICS---'
ros2 topic list | sort
echo '---SERVICES---'
ros2 service list | sort
echo '---TYPES---'
ros2 topic type /aima/hal/sensor/rgbd_head_front/depth_image
ros2 topic type /aima/hal/sensor/lidar_chest_front/lidar_pointcloud
ros2 topic type /aima/hal/joint/leg/state
echo '---QOS---'
ros2 topic info -v /aima/hal/sensor/lidar_chest_front/lidar_pointcloud
echo '---HOST---'
df -h /agibot/data/home/agi
uptime
if test -d "$root/src/x2_stair_autonomy"; then
  echo '---MACHINE_READABLE_REPORT---'
  export PYTHONPATH="$root/src:${PYTHONPATH:-}"
  python3 -m x2_stair_autonomy.compatibility --output "$root/validation/pc2_compatibility.json"
fi
'@
Write-Host 'Read-only compatibility probe. Enter the agi password at the SSH prompt.'
$probeBytes = [System.Text.Encoding]::UTF8.GetBytes($probe)
$probeBase64 = [Convert]::ToBase64String($probeBytes)
& ssh @sshOptions "$RobotUser@$RobotHost" "echo '$probeBase64' | base64 -d | bash"
if ($LASTEXITCODE -ne 0) { throw "PC2 probe failed with exit code $LASTEXITCODE" }
Write-Host "Local SDK baseline: $ExpectedSdk"
Write-Host 'Probe complete. No files or services were changed on the robot.'
