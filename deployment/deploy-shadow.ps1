[CmdletBinding()]
param(
    [string]$RobotHost = '10.0.1.41',
    [string]$RobotUser = 'agi',
    [switch]$Execute,
    [switch]$Resume,
    [switch]$SyncSource
)
$ErrorActionPreference = 'Stop'
if (-not $Execute) {
    Write-Host 'Dry run only. This script deploys a read-only shadow observer and dashboard.'
    Write-Host 'Re-run with -Execute after scripts\probe-pc2.ps1 passes.'
    exit 0
}
if ($RobotHost -eq '10.0.1.40') { throw 'Refusing deployment to motion-control PC1.' }
$project = Split-Path -Parent $PSScriptRoot
$destination = '/agibot/data/home/agi/x2dev/x2_stair_autonomy'
$knownHosts = Join-Path $project '..\x2_sensor_dashboard\deployment\known_hosts'
$sshOptions = @('-o', "UserKnownHostsFile=$knownHosts", '-o', 'StrictHostKeyChecking=accept-new')
$target = "$RobotUser@$RobotHost"
Write-Host 'Checking for an existing deployment. Existing data will not be overwritten.'
& ssh @sshOptions $target "test ! -e '$destination'"
if ($LASTEXITCODE -ne 0) {
    if (-not $Resume) {
        throw "$destination already exists. Use -Resume only for the partial deployment created by this script."
    }
    Write-Host 'Resume requested. Verifying the existing target before rebuilding it in place.'
    $verifyResume = @'
root=/agibot/data/home/agi/x2dev/x2_stair_autonomy
missing=0

# README.md is intentionally not an identity marker: it is not required to build.
if test -f "$root/deployment/x2-stair-shadow.service"; then
  echo 'Deployment signature: shadow service found'
elif test -f "$root/deployment/last-remote-build.log" && grep -qF '== Verify shadow-only source ==' "$root/deployment/last-remote-build.log"; then
  echo 'Deployment signature: earlier shadow build log found'
else
  echo 'UNRECOGNIZED: neither the shadow service nor the earlier shadow build signature exists'
  exit 40
fi

for required in \
  ros2/x2_stair_interfaces/package.xml \
  ros2/x2_stair_shadow/package.xml \
  src/x2_stair_autonomy/__init__.py; do
  if test -f "$root/$required"; then
    echo "FOUND: $required"
  else
    echo "MISSING: $required"
    missing=1
  fi
done
test "$missing" -eq 0
'@
    $verifyBytes = [System.Text.Encoding]::UTF8.GetBytes($verifyResume)
    $verifyBase64 = [Convert]::ToBase64String($verifyBytes)
    & ssh @sshOptions $target "echo '$verifyBase64' | base64 -d | bash"
    if ($LASTEXITCODE -ne 0) { throw 'The existing target is incomplete or is not the X2 stair shadow deployment. See FOUND/MISSING details above; nothing was changed.' }
    if ($SyncSource) {
        Write-Host 'Verified target. Synchronizing project-owned source; tokens and real approvals are excluded.'
        $sourceDirectories = @('deployment', 'ros2', 'scripts', 'src', 'static', 'training', 'tests', 'validation') |
            ForEach-Object { Join-Path $project $_ }
        & scp @sshOptions -r @sourceDirectories "${target}:${destination}/"
        if ($LASTEXITCODE -ne 0) { throw 'Directory source synchronization failed.' }
        $safeConfigFiles = @('sensor_extrinsics.example.json', 'stair_profiles.json', 'vendor_approval.example.json') |
            ForEach-Object { Join-Path $project "config\$_" }
        & scp @sshOptions @safeConfigFiles "${target}:${destination}/config/"
        if ($LASTEXITCODE -ne 0) { throw 'Safe configuration synchronization failed.' }
        $rootFiles = @('pyproject.toml', 'README.md') | ForEach-Object { Join-Path $project $_ }
        & scp @sshOptions @rootFiles "${target}:${destination}/"
        if ($LASTEXITCODE -ne 0) { throw 'Root source synchronization failed.' }
    }
} else {
    & ssh @sshOptions $target "mkdir -p '/agibot/data/home/agi/x2dev'"
    & scp @sshOptions -r $project "${target}:/agibot/data/home/agi/x2dev/"
    if ($LASTEXITCODE -ne 0) { throw 'Upload failed.' }
}
$bootstrap = @'
set -eo pipefail
root=/agibot/data/home/agi/x2dev/x2_stair_autonomy
log="$root/deployment/last-remote-build.log"
exec > >(tee "$log") 2>&1
trap 'code=$?; test "$code" -eq 0 || echo "FAILED (exit $code)"' EXIT
echo '== Verify shadow-only source =='
if grep -R -n -E '/aima/mc/locomotion/velocity|/aima/hal/joint/.*/command' \
  --include='*.py' --include='*.cpp' --include='*.cc' "$root/src" "$root/ros2"; then
  echo 'STOP: robot command topic found in executable source' >&2
  exit 30
fi
echo '== Environment =='
source /opt/ros/humble/setup.bash
source "$HOME/.aima/env/bashrc"
uname -m
python3 --version
echo '== Prepare ROS workspace =='
mkdir -p "$root/ros_ws/src"
for package in x2_stair_interfaces x2_stair_shadow; do
  package_target="$root/ros2/$package"
  package_link="$root/ros_ws/src/$package"
  if [[ -L "$package_link" ]]; then
    [[ "$(readlink -f "$package_link")" == "$(readlink -f "$package_target")" ]] || { echo "Unexpected symlink: $package_link" >&2; exit 31; }
  elif [[ -e "$package_link" ]]; then
    echo "Unexpected non-symlink workspace entry: $package_link" >&2
    exit 32
  else
    ln -s "$package_target" "$package_link"
  fi
done
cd "$root/ros_ws"
echo '== Build ROS interfaces and observer =='
colcon build --event-handlers console_direct+ --packages-select x2_stair_interfaces x2_stair_shadow
echo '== Validate Python source without network/package installation =='
PYTHONPATH="$root/src" python3 -m compileall -q "$root/src" "$root/ros2/x2_stair_shadow"
PYTHONPATH="$root/src:$root" python3 -m unittest discover -s "$root/tests" -v
chmod 0755 "$root/scripts/"*.sh "$root/deployment/"*.sh
mkdir -p "$HOME/.config/systemd/user"
cp "$root/deployment/x2-stair-shadow.service" "$HOME/.config/systemd/user/"
cp "$root/deployment/x2-stair-dashboard.service" "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
echo 'Built successfully. Services are staged but intentionally NOT enabled or started.'
'@
$bootstrapBytes = [System.Text.Encoding]::UTF8.GetBytes($bootstrap)
$bootstrapBase64 = [Convert]::ToBase64String($bootstrapBytes)
& ssh @sshOptions -t $target "echo '$bootstrapBase64' | base64 -d | bash"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Detailed remote log: $destination/deployment/last-remote-build.log" -ForegroundColor Yellow
    Write-Host "Use SSH to display that file, then copy the first ERROR section." -ForegroundColor Yellow
    throw 'Remote shadow build failed; use the detailed error printed above, not this wrapper line.'
}
Write-Host 'Shadow deployment built. Generate TLS and inspect the token before manually starting services.'
Write-Host 'No service was enabled, no motion source was registered, and no robot command topic was published.'
