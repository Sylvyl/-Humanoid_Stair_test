[CmdletBinding()]
param(
    [ValidateSet('test', 'dashboard', 'smoke')]
    [string]$Mode = 'test',
    [int]$Port = 8443
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$candidates = @(@(
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1),
    (Get-Command python3 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1),
    (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) })
if (-not $candidates) { throw 'Python 3.10+ was not found. Install Python before running the local tools.' }
$python = $candidates[0]
$env:PYTHONPATH = Join-Path $root 'src'
switch ($Mode) {
    'test' { & $python -m unittest discover -s (Join-Path $root 'tests') -v }
    'dashboard' { & $python -m x2_stair_autonomy.server --host 127.0.0.1 --port $Port }
    'smoke' { & $python (Join-Path $root 'training\x2_stairs\smoke_train.py') --episodes 32 --seed 7 }
}
exit $LASTEXITCODE
