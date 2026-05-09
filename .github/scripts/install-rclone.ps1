$ErrorActionPreference = "Stop"

$zipPath = Join-Path $env:RUNNER_TEMP "rclone-current-windows-amd64.zip"
$extractDir = Join-Path $env:RUNNER_TEMP "rclone-bin"

if (Test-Path $extractDir) {
    Remove-Item -LiteralPath $extractDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
Invoke-WebRequest -Uri "https://downloads.rclone.org/rclone-current-windows-amd64.zip" -OutFile $zipPath
Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

$rclone = Get-ChildItem -Path $extractDir -Recurse -Filter rclone.exe | Select-Object -First 1
if ($null -eq $rclone) {
    throw "rclone.exe not found in archive"
}

$rcloneDir = Split-Path -Parent $rclone.FullName
Add-Content -Path $env:GITHUB_PATH -Value $rcloneDir
& $rclone.FullName version
