#!/usr/bin/env bash
set -euo pipefail

RCLONE_ZIP="${RUNNER_TEMP:-/tmp}/rclone-current-linux-amd64.zip"
RCLONE_DIR="${RUNNER_TEMP:-/tmp}/rclone-bin"

rm -rf "$RCLONE_DIR"
mkdir -p "$RCLONE_DIR"
curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o "$RCLONE_ZIP"
python3 - "$RCLONE_ZIP" "$RCLONE_DIR" <<'PY'
import pathlib
import sys
import zipfile

zip_path = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as archive:
    for member in archive.namelist():
        if member.endswith("/rclone"):
            target = out_dir / "rclone"
            target.write_bytes(archive.read(member))
            target.chmod(0o755)
            break
    else:
        raise SystemExit("rclone binary not found in archive")
PY
echo "$RCLONE_DIR" >> "$GITHUB_PATH"
"$RCLONE_DIR/rclone" version
