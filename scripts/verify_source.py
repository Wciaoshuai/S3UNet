"""Verify source files without importing PyTorch or requiring data/weights."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
for relative,expected in manifest['files'].items():
    path=root/relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Source verification failed: '+relative)
print(f"Verified {len(manifest['files'])} source/configuration/documentation files")
