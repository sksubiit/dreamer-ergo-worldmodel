"""Verify the self-contained public asset bundle without Git or network access."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET


def verify():
    root = Path(__file__).resolve().parents[1] / "assets/ergocub"
    manifest = json.loads((root / "provenance.json").read_text())
    for item in manifest["files"]:
        actual = hashlib.sha256((root / item["path"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Asset hash mismatch: {item['path']}")
    referenced = {m.get("filename").removeprefix("package://ergoCub/")
                  for m in ET.parse(root / "model.urdf").getroot().findall(".//mesh")}
    bundled = {item["path"] for item in manifest["files"] if item["path"].startswith("meshes/")}
    if referenced != bundled:
        raise ValueError("URDF mesh references and public asset manifest differ")
    print(f"Verified {len(manifest['files'])} files from {manifest['repository']} "
          f"at {manifest['revision']}; {manifest['license']}")
    return manifest


if __name__ == "__main__":
    verify()
