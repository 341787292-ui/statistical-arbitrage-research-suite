from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify authors' public data files.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("paper_reproduction/data/official_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_reproduction/data"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "dlsa-reproduction/1.0"})
    with urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloadable = [
        item
        for item in manifest["files"]
        if item.get("published") and item.get("url") and item.get("sha256")
    ]
    for item in downloadable:
        destination = args.output_dir / item["filename"]
        expected = item["sha256"].lower()
        if destination.exists() and not args.force and sha256(destination) == expected:
            print(f"verified existing: {destination.name}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"downloading: {destination.name}")
        download(item["url"], temporary)
        observed = sha256(temporary)
        if observed != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {destination.name}: {observed} != {expected}"
            )
        temporary.replace(destination)
        print(f"verified: {destination.name}")


if __name__ == "__main__":
    main()
