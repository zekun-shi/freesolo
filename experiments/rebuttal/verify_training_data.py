"""Verify an exact token file before rebuttal training or a labelled pilot."""

import argparse
import hashlib
import json
from pathlib import Path
import time

EXPECTED_BYTES = 13400004440
EXPECTED_SHA256 = "5a387c0c5e36f18c84e0465296a3c818b73d6d031876659219bbd19c6aad46f7"
CORPORA = {
    "341m": (EXPECTED_BYTES, EXPECTED_SHA256),
    "743m": (77960195380, "9ae37aba95ce006afa73954fc1bb7ad7c8843c9d41c291078a62386f0b5cdddb"),
    "gb200-pilot": (837680814, "c595b8f267874a5dfec4c81f93f80e21d6c86998853b0fcf7af6da70f7197935"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=CORPORA, default="341m")
    args = parser.parse_args()
    expected_bytes, expected_sha256 = CORPORA[args.profile]
    path = Path(args.data).resolve()
    before = path.stat()
    if before.st_size != expected_bytes:
        raise ValueError(f"Training data size differs: {before.st_size}")
    started = time.time()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Data changed during hashing")
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"Training data checksum differs: {digest.hexdigest()}")
    result = dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest(),
                  mtime_ns=after.st_mtime_ns, inode=after.st_ino,
                  verified_at=time.time(), seconds=time.time() - started, profile=args.profile)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
