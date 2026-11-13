"""
Downloads the hand-labelled Sen1Floods11 subset (Sentinel-1 chips, labels, splits)
from the public Google Cloud bucket over plain HTTPS. Standard library only.

Usage:  python download_sen1floods11.py --out data
"""
import argparse
import json
import os
import socket
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

socket.setdefaulttimeout(60)  # a stalled connection raises instead of hanging forever

BUCKET = "sen1floods11"
PREFIXES = {
    "S1Hand": "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "LabelHand": "v1.1/data/flood_events/HandLabeled/LabelHand/",
    "flood_handlabeled": "v1.1/splits/flood_handlabeled/",
}


def list_objects(prefix):
    names, token = [], None
    while True:
        q = {"prefix": prefix, "maxResults": "1000", "fields": "items(name,size),nextPageToken"}
        if token:
            q["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.load(r)
        names += [(it["name"], int(it["size"])) for it in page.get("items", []) if not it["name"].endswith("/")]
        token = page.get("nextPageToken")
        if not token:
            return names


def fetch(name, size, dest):
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return "skip"
    url = f"https://storage.googleapis.com/{BUCKET}/" + urllib.parse.quote(name)
    for attempt in range(4):
        try:
            tmp = dest + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)
            return "ok"
        except Exception as e:  # retry transient network errors
            if attempt == 3:
                return f"failed: {e}"
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = []
    for folder, prefix in PREFIXES.items():
        os.makedirs(os.path.join(args.out, folder), exist_ok=True)
        objs = list_objects(prefix)
        print(f"{folder}: {len(objs)} files, {sum(s for _, s in objs) / 1e6:.0f} MB")
        jobs += [(n, s, os.path.join(args.out, folder, os.path.basename(n))) for n, s in objs]

    done, failed = 0, []
    with ThreadPoolExecutor(args.workers) as ex:
        for (n, _, _), res in zip(jobs, ex.map(lambda j: fetch(*j), jobs)):
            done += 1
            if res.startswith("failed"):
                failed.append((n, res))
            if done % 100 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} files")
    if failed:
        print(f"{len(failed)} files failed; run the script again to retry them:")
        for n, r in failed[:10]:
            print(" ", n, r)
        sys.exit(1)
    print("Download complete.")


if __name__ == "__main__":
    main()
