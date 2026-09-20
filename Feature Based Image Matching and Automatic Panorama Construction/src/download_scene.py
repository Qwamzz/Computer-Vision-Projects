"""
download_scene.py
=================
Fetches the real photograph that is used as the planar scene.

The panorama pipeline needs a scene that is large enough to cut several
overlapping views out of, and it needs to know the homography relating those
views exactly, otherwise the alignment error cannot be measured in pixels. Those
two requirements pull in opposite directions: a real photograph gives realistic
texture but no ground truth, while a procedurally drawn scene gives ground truth
but unrealistic texture.

This project satisfies both. A real photograph is used as the planar scene, and
the overlapping views are then cut out of it with homographies that are chosen
rather than estimated. The imagery a detector sees is genuine photographic
texture, and the ground truth is still exact.

The photograph is a view of the fishing harbour at Elmina, downloaded from
Wikimedia Commons under a free licence. It suits the task well: boat hulls,
masts, nets and roofs give dense corner and blob structure across the whole
frame, while the sky occupies a broad band that contains almost no features at
all, which is exactly the kind of uneven feature distribution a real panorama
has to cope with.

Nothing here is machine generated imagery.

Usage:
    python src/download_scene.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "panorama-project/1.0 (open source research project)"}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCENE_DIR = os.path.join(ROOT, "data", "scene")
SCENE_FILE = os.path.join(SCENE_DIR, "harbour_scene.jpg")

COMMONS_TITLE = "File:Zicht op vissersboten in de haven - Elmina - 20374853 - RCE.jpg"
RENDER_WIDTH = 2400


def _fetch(url, tries=6):
    """Fetch a URL, backing off politely if Commons rate limits the client."""
    delay = 3.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as fh:
                return fh.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == tries - 1:
                raise
            print("   rate limited, waiting %.0fs" % delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _plain(value):
    """Strip the small amount of HTML that Commons puts in metadata fields."""
    out, skip = [], False
    for ch in value:
        if ch == "<":
            skip = True
        elif ch == ">":
            skip = False
        elif not skip:
            out.append(ch)
    return " ".join("".join(out).split())


def download(force=False):
    os.makedirs(SCENE_DIR, exist_ok=True)
    if os.path.exists(SCENE_FILE) and not force:
        print("scene already present:", SCENE_FILE)
        return SCENE_FILE

    params = {"action": "query", "titles": COMMONS_TITLE, "prop": "imageinfo",
              "iiprop": "url|size|extmetadata", "iiurlwidth": str(RENDER_WIDTH),
              "format": "json"}
    info = json.loads(_fetch(API + "?" + urllib.parse.urlencode(params)))
    page = list(info["query"]["pages"].values())[0]
    ii = page["imageinfo"][0]
    url = (ii.get("thumburl") or ii["url"]).split("?")[0]

    blob = _fetch(url)
    with open(SCENE_FILE, "wb") as fh:
        fh.write(blob)
    print("saved %s  (%d bytes)" % (SCENE_FILE, len(blob)))

    meta = ii.get("extmetadata", {})
    record = {
        "commons_title": COMMONS_TITLE,
        "url": url,
        "licence": _plain(meta.get("LicenseShortName", {}).get("value", "see Commons")),
        "author": _plain(meta.get("Artist", {}).get("value", "see Commons")),
        "credit": _plain(meta.get("Credit", {}).get("value", "")),
        "original_pixels": "%dx%d" % (ii["width"], ii["height"]),
        "bytes": len(blob),
    }
    with open(os.path.join(ROOT, "data", "scene_source.json"), "w",
              encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    with open(os.path.join(ROOT, "data", "SOURCES.md"), "w", encoding="utf-8") as fh:
        fh.write("# Image source\n\n")
        fh.write("The planar scene used by this project is a real photograph ")
        fh.write("downloaded from Wikimedia Commons by `src/download_scene.py`. ")
        fh.write("No image in this project was produced by an image generation ")
        fh.write("model.\n\n")
        fh.write("| Field | Value |\n|---|---|\n")
        fh.write("| Local file | `data/scene/harbour_scene.jpg` |\n")
        fh.write("| Commons title | %s |\n" % COMMONS_TITLE.replace("File:", ""))
        fh.write("| Licence | %s |\n" % record["licence"])
        fh.write("| Author | %s |\n" % record["author"])
        fh.write("| Original size | %s pixels |\n" % record["original_pixels"])
        fh.write("| Source URL | %s |\n" % record["url"])
    print("wrote data/SOURCES.md")
    return SCENE_FILE


if __name__ == "__main__":
    download()
