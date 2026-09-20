"""
download_data.py
Automated road and lane boundary detection and scene segmentation.

Downloads the road scene dataset from Wikimedia Commons and records the
licence and author of every file in data/SOURCES.md.

All twelve images are real photographs of real roads, taken by other people
and published under free licences. Nothing in this project was produced by an
image generation model. The set was chosen to cover the six imaging conditions
the system is tested against, two photographs each:

    good daylight      clear markings, even lighting, the easy case
    shadows            dappled shade from roadside trees across the carriageway
    poor illumination  haze and heavy overcast, low contrast
    curved road        the straight line model is wrong by construction
    worn markings      unsealed laterite roads with no markings at all
    occlusion          vehicles and people standing on the carriageway

Images are stored at a working width of 1280 pixels with the aspect ratio
preserved, which is large enough to resolve lane markings and small enough to
keep the parameter sweeps quick.

Usage:
    python src/download_data.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "road-lane-project/1.0 (open source research project)"}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "data", "images")

WORK_WIDTH = 1280

# (local name, Commons title, condition, one line note on what it tests)
DATASET = [
    ("daylight_01.jpg", "File:Babouin (120916011).jpeg", "daylight",
     "Straight sealed road, continuous white edge lines, low sun."),
    ("daylight_02.jpg", "File:Long Smooth Road.jpg", "daylight",
     "Two lane road with a painted centre line and clear edges."),

    ("shadow_01.jpg", "File:African Road.jpg", "shadows",
     "Avenue of trees casting hard dappled shadow bands across the road."),
    ("shadow_02.jpg", "File:Road in Shai Hills Forest Reserve.jpg", "shadows",
     "Unsealed road under trees, shadow contrast rivals the road edge."),

    ("lowlight_01.jpg", "File:On the road from Obuasi to Cape coast - panoramio.jpg",
     "poor_illumination", "Haze and flat overcast light, very low edge contrast."),
    ("lowlight_02.jpg", "File:Road from Kumasi to Obuasi - panoramio.jpg",
     "poor_illumination", "Overcast, wet looking surface, weak markings."),

    ("curve_01.jpg", "File:Roads in Ghana.jpg", "curve",
     "Strong single bend with clear markings, defeats the straight line model."),
    ("curve_02.jpg", "File:Nice Road Curve.jpg", "curve",
     "Sweeping curve seen from the inside of the bend."),

    ("worn_01.jpg", "File:Mim Cashew plantation.jpg", "worn_markings",
     "Unsealed laterite road, no markings at all, boundary is a colour change."),
    ("worn_02.jpg", "File:Mole 2 (120915995).jpeg", "worn_markings",
     "Straight laterite road, boundary is a colour change against the verge."),

    ("occlusion_01.jpg", "File:Female Street Sellers Accra 12.jpg", "occlusion",
     "People and vehicles standing on the carriageway, breaking the edges."),
    ("occlusion_02.jpg", "File:Farm Tractor.jpg", "occlusion",
     "A tractor and a pickup occupying the carriageway ahead."),
]


def _fetch(url, tries=6):
    """Fetch a URL, backing off politely if Commons rate limits the client."""
    delay = 4.0
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


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    rows = []
    for name, title, condition, note in DATASET:
        dest = os.path.join(IMG_DIR, name)
        params = {"action": "query", "titles": title, "prop": "imageinfo",
                  "iiprop": "url|size|extmetadata", "iiurlwidth": str(WORK_WIDTH),
                  "format": "json"}
        info = json.loads(_fetch(API + "?" + urllib.parse.urlencode(params)))
        page = list(info["query"]["pages"].values())[0]
        ii = page["imageinfo"][0]
        url = (ii.get("thumburl") or ii["url"]).split("?")[0]

        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            blob = open(dest, "rb").read()
        else:
            blob = _fetch(url)
            with open(dest, "wb") as fh:
                fh.write(blob)
            time.sleep(1.2)

        meta = ii.get("extmetadata", {})
        rows.append({
            "file": name, "condition": condition, "note": note,
            "commons_title": title.replace("File:", ""),
            "licence": _plain(meta.get("LicenseShortName", {}).get("value", "see Commons")),
            "author": _plain(meta.get("Artist", {}).get("value", "see Commons")),
            "url": url, "original_pixels": "%dx%d" % (ii["width"], ii["height"]),
            "bytes": len(blob),
        })
        print("saved %-18s %-18s %7d bytes" % (name, condition, len(blob)))

    with open(os.path.join(ROOT, "data", "sources.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    with open(os.path.join(ROOT, "data", "SOURCES.md"), "w", encoding="utf-8") as fh:
        fh.write("# Image sources\n\n")
        fh.write("Every image below is a real photograph downloaded from Wikimedia ")
        fh.write("Commons by `src/download_data.py`. No image in this project was ")
        fh.write("produced by an image generation model.\n\n")
        fh.write("| File | Condition | Commons title | Licence | Author |\n")
        fh.write("|---|---|---|---|---|\n")
        for r in rows:
            fh.write("| `%s` | %s | %s | %s | %s |\n" %
                     (r["file"], r["condition"], r["commons_title"][:52],
                      r["licence"], r["author"][:34]))
    print("wrote data/SOURCES.md")


if __name__ == "__main__":
    main()
