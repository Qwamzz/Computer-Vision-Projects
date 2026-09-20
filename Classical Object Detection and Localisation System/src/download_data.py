"""
download_data.py
Classical Object Detection and Localisation
Downloads the real source imagery used by this project from Wikimedia Commons.

Nothing here is AI generated. Every file is a real photograph or a real vector
emblem released under a free licence. The script records the licence of each
file in data/SOURCES.md so the provenance of the dataset is auditable.

Usage:
    python src/download_data.py
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "classical-object-detection/1.0 (open source research project)"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# (commons file title, destination sub folder, local name, render width in px)
FILES = [
    # The target object: the University of Ghana coat of arms (public domain).
    ("File:UoG CoA 2017.svg", "template", "ug_logo_master.png", 900),

    # Real background photographs used to build the test scenes.
    ("File:University Of Ghana, Accra, Ghana - panoramio (12).jpg",
     "backgrounds", "bg_ug_campus.jpg", 1400),
    ("File:Balme Library of University of Ghana, Accra, Ghana.jpg",
     "backgrounds", "bg_balme_library.jpg", 1400),
    ("File:Entrance to Mensah Sarbah Hall.jpg",
     "backgrounds", "bg_mensah_sarbah.jpg", 1400),
    ("File:Oxford Street, Korle-Klottey (P1090853).jpg",
     "backgrounds", "bg_accra_street.jpg", 1400),
    ("File:ACCRA, GHANA, SCENE.jpg",
     "backgrounds", "bg_accra_scene.jpg", 1400),
    ("File:Professional woman working at desk with laptop and documents in bright office setting.jpg",
     "backgrounds", "bg_office_desk.jpg", 1400),
    ("File:Brick wall close-up view.jpg",
     "backgrounds", "bg_brick_wall.jpg", 1400),
    ("File:McGill University Library Cybertheque.jpg",
     "backgrounds", "bg_library_interior.jpg", 1400),
    ("File:School of Data Science - University of Virginia - interior - 4.jpg",
     "backgrounds", "bg_lecture_interior.jpg", 1400),

    # Visually similar emblems, used as clutter and as hard negatives.
    ("File:University of Ghana Business School Official Logo.png",
     "distractors", "distractor_ugbs.png", 400),
    ("File:College of Health Sciences (University of Ghana).png",
     "distractors", "distractor_chs.png", 400),
]


def fetch(url, tries=6):
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


def imageinfo(title, width):
    params = {
        "action": "query", "titles": title, "prop": "imageinfo",
        "iiprop": "url|size|extmetadata", "iiurlwidth": str(width), "format": "json",
    }
    data = json.loads(fetch(API + "?" + urllib.parse.urlencode(params)))
    page = list(data["query"]["pages"].values())[0]
    return page["imageinfo"][0]


def plain(value):
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
    rows = []
    for title, folder, name, width in FILES:
        dest_dir = os.path.join(DATA, folder)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, name)
        info = imageinfo(title, width)
        url = info.get("thumburl") or info["url"]
        url = url.split("?")[0]
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            blob = open(dest, "rb").read()
        else:
            blob = fetch(url)
            with open(dest, "wb") as fh:
                fh.write(blob)
            time.sleep(1.5)
        meta = info.get("extmetadata", {})
        rows.append({
            "file": os.path.join(folder, name).replace("\\", "/"),
            "commons_title": title,
            "licence": plain(meta.get("LicenseShortName", {}).get("value", "see Commons")),
            "author": plain(meta.get("Artist", {}).get("value", "see Commons")),
            "url": url,
            "bytes": len(blob),
        })
        print("saved %-42s %8d bytes" % (name, len(blob)))

    with open(os.path.join(DATA, "SOURCES.md"), "w", encoding="utf-8") as fh:
        fh.write("# Image sources\n\n")
        fh.write("All imagery below was downloaded from Wikimedia Commons by ")
        fh.write("`src/download_data.py`. No image in this project was generated ")
        fh.write("by an image synthesis model.\n\n")
        fh.write("| Local file | Commons title | Licence | Author |\n")
        fh.write("|---|---|---|---|\n")
        for r in rows:
            fh.write("| `%s` | %s | %s | %s |\n" %
                     (r["file"], r["commons_title"], r["licence"], r["author"][:70]))
    with open(os.path.join(DATA, "sources.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print("wrote data/SOURCES.md")


if __name__ == "__main__":
    main()
