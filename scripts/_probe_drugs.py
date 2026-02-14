"""Quick probe: which drugs have the most images in ePillID?"""
import urllib.request, json, os, time, urllib.parse
from collections import defaultdict

seg_dir = "data/ePillID_data/classification_data/segmented_nih_pills_224"
seg_files = os.listdir(seg_dir)
seg_prefixes = defaultdict(list)
for f in seg_files:
    p = f.split("_", 1)[0]
    seg_prefixes[p].append(f)

candidates = [
    "metoprolol tartrate", "metoprolol succinate",
    "losartan potassium", "levothyroxine sodium",
    "simvastatin", "sertraline hydrochloride",
    "tramadol hydrochloride", "trazodone hydrochloride",
    "alprazolam", "clopidogrel bisulfate",
    "furosemide", "pantoprazole sodium",
    "prednisone", "ciprofloxacin hydrochloride",
    "hydrochlorothiazide", "azithromycin",
    "warfarin sodium", "carvedilol",
]

for drug in candidates:
    q = urllib.parse.quote(f'openfda.generic_name:"{drug}"')
    url = f"https://api.fda.gov/drug/label.json?search={q}&limit=100"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PillCare/1.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        codes = set()
        for r in data.get("results", []):
            for ndc in r.get("openfda", {}).get("product_ndc", []):
                parts = ndc.split("-")
                if len(parts) >= 2:
                    codes.add(f"{parts[0].zfill(4)}-{parts[1].zfill(4)}")
                    codes.add(ndc.rsplit('-', 1)[0] if len(parts) > 2 else ndc)
        matched = sum(1 for c in codes if c in seg_prefixes)
        img_count = sum(len(seg_prefixes[c]) for c in codes if c in seg_prefixes)
        print(f"{drug:40s}  codes={len(codes):3d}  matches={matched:3d}  images={img_count:4d}")
    except Exception as e:
        print(f"{drug:40s}  ERROR: {e}")
    time.sleep(0.3)
