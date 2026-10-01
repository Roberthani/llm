"""Dev helper: analyse the photo fixture, apply the standard test edits, write crops + diff."""
import json, sys, time
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from trueedit.ingest import load_document
from trueedit.ocr.ppocr import load_models
from trueedit.preprocess import prepare_page
from trueedit.ocr.pipeline import analyze_page
from trueedit.editing import render_page_edits
from trueedit.render import composite

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/s/edits"); out.mkdir(parents=True, exist_ok=True)
fx = sys.argv[2] if len(sys.argv) > 2 else "order_photo.jpg"
m = load_models()
d = load_document((ROOT / "tests/fixtures/out" / fx).read_bytes())
P = prepare_page(d.pages[0].image, m, d.pages[0].is_photo)
A = analyze_page(P.canvas, P.ocr_input, m)
words = [w for r in A["regions"] for w in r["words"]]
def find(t, after=None):
    for r in A["regions"]:
        for w in r["words"]:
            if w["text"] == t and (after is None or after in r["text"]): return w
    raise KeyError(t)
def line_with(t):
    return next(r for r in A["regions"] if r["text"] == t)
edits = []
w = find("Daniel"); edits.append({"id": "A", "bbox": w["bbox"], "text": "Michael", "target_ids": [w["id"]]})
w = find("Kowalski"); edits.append({"id": "B", "bbox": w["bbox"], "text": "Johnson", "target_ids": [w["id"]]})
r = line_with("Organic Potting Soil 50 L"); edits.append({"id": "C", "bbox": r["bbox"], "text": "Premium Garden Compost 40 L", "target_ids": [r["id"]]})
w = find("N/A", "Instructions"); edits.append({"id": "D", "bbox": w["bbox"], "text": "", "target_ids": [w["id"]]})
w = find("N/A", "Email"); edits.append({"id": "E", "bbox": w["bbox"], "text": "dan.k@example.com", "target_ids": [w["id"]]})
t = time.time()
patches = render_page_edits(P.canvas, A, edits, "dev")
print("render secs", round(time.time() - t, 2))
res = composite(P.canvas, patches)
diff = np.any(res != P.canvas, axis=2)
print("changed px", int(diff.sum()), "of", diff.size, f"({diff.sum()/diff.size:.4%})")
for e, p in zip(edits, patches):
    print(e["id"], e["text"], p.bbox, p.info.get("style"), p.info.get("fit", {}).get("score"), p.info["warnings"], p.info.get("condensed"))
    b = p.bbox or e["bbox"]
    pad = 40
    y0, y1, x0, x1 = max(0, b[1]-pad), b[3]+pad, max(0, min(b[0], e["bbox"][0])-pad), max(b[2], e["bbox"][2])+pad
    a = P.canvas[y0:y1, x0:x1]; bb = res[y0:y1, x0:x1]
    dm = (diff[y0:y1, x0:x1]*255).astype(np.uint8); dm = cv2.cvtColor(dm, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out / f"{e['id']}.png"), cv2.resize(np.vstack([a, bb, dm]), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST))
cv2.imwrite(str(out / "full_before.jpg"), P.canvas); cv2.imwrite(str(out / "full_after.jpg"), res)
