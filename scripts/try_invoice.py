"""Dev helper: standard edits on the invoice fixture (font not shipped with the app)."""
import sys, cv2, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from trueedit.ingest import load_document
from trueedit.ocr.ppocr import load_models
from trueedit.preprocess import prepare_page
from trueedit.ocr.pipeline import analyze_page
from trueedit.editing import render_page_edits
from trueedit.render import composite
m = load_models(); d = load_document((ROOT / 'tests/fixtures/out/invoice_photo.jpg').read_bytes())
P = prepare_page(d.pages[0].image, m, True); A = analyze_page(P.canvas, P.ocr_input, m)
def W(t): return next(w for r in A['regions'] for w in r['words'] if w['text'] == t)
def Lr(t): return next(r for r in A['regions'] if r['text'] == t)
eds = []
for i, (t, n) in enumerate([('Priya', 'Margaret'), ('Raman', 'Okafor'), ('N/A', '90 days'), ('MAP-77215', 'MAP-77299')]):
    w = W(t); eds.append({'id': str(i), 'bbox': w['bbox'], 'text': n, 'target_ids': [w['id']]})
r = Lr('Synthetic Oil Filter'); eds.append({'id': 'L', 'bbox': r['bbox'], 'text': 'Cabin Air Filter', 'target_ids': [r['id']]})
ps = render_page_edits(P.canvas, A, eds, 'inv'); res = composite(P.canvas, ps)
out = []
for e, p in zip(eds, ps):
    print(e['text'], p.info.get('style'), p.info.get('moved'), p.info['warnings'])
    b = p.bbox; y0, y1, x0, x1 = b[1] - 30, b[3] + 30, max(0, b[0] - 150), b[2] + 60
    pair = np.vstack([P.canvas[y0:y1, x0:x1], res[y0:y1, x0:x1]]); out.append(cv2.copyMakeBorder(pair, 0, 6, 0, max(0, 900 - pair.shape[1]), cv2.BORDER_CONSTANT, value=(255, 0, 255))[:, :900])
cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else '/tmp/claude-0/s/inv_edits.png', np.vstack(out))
