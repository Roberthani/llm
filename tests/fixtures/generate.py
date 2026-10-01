"""Generate realistic test documents for TrueEdit.

Produces (in tests/fixtures/out/):
  order_clean.png        flat render of a retail order confirmation (~240 DPI)
  order_clean.json       ground truth: every word with its pixel box + semantic field
  order_photo.jpg        simulated phone photo: perspective, fold, shadow, uneven light,
                         blur, sensor noise, JPEG compression
  order_photo.json       ground-truth mapping clean -> photo (homography + fold params)
  order_photo_rot90.jpg  same photo stored rotated 90 degrees (no EXIF hint)
  order_photo_exif.jpg   photo stored rotated, with an EXIF orientation tag
  order.pdf              single-page vector PDF (Chromium print)
  order_multipage.pdf    3-page vector PDF
  order_scan.pdf         image-only (scanned style) PDF
  order_lowres.jpg       low-resolution photo (~800 px tall)
  order_huge.png         very large image (12000 px tall)
  order.heic             HEIC version of the photo (if pillow-heif can encode)
  blank.png              empty page
  corrupt.jpg            truncated / garbage JPEG
  notes.txt              unsupported file type

Ground truth is extracted from the browser DOM (Range.getClientRects per word), so no
coordinates are hard-coded anywhere.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "out"
CHROME = os.environ.get(
    "TRUEEDIT_CHROME", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
)
SCALE = 2.5  # device scale factor -> 816 css px * 2.5 = 2040 px wide (240 DPI)


def barcode_svg(value: str) -> str:
    import barcode
    from barcode.writer import SVGWriter

    code = barcode.get("code128", value, writer=SVGWriter())
    svg = code.render(
        writer_options={"module_width": 0.30, "module_height": 11, "write_text": False, "quiet_zone": 1}
    ).decode()
    return svg[svg.index("<svg"):]


LOGO_SVG = """
<svg width="58" height="58" viewBox="0 0 58 58" xmlns="http://www.w3.org/2000/svg">
  <circle cx="29" cy="29" r="27" fill="#1f6b4a"/>
  <path d="M29 9 C41 19 43 33 29 49 C15 33 17 19 29 9 Z" fill="#9fd36b"/>
  <path d="M29 14 L29 46" stroke="#1f6b4a" stroke-width="2.4"/>
  <path d="M29 27 L37 21 M29 34 L21 28 M29 40 L36 35" stroke="#1f6b4a" stroke-width="2"/>
</svg>
"""

ITEMS = [
    ("1", "40118-22", "Cedar Raised Garden Bed 4x8 ft", "2", "189.99"),
    ("2", "51007-04", "Organic Potting Soil 50 L", "6", "14.49"),
    ("3", "30562-11", "Stainless Steel Hand Trowel", "1", "24.95"),
    ("4", "77310-08", "Drip Irrigation Starter Kit", "1", "69.00"),
    ("5", "12094-31", "Heirloom Tomato Seeds (pack)", "4", "3.79"),
    ("6", "65021-17", "Galvanized Watering Can 9 L", "1", "42.50"),
]


def money(v: float) -> str:
    return f"{v:,.2f}"


def order_html(page_no: int = 1, pages: int = 1, first="Daniel", last="Kowalski") -> str:
    rows = []
    sub = 0.0
    for ln, sku, desc, qty, price in ITEMS:
        amt = int(qty) * float(price)
        sub += amt
        rows.append(
            f'<tr><td class="c" data-field="line">{ln}</td><td data-field="sku">{sku}</td>'
            f'<td data-field="item_desc">{desc}</td><td class="r" data-field="qty">{qty}</td>'
            f'<td class="r" data-field="price">{price}</td><td class="r" data-field="amount">{money(amt)}</td></tr>'
        )
    disc = round(sub * 0.05, 2)
    tax = round((sub - disc) * 0.13, 2)
    total = sub - disc + tax
    deposit = 200.0
    extra = ""
    if page_no > 1:
        extra = f"""<div class="sec"><div class="h2" data-field="section">Page {page_no} - Terms and Conditions</div>
        <p class="fine" data-field="terms">All sales of special-order items are final. Returns of regular stock are accepted
        within 30 days with original receipt. Delivery windows are estimates and may change due to weather or supplier
        delays. Assembly services are subject to availability. Prices include applicable eco fees. Reference
        order NW-2026-118734 for all inquiries.</p></div>"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
    @page {{ size: 8.5in 11in; margin: 0; }}
    html,body {{ margin:0; padding:0; background:#fff; }}
    body {{ width:816px; height:1056px; box-sizing:border-box; padding:44px 48px; font-family:'Liberation Sans',Arial,sans-serif; color:#1b1b1b; font-size:11px; position:relative; }}
    .top {{ display:flex; justify-content:space-between; align-items:flex-start; }}
    .brand {{ display:flex; align-items:center; gap:10px; }}
    .wordmark {{ font-family:'Liberation Serif',serif; font-size:24px; font-weight:bold; color:#1f6b4a; letter-spacing:1px; }}
    .tag {{ font-size:9px; color:#4d7d63; letter-spacing:2px; }}
    .title {{ text-align:right; }}
    .title .t {{ font-size:20px; font-weight:bold; letter-spacing:0.5px; }}
    .meta td {{ padding:1px 0 1px 10px; font-size:11px; }}
    .meta td.k {{ color:#555; text-align:right; }}
    .meta td.v {{ font-weight:bold; text-align:right; }}
    .rule {{ border-top:2px solid #1f6b4a; margin:14px 0 12px; }}
    .cols {{ display:flex; gap:24px; }}
    .box {{ flex:1; border:1px solid #9a9a9a; padding:8px 10px; min-height:92px; }}
    .h2 {{ font-size:10px; font-weight:bold; color:#1f6b4a; letter-spacing:1px; margin-bottom:4px; }}
    .line {{ line-height:15px; }}
    .lbl {{ color:#555; }}
    table.items {{ width:100%; border-collapse:collapse; margin-top:16px; }}
    table.items th {{ background:#e8efe9; font-size:10px; text-align:left; border:1px solid #7d7d7d; padding:5px 6px; }}
    table.items td {{ border:1px solid #7d7d7d; padding:5px 6px; font-size:11px; }}
    table.items .r {{ text-align:right; }} table.items .c {{ text-align:center; }}
    .lower {{ display:flex; justify-content:space-between; margin-top:14px; }}
    .notes {{ width:52%; font-size:10.5px; }}
    table.tot {{ border-collapse:collapse; width:40%; }}
    table.tot td {{ padding:3px 6px; font-size:11px; border-bottom:1px solid #d0d0d0; }}
    table.tot td.r {{ text-align:right; }}
    table.tot tr.grand td {{ font-weight:bold; font-size:13px; border-top:2px solid #1b1b1b; border-bottom:2px solid #1b1b1b; }}
    .bc {{ position:absolute; left:48px; bottom:78px; text-align:center; }}
    .bc .num {{ font-family:'Liberation Mono',monospace; font-size:11px; letter-spacing:2px; }}
    .fine {{ font-size:8.5px; color:#444; line-height:12px; }}
    .foot {{ position:absolute; left:300px; right:48px; bottom:70px; }}
    .sec {{ margin-top:18px; }}
    </style></head><body>
    <div class="top">
      <div class="brand" data-graphic="logo">{LOGO_SVG}<div><div class="wordmark" data-field="logo_text">NORTHWIND</div><div class="tag" data-field="logo_text">HOME &amp; GARDEN</div></div></div>
      <div class="title"><div class="t" data-field="doc_title">ORDER CONFIRMATION</div>
        <table class="meta" align="right">
          <tr><td class="k">Order No.</td><td class="v" data-field="order_no">NW-2026-118734</td></tr>
          <tr><td class="k">Order Date</td><td class="v" data-field="date">2026-09-14</td></tr>
          <tr><td class="k">Reference</td><td class="v" data-field="reference">PO-55821-CA</td></tr>
          <tr><td class="k">Store</td><td class="v" data-field="store">#0482 Mississauga</td></tr>
        </table></div>
    </div>
    <div class="rule"></div>
    <div class="cols">
      <div class="box"><div class="h2">SOLD TO</div>
        <div class="line"><b data-field="first_name">{first}</b> <b data-field="last_name">{last}</b></div>
        <div class="line" data-field="address">1482 Lakeshore Rd W</div>
        <div class="line" data-field="address">Mississauga, ON L5H 1G2</div>
        <div class="line"><span class="lbl">Phone:</span> <span data-field="phone">(905) 555-0148</span></div>
        <div class="line"><span class="lbl">Email:</span> <span data-field="email">N/A</span></div>
      </div>
      <div class="box"><div class="h2">DELIVERY</div>
        <div class="line"><span class="lbl">Method:</span> <span data-field="method">Curbside Delivery</span></div>
        <div class="line"><span class="lbl">Scheduled:</span> <span data-field="scheduled">2026-09-21, 9am-1pm</span></div>
        <div class="line"><span class="lbl">Instructions:</span> <span data-field="instructions">N/A</span></div>
        <div class="line"><span class="lbl">Salesperson:</span> <span data-field="salesperson">M. Tremblay</span></div>
        <div class="line"><span class="lbl">Customer ID:</span> <span data-field="customer_id">C-7720394</span></div>
      </div>
    </div>
    <table class="items">
      <tr><th style="width:34px">LINE</th><th style="width:78px">SKU</th><th>DESCRIPTION</th><th style="width:40px">QTY</th><th style="width:76px">UNIT PRICE</th><th style="width:80px">AMOUNT</th></tr>
      {''.join(rows)}
    </table>
    <div class="lower">
      <div class="notes"><div class="h2">NOTES</div>
        <div class="line"><span class="lbl">Gift message:</span> <span data-field="gift">N/A</span></div>
        <div class="line"><span class="lbl">Payment:</span> <span data-field="payment">VISA ending 4417</span></div>
        <div class="line"><span class="lbl">Deposit paid:</span> <span data-field="deposit">{money(deposit)}</span></div>
      </div>
      <table class="tot">
        <tr><td>Subtotal</td><td class="r" data-field="subtotal">{money(sub)}</td></tr>
        <tr><td>Discount (5%)</td><td class="r" data-field="discount">-{money(disc)}</td></tr>
        <tr><td>HST 13%</td><td class="r" data-field="tax">{money(tax)}</td></tr>
        <tr class="grand"><td>TOTAL</td><td class="r" data-field="total">{money(total)}</td></tr>
        <tr><td>Balance Due</td><td class="r" data-field="balance">{money(total - deposit)}</td></tr>
      </table>
    </div>
    {extra}
    <div class="bc" data-graphic="barcode">{barcode_svg('NW2026118734')}<div class="num" data-field="barcode_text">NW2026118734</div></div>
    <div class="foot"><p class="fine" data-field="fine">Thank you for shopping with Northwind Home &amp; Garden. Questions about this order? Call 1-800-555-0199
    or visit any store. Keep this confirmation for warranty and returns. Page {page_no} of {pages}.</p></div>
    </body></html>"""


WORD_JS = r"""
() => {
  const out = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const text = node.textContent;
    if (!text.trim()) continue;
    const parent = node.parentElement;
    if (parent.closest('svg')) continue;
    const fieldEl = parent.closest('[data-field]');
    const field = fieldEl ? fieldEl.getAttribute('data-field') : null;
    const style = getComputedStyle(parent);
    const re = /\S+/g; let m;
    while ((m = re.exec(text))) {
      const r = document.createRange();
      r.setStart(node, m.index); r.setEnd(node, m.index + m[0].length);
      const rects = r.getClientRects();
      if (!rects.length) continue;
      let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
      for (const q of rects) { x0=Math.min(x0,q.left); y0=Math.min(y0,q.top); x1=Math.max(x1,q.right); y1=Math.max(y1,q.bottom); }
      out.push({text: m[0], box:[x0,y0,x1,y1], field, font_size: parseFloat(style.fontSize), bold: parseInt(style.fontWeight) >= 600 || parent.tagName==='B'});
    }
  }
  const graphics = [];
  for (const el of document.querySelectorAll('[data-graphic]')) {
    const kind = el.getAttribute('data-graphic');
    const svg = el.querySelector('svg');
    const r = svg.getBoundingClientRect();
    graphics.push({kind, box:[r.left, r.top, r.right, r.bottom]});
  }
  const tables = [];
  for (const t of document.querySelectorAll('table.items')) {
    const r = t.getBoundingClientRect();
    tables.push({box:[r.left,r.top,r.right,r.bottom], rows: t.rows.length, cols: t.rows[0].cells.length});
  }
  return {words: out, graphics, tables};
}
"""


def render_html(html: str, png_path: Path | None, pdf_path: Path | None):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME)
        page = b.new_page(viewport={"width": 816, "height": 1056}, device_scale_factor=SCALE)
        page.set_content(html)
        page.wait_for_timeout(150)
        gt = page.evaluate(WORD_JS)
        if png_path:
            page.screenshot(path=str(png_path), full_page=False)
        if pdf_path:
            page.pdf(path=str(pdf_path), width="8.5in", height="11in", print_background=True)
        b.close()
    for w in gt["words"]:
        w["box"] = [round(v * SCALE, 2) for v in w["box"]]
    for g in gt["graphics"] + gt["tables"]:
        g["box"] = [round(v * SCALE, 2) for v in g["box"]]
    return gt


def render_pdf_pages(htmls: list[str], pdf_path: Path):
    from playwright.sync_api import sync_playwright

    import fitz

    out = fitz.open()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME)
        page = b.new_page(viewport={"width": 816, "height": 1056})
        for h in htmls:
            page.set_content(h)
            data = page.pdf(width="8.5in", height="11in", print_background=True)
            src = fitz.open("pdf", data)
            out.insert_pdf(src, from_page=0, to_page=0)
        b.close()
    out.save(str(pdf_path))


# ---------------------------------------------------------------- photo simulation

def fold_displace(x: np.ndarray, y: np.ndarray, p: dict):
    """Mild paper fold: points below the fold line shift slightly (non-linear bend)."""
    fy = p["fold_y"]
    t = 1.0 / (1.0 + np.exp(np.clip(-(y - fy) / p["fold_soft"], -60, 60)))  # 0 above, 1 below
    dx = p["fold_dx"] * t * (x / p["w"])
    dy = p["fold_dy"] * t * np.sin(np.pi * x / p["w"])
    return x + dx, y + dy


def clean_to_photo(pts: np.ndarray, meta: dict) -> np.ndarray:
    """Map clean-page pixel coords (N,2) to photo pixel coords."""
    x, y = fold_displace(pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64), meta["fold"])
    H = np.array(meta["H"])
    v = np.stack([x, y, np.ones_like(x)], 1) @ H.T
    return v[:, :2] / v[:, 2:3]


def simulate_photo(clean: np.ndarray, seed: int = 7, out_w: int = 3024, out_h: int = 4032):
    rng = np.random.default_rng(seed)
    h, w = clean.shape[:2]
    fold = {"fold_y": h * 0.47, "fold_soft": 6.0, "fold_dx": 3.0, "fold_dy": 5.0, "w": float(w)}

    # 1) fold: inverse map via fixed-point iteration (displacement is small)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    sx, sy = gx.copy(), gy.copy()
    for _ in range(4):
        fx, fy_ = fold_displace(sx, sy, fold)
        sx = sx - (fx - gx)
        sy = sy - (fy_ - gy)
    folded = cv2.remap(clean, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # 2) perspective onto a phone-sized frame
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[452, 520], [2638, 430], [2790, 3560], [300, 3655]])
    dst += rng.normal(0, 6, dst.shape).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    bg = np.zeros((out_h, out_w, 3), np.float32)
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    # wooden desk-ish background
    grain = (np.sin(xx / 37.0 + np.sin(yy / 190.0) * 3) * 0.5 + 0.5)
    bg[..., 0] = 60 + 25 * grain
    bg[..., 1] = 85 + 30 * grain
    bg[..., 2] = 120 + 40 * grain
    bg += rng.normal(0, 6, bg.shape)
    warped = cv2.warpPerspective(folded, H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    # soft drop shadow of the paper on the desk
    sh = cv2.GaussianBlur(cv2.warpAffine(mask, np.float32([[1, 0, 18], [0, 1, 24]]), (out_w, out_h)), (0, 0), 25)
    bg *= (1 - 0.45 * (sh.astype(np.float32) / 255.0))[..., None]
    img = warped.astype(np.float32) * m + bg * (1 - m)

    # 3) illumination: gradient + soft shadow (hand/phone) + fold shading
    light = 0.78 + 0.27 * (xx / out_w) * 0.6 + 0.27 * (1 - yy / out_h) * 0.4
    blob = np.exp(-(((xx - 2500) / 700.0) ** 2 + ((yy - 3300) / 600.0) ** 2))
    light *= 1 - 0.38 * blob
    # fold line shading: compute clean-y for each photo pixel approximately via inverse H
    Hi = np.linalg.inv(H)
    den = Hi[2, 0] * xx + Hi[2, 1] * yy + Hi[2, 2]
    cy = (Hi[1, 0] * xx + Hi[1, 1] * yy + Hi[1, 2]) / den
    d = (cy - fold["fold_y"]) / 10.0
    fold_shade = 1 - 0.13 * np.exp(-d ** 2) + 0.05 * np.exp(-((d + 1.6) ** 2))
    fold_shade = np.where(cy > fold["fold_y"], fold_shade - 0.04, fold_shade)
    light = light * (fold_shade * m[..., 0] + (1 - m[..., 0]))
    img *= light[..., None]
    img *= np.array([0.93, 0.98, 1.03], np.float32)  # warm white balance (BGR)

    # 4) optics + sensor
    img = cv2.GaussianBlur(img, (0, 0), 1.1)
    img += rng.normal(0, 3.5, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    meta = {"H": H.tolist(), "fold": fold, "clean_size": [w, h], "photo_size": [out_w, out_h]}
    return img, meta


def write_jpeg(path: Path, img: np.ndarray, q: int = 88, exif_orientation: int | None = None):
    from PIL import Image

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    kw = {"quality": q}
    if exif_orientation:
        ex = Image.Exif()
        ex[0x0112] = exif_orientation
        kw["exif"] = ex.tobytes()
    pil.save(path, "JPEG", **kw)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    html = order_html()
    gt = render_html(html, OUT / "order_clean.png", OUT / "order.pdf")
    (OUT / "order_clean.json").write_text(json.dumps(gt, indent=1))
    clean = cv2.imread(str(OUT / "order_clean.png"))
    print("clean", clean.shape, "words", len(gt["words"]))

    photo, meta = simulate_photo(clean)
    write_jpeg(OUT / "order_photo.jpg", photo, 88)
    (OUT / "order_photo.json").write_text(json.dumps(meta))
    # rotated copies: pixels rotated 90 CW without EXIF (needs content-based orientation)
    write_jpeg(OUT / "order_photo_rot90.jpg", cv2.rotate(photo, cv2.ROTATE_90_CLOCKWISE), 88)
    # stored rotated 90 CCW with EXIF Orientation=6 ("rotate 90 CW to display")
    write_jpeg(OUT / "order_photo_exif.jpg", cv2.rotate(photo, cv2.ROTATE_90_COUNTERCLOCKWISE), 88, exif_orientation=6)
    small = cv2.resize(photo, (photo.shape[1] * 800 // photo.shape[0], 800), interpolation=cv2.INTER_AREA)
    write_jpeg(OUT / "order_lowres.jpg", small, 80)
    huge = cv2.resize(clean, (clean.shape[1] * 12000 // clean.shape[0], 12000), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(OUT / "order_huge.png"), huge, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(str(OUT / "blank.png"), np.full((2200, 1700, 3), 252, np.uint8))
    good = (OUT / "order_photo.jpg").read_bytes()
    (OUT / "corrupt.jpg").write_bytes(good[:2000] + os.urandom(3000))
    (OUT / "notes.txt").write_text("this is not a document image\n")

    render_pdf_pages([order_html(i, 3) for i in (1, 2, 3)], OUT / "order_multipage.pdf")

    import fitz

    doc = fitz.open()
    pg = doc.new_page(width=612, height=792)
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 85])
    pg.insert_image(pg.rect, stream=buf.tobytes())
    doc.save(str(OUT / "order_scan.pdf"))

    try:
        import pillow_heif
        from PIL import Image

        pillow_heif.register_heif_opener()
        Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)).save(OUT / "order.heic", quality=90)
    except Exception as e:  # pragma: no cover
        print("HEIC encode unavailable:", e, file=sys.stderr)
    print("fixtures written to", OUT)


if __name__ == "__main__":
    main()
