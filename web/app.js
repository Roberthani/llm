// TrueEdit OCR — web client (no build step).
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const api = async (path, opts = {}) => {
  const o = { ...opts, headers: { ...(opts.headers || {}) } };
  if (o.json !== undefined) { o.body = JSON.stringify(o.json); o.headers["Content-Type"] = "application/json"; delete o.json; }
  let r;
  try { r = await fetch(path, o); } catch (e) { throw new AppError("network", "Can't reach the server. Check your connection and try again."); }
  if (!r.ok) {
    let body = null; try { body = await r.json(); } catch (_) {}
    const er = body && body.error;
    throw new AppError(er ? er.code : `http_${r.status}`, er ? er.message : `Request failed (${r.status})`, er && er.hint);
  }
  return r;
};
class AppError extends Error { constructor(code, msg, hint) { super(msg); this.code = code; this.hint = hint; } }

const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch (_) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (_) {} },
};

const S = {
  pid: null, meta: null, page: 0, A: {}, doc: { edits: [], review: {}, regions: [] },
  undo: [], redo: [], patches: {}, pos: {}, info: {}, sel: null, mode: "select",
  view: { z: 1, x: 0, y: 0 }, renderSeq: 0, saveTimer: null, saving: null, find: { hits: [], i: -1 }, imgSize: {},
};
window.__trueedit = S; // handy for debugging / automated tests

// ------------------------------------------------------------------ screens
function show(id) { for (const s of $$(".screen")) s.hidden = s.id !== id; }
function toast(msg, kind = "", ms = 3200) {
  const t = document.createElement("div"); t.className = `toast ${kind}`; t.textContent = msg;
  $("#toasts").appendChild(t); setTimeout(() => t.remove(), ms);
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ------------------------------------------------------------------ upload
async function initUpload() {
  const drop = $("#drop");
  $("#file").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
  $("#camera").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
  $("#projfile").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0], true));
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file").click(); } });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) upload(f, /\.trueedit$/i.test(f.name)); });
  renderRecent();
  api("/api/samples").then((r) => r.json()).then(({ samples }) => {
    if (!samples.length) return;
    const b = $("#btn-sample"); b.hidden = false;
    b.onclick = async () => {
      const name = samples.find((s) => s.startsWith("order")) || samples[0];
      const blob = await (await api(`/api/samples/${name}`)).blob();
      upload(new File([blob], name, { type: "image/jpeg" }));
    };
  }).catch(() => {});
  try {
    const h = await (await api("/api/health")).json();
    $("#maxmb").textContent = h.max_upload_mb;
    $("#engine-line").textContent = h.ocr.available
      ? `OCR: ${h.ocr.primary}${h.ocr.secondary && h.ocr.secondary.length ? " + " + h.ocr.secondary.join(", ") + " for verification" : ""}.`
      : `OCR engine unavailable (${h.ocr.error || "unknown"}). You can still edit by drawing regions manually.`;
  } catch (e) { $("#engine-line").textContent = e.message; }
}

function renderRecent() {
  const rec = store.get("trueedit.recent", []);
  const box = $("#recent"); const ul = $("ul", box); ul.innerHTML = "";
  box.hidden = !rec.length;
  for (const r of rec.slice(0, 6)) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${esc(r.name)}</span><span class="muted small">${new Date(r.t).toLocaleString()}</span>`;
    li.onclick = () => openProject(r.id);
    ul.appendChild(li);
  }
}
function remember(id, name) {
  const rec = store.get("trueedit.recent", []).filter((r) => r.id !== id);
  rec.unshift({ id, name, t: Date.now() }); store.set("trueedit.recent", rec.slice(0, 10));
}

async function upload(file, isProject = false) {
  const errBox = $("#up-error"); errBox.hidden = true;
  show("screen-analyze"); setProgress(0.02, "Uploading…"); $("#an-error").hidden = true; $("#an-actions").hidden = true;
  $("#an-title").textContent = isProject ? "Opening project…" : "Analyzing document…";
  const fd = new FormData(); fd.append("file", file, file.name);
  try {
    const r = await api(isProject ? "/api/projects/import" : "/api/projects", { method: "POST", body: fd });
    const m = await r.json();
    remember(m.id, m.filename);
    location.hash = `p=${m.id}`;
    await waitForAnalysis(m.id);
  } catch (e) {
    show("screen-upload");
    errBox.hidden = false;
    errBox.innerHTML = `<b>${esc(e.message)}</b>${e.hint ? `<span>${esc(e.hint)}</span>` : ""}`;
  }
}

function setProgress(f, stage) {
  $("#an-bar").style.width = `${Math.max(2, Math.round(f * 100))}%`;
  $("#an-stage").textContent = stage;
}

async function waitForAnalysis(pid) {
  show("screen-analyze");
  for (;;) {
    let st;
    try { st = await (await api(`/api/projects/${pid}/status`)).json(); }
    catch (e) { setProgress(0, e.message); await sleep(2000); continue; }
    setProgress(st.progress || 0, st.stage || "");
    if (st.status === "ready") { await openProject(pid); return; }
    if (st.status === "error") { showAnalysisError(pid, st.error || {}); return; }
    await sleep(600);
  }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function showAnalysisError(pid, er) {
  $("#an-title").textContent = "Analysis didn't finish";
  const box = $("#an-error"); box.hidden = false; box.textContent = er.message || "Analysis failed.";
  const acts = $("#an-actions"); acts.innerHTML = ""; acts.hidden = false;
  for (const r of er.recovery || [{ action: "retry", label: "Retry" }]) {
    const b = document.createElement("button"); b.className = "btn" + (r.action === "retry" || r.action === "retry_fast" ? " primary" : "");
    b.textContent = r.label;
    b.onclick = async () => {
      const mode = r.action === "retry_fast" ? "fast" : r.action === "manual" ? "manual" : "full";
      box.hidden = true; acts.hidden = true; $("#an-title").textContent = "Analyzing document…";
      await api(`/api/projects/${pid}/analyze`, { method: "POST", json: { mode } });
      waitForAnalysis(pid);
    };
    acts.appendChild(b);
  }
  const home = document.createElement("button"); home.className = "btn ghost"; home.textContent = "Upload a different file";
  home.onclick = goHome; acts.appendChild(home);
}

function goHome() {
  flushSave();
  location.hash = ""; S.pid = null; show("screen-upload"); renderRecent();
}

// ------------------------------------------------------------------ project
async function openProject(pid) {
  show("screen-analyze"); setProgress(0.95, "Loading editor…");
  let m;
  try { m = await (await api(`/api/projects/${pid}`)).json(); }
  catch (e) { show("screen-upload"); const b = $("#up-error"); b.hidden = false; b.textContent = e.code === "not_found" ? "That project no longer exists." : e.message; location.hash = ""; return; }
  if (m.status !== "ready") { if (m.status === "error") return showAnalysisError(pid, m.error || {}); return waitForAnalysis(pid); }
  Object.assign(S, { pid, meta: m, page: 0, A: {}, patches: {}, pos: {}, info: {}, sel: null, undo: [], redo: [] });
  remember(pid, m.filename);
  location.hash = `p=${pid}`;
  const doc = await (await api(`/api/projects/${pid}/edits`)).json();
  S.doc = { edits: doc.edits || [], review: doc.review || {}, regions: doc.regions || [] };
  await Promise.all(m.pages.map(async (p) => { S.A[p.index] = await (await api(`/api/projects/${pid}/pages/${p.index}/analysis`)).json(); }));
  $("#docname").textContent = m.filename;
  show("screen-editor");
  await loadPage(0, true);
  updateCounts(); renderInfo();
  for (let i = 0; i < m.pages.length; i++) if (pageEdits(i).length) renderPatches(i);
}

async function loadPage(n, fit = false) {
  S.page = n; clearSel();
  const pm = S.meta.pages[n];
  $("#pager").hidden = S.meta.pages.length < 2;
  $("#pg-label").textContent = `${n + 1} / ${S.meta.pages.length}`;
  $("#pg-prev").disabled = n === 0; $("#pg-next").disabled = n >= S.meta.pages.length - 1;
  const img = $("#page-img");
  const page = $("#page");
  page.style.width = pm.width + "px"; page.style.height = pm.height + "px";
  img.width = pm.width; img.height = pm.height;
  await new Promise((res) => { img.onload = res; img.onerror = res; img.src = `/api/projects/${S.pid}/pages/${n}/image/canvas?fmt=jpg`; });
  const warn = (pm.warnings || []).filter((w) => !/^Applied camera/.test(w));
  const ban = $("#page-banner"); ban.hidden = !warn.length; ban.textContent = warn.join(" ");
  if (warn.length) setTimeout(() => (ban.hidden = true), 9000);
  if (fit) fitView();
  drawRegions(); drawPatchLayer(); renderPatches(n);
}

const pageEdits = (n = S.page) => S.doc.edits.filter((e) => (e.page | 0) === n);
const pageRegions = (n = S.page) => S.doc.regions.filter((r) => (r.page | 0) === n);

// ------------------------------------------------------------------ text model helpers
function corrections() {
  const m = {};
  for (const [, d] of Object.entries(S.doc.review)) if (d && d.status === "corrected" && d.target && d.text != null) m[d.target] = d.text;
  return m;
}
function allLines(n = S.page) { return [...(S.A[n]?.regions || []), ...pageRegions(n)]; }
function wordIndex(n = S.page) {
  const idx = {};
  for (const r of allLines(n)) for (const w of r.words || []) idx[w.id] = { w, r };
  return idx;
}
function ocrText(id, n = S.page) {
  const c = corrections();
  if (c[id] != null) return c[id];
  const wi = wordIndex(n)[id]; if (wi) {
    if (c[wi.r.id] != null && wi.r.words.length === 1) return c[wi.r.id];
    return wi.w.text;
  }
  const r = allLines(n).find((r) => r.id === id);
  if (r) return r.words.map((w) => ocrText(w.id, n)).join(" ");
  return "";
}
const keyOf = (ids) => [...ids].sort().join("|");
function findEdit(ids, n = S.page) { const k = keyOf(ids); return S.doc.edits.find((e) => (e.page | 0) === n && keyOf(e.target_ids || []) === k); }
function currentText(ids, n = S.page) { const e = findEdit(ids, n); return e ? e.text : ids.map((i) => ocrText(i, n)).join(" "); }
function unionBox(bs) { bs = bs.filter(Boolean); return bs.length ? [Math.min(...bs.map((b) => b[0])), Math.min(...bs.map((b) => b[1])), Math.max(...bs.map((b) => b[2])), Math.max(...bs.map((b) => b[3]))] : null; }

function flaggedWordIds(n = S.page) {
  const out = new Set(); const done = S.doc.review;
  for (const it of S.A[n]?.review || []) {
    if (done[it.id]) continue;
    if (it.word_ids && it.word_ids.length) it.word_ids.forEach((w) => out.add(w));
    else { const r = (S.A[n].regions || []).find((r) => r.id === it.region_id); r && r.words.forEach((w) => out.add(w.id)); }
  }
  return out;
}

// ------------------------------------------------------------------ drawing
function drawRegions() {
  const layer = $("#region-layer"); layer.innerHTML = "";
  const A = S.A[S.page]; if (!A) return;
  const flagged = flaggedWordIds();
  const pos = S.pos[S.page] || {};
  const edited = new Set(pageEdits().flatMap((e) => e.target_ids || []));
  const frag = document.createDocumentFragment();
  for (const g of [...(A.graphics || []), ...(A.barcodes || [])]) {
    const b = g.group_box || g.box; const d = document.createElement("div"); d.className = "g";
    place(d, b); d.innerHTML = `<span>${g.kind === "barcode" ? "Barcode" : g.kind === "logo" ? "Logo" : "Graphic"} · protected</span>`;
    frag.appendChild(d);
  }
  for (const r of allLines()) {
    const locked = r.locked && !(S.doc.review[`unlock:${r.id}`]);
    for (const w of r.words || []) {
      let b = w.id in pos ? pos[w.id] : w.bbox;
      if (edited.has(r.id)) b = r.id in pos ? null : b;
      if (!b) continue;
      const d = document.createElement("div");
      d.className = "r" + (flagged.has(w.id) ? " low" : "") + (edited.has(w.id) || edited.has(r.id) ? " edited" : "") + (locked ? " locked" : "") + (r.source === "manual" ? " manual" : "");
      d.dataset.w = w.id; d.dataset.l = r.id;
      d.title = locked ? `${r.role === "barcode_text" ? "Barcode number" : "Logo text"} — protected` : `${ocrText(w.id)}  (${Math.round((w.conf ?? 1) * 100)}%)`;
      place(d, b); frag.appendChild(d);
    }
    if (edited.has(r.id) && pos[r.id]) {
      const d = document.createElement("div"); d.className = "r edited"; d.dataset.l = r.id; d.dataset.line = "1"; place(d, pos[r.id]); frag.appendChild(d);
    }
  }
  layer.appendChild(frag);
  layer.classList.toggle("hide-boxes", !$("#show-boxes").checked);
  markHits();
}
function place(el, b) { el.style.left = b[0] + "px"; el.style.top = b[1] + "px"; el.style.width = Math.max(2, b[2] - b[0]) + "px"; el.style.height = Math.max(2, b[3] - b[1]) + "px"; }

function drawPatchLayer() {
  const layer = $("#patch-layer"); layer.innerHTML = "";
  for (const p of S.patches[S.page] || []) {
    if (!p.png) continue;
    const im = document.createElement("img"); im.src = "data:image/png;base64," + p.png; im.alt = "";
    im.style.left = p.x + "px"; im.style.top = p.y + "px"; im.style.width = p.w + "px"; im.style.height = p.h + "px";
    im.dataset.edit = p.edit_id; layer.appendChild(im);
  }
}

let renderTimers = {};
function renderPatches(n = S.page) {
  clearTimeout(renderTimers[n]);
  renderTimers[n] = setTimeout(() => doRender(n), 60);
}
async function doRender(n) {
  const seq = ++S.renderSeq;
  const edits = pageEdits(n);
  if (!edits.length) { S.patches[n] = []; S.pos[n] = {}; S.info[n] = {}; if (n === S.page) { drawPatchLayer(); drawRegions(); } return; }
  try {
    const r = await (await api(`/api/projects/${S.pid}/pages/${n}/render`, { method: "POST", json: { edits, regions: pageRegions(n) } })).json();
    if (seq !== S.renderSeq && n === S.page) { /* a newer render is in flight for this page */ }
    S.patches[n] = r.patches;
    const pos = {}; const info = {};
    const widx = wordIndex(n);
    r.patches.forEach((p, i) => {
      const e = edits[i]; info[e.id] = p.info;
      const nb = p.info.new_text_bbox;
      const ids = e.target_ids || [];
      ids.forEach((id, k) => { pos[id] = k === 0 ? nb : null; });
      for (const mv of p.info.moved || []) {
        const b = pos[mv.id] || widx[mv.id]?.w.bbox; if (b) pos[mv.id] = [b[0] + mv.dx, b[1], b[2] + mv.dx, b[3]];
      }
    });
    S.pos[n] = pos; S.info[n] = info;
    if (n === S.page) { drawPatchLayer(); drawRegions(); refreshSelWarnings(); }
    renderChanges();
  } catch (e) { toast(`Couldn't render edit: ${e.message}`, "error"); }
}

// ------------------------------------------------------------------ view (zoom / pan)
function applyView() {
  const { z, x, y } = S.view;
  $("#page").style.transform = `translate(${x}px, ${y}px) scale(${z})`;
  document.documentElement.style.setProperty("--inv", (1 / z).toFixed(4));
  $("#zoom-label").textContent = `${Math.round(z * 100)}%`;
}
function stageRect() { return $("#stage").getBoundingClientRect(); }
function fitView() {
  const pm = S.meta.pages[S.page]; const r = stageRect();
  const mob = innerWidth <= 820;
  const z = Math.min((r.width - 24) / pm.width, (r.height - (mob ? 70 : 90)) / pm.height);
  S.view = { z, x: (r.width - pm.width * z) / 2, y: mob ? 56 : 16 };
  applyView();
}
function zoomAt(f, cx, cy) {
  const v = S.view; const nz = Math.min(8, Math.max(0.05, v.z * f));
  const k = nz / v.z; v.x = cx - (cx - v.x) * k; v.y = cy - (cy - v.y) * k; v.z = nz; applyView();
}
function zoomTo(b) {
  const r = stageRect(); const pad = innerWidth <= 820 ? 3.5 : 4;
  const w = b[2] - b[0], h = b[3] - b[1];
  const z = Math.min(3, Math.max(S.view.z, Math.min(r.width / (w * pad), (r.height * 0.5) / (h * 6))));
  const visH = innerWidth <= 820 ? r.height * 0.42 : r.height;
  S.view = { z, x: r.width / 2 - ((b[0] + b[2]) / 2) * z, y: visH / 2 - ((b[1] + b[3]) / 2) * z }; applyView();
}
const toPage = (cx, cy) => { const r = stageRect(); return [(cx - r.left - S.view.x) / S.view.z, (cy - r.top - S.view.y) / S.view.z]; };

function initStage() {
  const st = $("#stage");
  const pts = new Map(); let gesture = null;
  st.addEventListener("wheel", (e) => {
    e.preventDefault(); const r = stageRect();
    if (e.ctrlKey || e.metaKey) zoomAt(Math.exp(-e.deltaY * 0.01), e.clientX - r.left, e.clientY - r.top);
    else { S.view.x -= e.deltaX; S.view.y -= e.deltaY; applyView(); }
  }, { passive: false });
  st.addEventListener("pointerdown", (e) => {
    if (e.button > 0 && e.pointerType === "mouse" && e.button !== 1) return;
    if (e.target.closest(".floating, .banner, .toast")) return;
    st.setPointerCapture(e.pointerId);
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      gesture = { kind: "pinch", d: Math.hypot(a[0] - b[0], a[1] - b[1]), z: S.view.z, x: S.view.x, y: S.view.y, c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] };
      return;
    }
    const handle = e.target.closest(".h"); const selBox = e.target.closest(".sel.adjust");
    if (handle || selBox) {
      gesture = { kind: handle ? "resize" : "move", corner: handle && handle.dataset.c, start: toPage(e.clientX, e.clientY), box: [...S.sel.bbox] };
      return;
    }
    const mode = e.button === 1 ? "pan" : S.mode;
    if (mode === "draw") { gesture = { kind: "draw", start: toPage(e.clientX, e.clientY) }; return; }
    gesture = { kind: "maybe-tap", sx: e.clientX, sy: e.clientY, x: S.view.x, y: S.view.y, target: e.target, pan: mode === "pan" };
    if (mode === "pan") st.classList.add("panning");
  });
  st.addEventListener("pointermove", (e) => {
    if (!pts.has(e.pointerId)) return;
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (!gesture) return;
    if (gesture.kind === "pinch" && pts.size >= 2) {
      const [a, b] = [...pts.values()]; const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      const r = stageRect(); const c = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      const nz = Math.min(8, Math.max(0.05, gesture.z * d / gesture.d)); const k = nz / gesture.z;
      const cx = gesture.c[0] - r.left, cy = gesture.c[1] - r.top;
      S.view = { z: nz, x: cx - (cx - gesture.x) * k + (c[0] - gesture.c[0]), y: cy - (cy - gesture.y) * k + (c[1] - gesture.c[1]) }; applyView();
    } else if (gesture.kind === "maybe-tap" || gesture.kind === "pan") {
      const dx = e.clientX - gesture.sx, dy = e.clientY - gesture.sy;
      if (gesture.kind === "maybe-tap" && Math.hypot(dx, dy) > 6) { gesture.kind = "pan"; st.classList.add("panning"); }
      if (gesture.kind === "pan") { S.view.x = gesture.x + dx; S.view.y = gesture.y + dy; applyView(); }
    } else if (gesture.kind === "draw") {
      const p = toPage(e.clientX, e.clientY); gesture.box = norm([...gesture.start, ...p]); showDrawBox(gesture.box);
    } else if (gesture.kind === "resize" || gesture.kind === "move") {
      const p = toPage(e.clientX, e.clientY); const dx = p[0] - gesture.start[0], dy = p[1] - gesture.start[1];
      const b = [...gesture.box];
      if (gesture.kind === "move") { b[0] += dx; b[2] += dx; b[1] += dy; b[3] += dy; }
      else { const c = gesture.corner; if (c.includes("w")) b[0] += dx; if (c.includes("e")) b[2] += dx; if (c.includes("n")) b[1] += dy; if (c.includes("s")) b[3] += dy; }
      S.sel.bbox = norm(b).map(Math.round); drawSel();
    }
  });
  const end = (e) => {
    if (!pts.has(e.pointerId)) return;
    pts.delete(e.pointerId); st.classList.remove("panning");
    if (!gesture) return;
    if (gesture.kind === "pinch") { if (pts.size === 0) gesture = null; return; }
    if (gesture.kind === "maybe-tap" && !gesture.pan) onTap(gesture.target, e);
    if (gesture.kind === "draw" && gesture.box) { $(".drawbox")?.remove(); const b = gesture.box.map(Math.round); if (b[2] - b[0] > 6 && b[3] - b[1] > 6) createManualRegion(b); }
    if (gesture.kind === "resize" || gesture.kind === "move") { S.sel.adjusted = true; }
    gesture = null;
  };
  st.addEventListener("pointerup", end); st.addEventListener("pointercancel", end);
  $("#zoom-in").onclick = () => { const r = stageRect(); zoomAt(1.25, r.width / 2, r.height / 2); };
  $("#zoom-out").onclick = () => { const r = stageRect(); zoomAt(0.8, r.width / 2, r.height / 2); };
  $("#zoom-fit").onclick = fitView;
  for (const [id, m] of [["#mode-select", "select"], ["#mode-pan", "pan"], ["#mode-draw", "draw"]]) $(id).onclick = () => setMode(m);
  addEventListener("resize", () => { if (S.pid && !$("#screen-editor").hidden) applyView(); });
}
function norm(b) { return [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])]; }
function showDrawBox(b) { let d = $(".drawbox"); if (!d) { d = document.createElement("div"); d.className = "drawbox"; $("#sel-layer").appendChild(d); } place(d, b); }
function setMode(m) {
  S.mode = m; for (const [id, k] of [["#mode-select", "select"], ["#mode-pan", "pan"], ["#mode-draw", "draw"]]) $(id).classList.toggle("active", k === m);
  const st = $("#stage"); st.classList.toggle("pan", m === "pan"); st.classList.toggle("draw", m === "draw");
  if (m === "draw") toast("Drag a box around the text you want to edit.");
}

let lastTap = { t: 0, w: null };
function onTap(target) {
  const r = target.closest && target.closest(".r");
  if (!r) { clearSel(); lastTap = { t: 0, w: null }; return; }
  const now = Date.now();
  const dbl = now - lastTap.t < 400 && lastTap.w === (r.dataset.w || r.dataset.l);
  lastTap = { t: now, w: r.dataset.w || r.dataset.l };
  // double-click / double-tap selects the whole line
  if (r.dataset.line || dbl) return selectLine(r.dataset.l);
  selectWord(r.dataset.w, r.dataset.l);
}

// ------------------------------------------------------------------ selection & edit panel
function lineById(id, n = S.page) { return allLines(n).find((r) => r.id === id); }
function selectWord(wid, lid) {
  const line = lineById(lid); if (!line) return;
  // if the whole line is already edited as a unit, keep editing the line
  if (findEdit([lid])) return selectLine(lid);
  const w = line.words.find((w) => w.id === wid);
  const e = findEdit([wid]);
  S.sel = { ids: [wid], line: lid, kind: line.source === "manual" ? "manual" : "word", bbox: e ? [...e.bbox] : [...w.bbox], conf: w.conf, locked: line.locked && !S.doc.review[`unlock:${lid}`], role: line.role };
  openEditPane();
}
function selectLine(lid) {
  const line = lineById(lid); if (!line) return;
  const e = findEdit([lid]);
  S.sel = { ids: [lid], line: lid, kind: line.source === "manual" ? "manual" : "line", bbox: e ? [...e.bbox] : [...line.bbox], conf: line.conf, locked: line.locked && !S.doc.review[`unlock:${lid}`], role: line.role };
  openEditPane();
}
function clearSel() { S.sel = null; $("#sel-layer").innerHTML = ""; if (!$("#pane-edit").hidden) showPane("info"); }

function drawSel() {
  const L = $("#sel-layer"); L.innerHTML = "";
  if (!S.sel) return;
  const d = document.createElement("div"); d.className = "sel" + (S.sel.adjust ? " adjust" : "");
  const e = findEdit(S.sel.ids); const nb = e && S.info[S.page]?.[e.id]?.new_text_bbox;
  place(d, S.sel.adjust ? S.sel.bbox : unionBox([S.sel.bbox, nb]));
  if (S.sel.adjust) for (const c of ["nw", "ne", "sw", "se"]) { const h = document.createElement("div"); h.className = `h ${c}`; h.dataset.c = c; d.appendChild(h); }
  L.appendChild(d);
}

function openEditPane(focus = true) {
  const s = S.sel; showPane("edit"); drawSel();
  const orig = s.ids.map((i) => ocrText(i)).join(" ");
  $("#edit-title").textContent = s.kind === "line" ? "Edit line" : s.kind === "manual" ? "Edit region" : "Edit text";
  $("#edit-orig").textContent = orig || "—";
  const pc = $("#edit-conf"); const c = s.conf ?? 1;
  pc.textContent = s.kind === "manual" ? "manual" : `${Math.round(c * 100)}% sure`; pc.className = "pill " + (c < 0.85 ? "low" : "good");
  const flagged = flaggedWordIds(); const isFlag = s.ids.some((i) => flagged.has(i)) || (s.kind === "line" && lineById(s.line).words.some((w) => flagged.has(w.id)));
  const ew = $("#edit-warn"); ew.hidden = !isFlag; ew.textContent = isFlag ? "This reading is uncertain — check the original text above against the page before editing." : "";
  const lk = $("#edit-locked"); lk.hidden = !s.locked;
  if (s.locked) $("span", lk).textContent = s.role === "barcode_text" ? "This number belongs to the barcode. Changing it will not change the barcode." : "This text is part of a logo and is protected.";
  const inp = $("#edit-input"); inp.disabled = !!s.locked;
  inp.value = currentText(s.ids);
  const e = findEdit(s.ids);
  $("#btn-revert").hidden = !e;
  $("#btn-scope").textContent = s.kind === "line" ? "Select single word" : "Select whole line";
  $("#btn-scope").hidden = s.kind === "manual" || (lineById(s.line)?.words.length || 0) < 2;
  $("#btn-adjust").classList.toggle("on", !!s.adjust);
  for (const b of ["#btn-apply", "#btn-delete", "#btn-adjust"]) $(b).disabled = !!s.locked;
  // style controls
  const st = (e && e.style) || {};
  $("#st-family").value = st.family || ""; $("#st-bold").value = st.bold === true ? "1" : st.bold === false ? "0" : "";
  $("#st-size").value = st.size_px || ""; $("#st-align").value = st.align || "";
  $("#st-color-auto").checked = !st.color; $("#st-color").value = st.color || (S.info[S.page]?.[e?.id]?.style?.color) || "#222222";
  $("#nudge-val").textContent = st.dx || st.dy ? `${st.dx || 0}, ${st.dy || 0} px` : "";
  const fit = e && S.info[S.page]?.[e.id]?.style;
  $("#style-summary").textContent = fit ? `· ${fit.family}${fit.bold ? " bold" : ""} ${fit.size_px}px` : "";
  refreshSelWarnings();
  if (focus && innerWidth > 820 && !s.locked) setTimeout(() => { inp.focus(); inp.select(); }, 30);
  ensureSelVisible();
}
function ensureSelVisible() {
  if (!S.sel) return; const r = stageRect(); const b = S.sel.bbox; const v = S.view;
  const sx0 = b[0] * v.z + v.x, sy0 = b[1] * v.z + v.y, sx1 = b[2] * v.z + v.x, sy1 = b[3] * v.z + v.y;
  const visH = innerWidth <= 820 ? r.height * 0.42 : r.height;
  if (sx0 < 0 || sy0 < 0 || sx1 > r.width || sy1 > visH || (sx1 - sx0) < 24) zoomTo(b);
}
function refreshSelWarnings() {
  if (!S.sel) return; const e = findEdit(S.sel.ids); const inf = e && S.info[S.page]?.[e.id];
  const ew = $("#edit-warn");
  if (inf && inf.warnings && inf.warnings.length) { ew.hidden = false; ew.textContent = inf.warnings.join(" "); }
  if (inf && inf.style) $("#style-summary").textContent = `· ${inf.style.family}${inf.style.bold ? " bold" : ""} ${inf.style.size_px}px`;
  drawSel();
}

function styleFromControls(prev = {}) {
  const st = { ...prev };
  const fam = $("#st-family").value; fam ? (st.family = fam) : delete st.family;
  const b = $("#st-bold").value; b === "" ? delete st.bold : (st.bold = b === "1");
  const sz = parseFloat($("#st-size").value); sz > 0 ? (st.size_px = sz) : delete st.size_px;
  const al = $("#st-align").value; al ? (st.align = al) : delete st.align;
  if ($("#st-color-auto").checked) delete st.color; else st.color = $("#st-color").value;
  return st;
}

function applyEdit(text, opts = {}) {
  const s = S.sel; if (!s || s.locked) return;
  const existing = findEdit(s.ids);
  const orig = s.ids.map((i) => ocrText(i)).join(" ");
  if (!existing && text === orig && !s.adjusted && !opts.style) { toast("No change."); return; }
  commit(() => {
    const e = existing || { id: "e" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6), page: S.page, target_ids: [...s.ids], created: Date.now() };
    e.text = text; e.original_text = orig; e.bbox = [...s.bbox]; e.source = s.kind === "manual" ? "manual" : "ocr";
    e.style = opts.style !== undefined ? opts.style : styleFromControls(existing?.style);
    if (!existing) S.doc.edits.push(e);
  });
  toast(text ? "Text replaced" : "Text deleted");
  if (S.sel) openEditPane(false);
  document.activeElement && document.activeElement.blur && document.activeElement.blur();
}
function revertEdit() {
  const e = S.sel && findEdit(S.sel.ids); if (!e) return;
  commit(() => { S.doc.edits = S.doc.edits.filter((x) => x !== e); });
  const l = lineById(S.sel.line); if (l) S.sel.bbox = S.sel.kind === "line" ? [...l.bbox] : [...(l.words.find((w) => w.id === S.sel.ids[0])?.bbox || S.sel.bbox)];
  openEditPane(); toast("Original restored");
}

function initEditPane() {
  $("#btn-apply").onclick = () => applyEdit($("#edit-input").value.replace(/\s+/g, " ").trim());
  $("#btn-delete").onclick = () => { $("#edit-input").value = ""; applyEdit(""); };
  $("#btn-revert").onclick = revertEdit;
  $("#edit-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#btn-apply").click(); } if (e.key === "Escape") clearSel(); });
  $("#btn-scope").onclick = () => { const s = S.sel; if (!s) return; if (s.kind === "line") { const w = lineById(s.line).words[0]; selectWord(w.id, s.line); } else selectLine(s.line); };
  $("#btn-adjust").onclick = () => { if (!S.sel) return; S.sel.adjust = !S.sel.adjust; $("#btn-adjust").classList.toggle("on", S.sel.adjust); drawSel(); if (S.sel.adjust) toast("Drag the box or its corners, then Apply."); };
  $("#btn-unlock").onclick = () => { if (!S.sel) return; commit(() => { S.doc.review[`unlock:${S.sel.line}`] = { status: "unlocked" }; }); S.sel.locked = false; openEditPane(); };
  const restyle = () => { const e = S.sel && findEdit(S.sel.ids); if (e) applyEdit(e.text, { style: styleFromControls(e.style) }); };
  for (const id of ["#st-family", "#st-bold", "#st-size", "#st-align", "#st-color", "#st-color-auto"]) $(id).addEventListener("change", restyle);
  for (const b of $$("[data-nudge]")) b.onclick = () => {
    const e = S.sel && findEdit(S.sel.ids); if (!e) { toast("Apply the text first, then nudge it."); return; }
    const [dx, dy] = b.dataset.nudge.split(",").map(Number); const st = { ...(e.style || {}) };
    st.dx = (st.dx || 0) + dx; st.dy = (st.dy || 0) + dy; applyEdit(e.text, { style: st });
    $("#nudge-val").textContent = `${st.dx}, ${st.dy} px`;
  };
}

async function createManualRegion(b) {
  setMode("select");
  const id = "m" + Date.now().toString(36);
  let res = { text: "", conf: 0 };
  try { res = await (await api(`/api/projects/${S.pid}/pages/${S.page}/ocr-region`, { method: "POST", json: { bbox: b } })).json(); } catch (e) { toast(e.message, "error"); }
  const region = { id, page: S.page, kind: "line", source: "manual", text: res.text || "", conf: res.conf || 0, bbox: b, role: "text", locked: false,
    words: [{ id: id + "-w0", text: res.text || "", conf: res.conf || 0, bbox: b, style: {} }], style: { align: "left" }, flags: [], table: null };
  commit(() => { S.doc.regions.push(region); });
  S.sel = { ids: [id], line: id, kind: "manual", bbox: [...b], conf: res.conf || 0 };
  openEditPane();
  if (res.text) toast(res.needs_review ? `Read "${res.text}" — uncertain, please check.` : `Read "${res.text}"`);
  else toast("No text recognised here — type what the new text should be.");
}

// ------------------------------------------------------------------ history & save
function snapshot() { return JSON.stringify(S.doc); }
function commit(fn) {
  S.undo.push(snapshot()); if (S.undo.length > 200) S.undo.shift(); S.redo = [];
  fn(); afterChange();
}
function undo() { if (!S.undo.length) return; S.redo.push(snapshot()); S.doc = JSON.parse(S.undo.pop()); afterChange(true); toast("Undone"); }
function redo() { if (!S.redo.length) return; S.undo.push(snapshot()); S.doc = JSON.parse(S.redo.pop()); afterChange(true); toast("Redone"); }
function afterChange(historyNav = false) {
  updateCounts(); scheduleSave();
  for (let i = 0; i < S.meta.pages.length; i++) renderPatches(i);
  drawRegions(); renderChanges();
  if (!$("#pane-review").hidden) renderReview();
  if (historyNav && S.sel) { const e = findEdit(S.sel.ids); if (!e && S.sel.kind === "manual" && !lineById(S.sel.line)) clearSel(); else openEditPane(false); }
}
function scheduleSave() { clearTimeout(S.saveTimer); S.saveTimer = setTimeout(flushSave, 350); }
async function flushSave() {
  clearTimeout(S.saveTimer); if (!S.pid) return;
  const body = { edits: S.doc.edits, review: S.doc.review, regions: S.doc.regions };
  const p = api(`/api/projects/${S.pid}/edits`, { method: "PUT", json: body }).catch((e) => toast(`Not saved: ${e.message}`, "error"));
  S.saving = p; await p;
}
function updateCounts() {
  let n = 0; for (const p of S.meta.pages) for (const it of S.A[p.index]?.review || []) if (!S.doc.review[it.id]) n++;
  const rc = $("#review-count"); rc.hidden = !n; rc.textContent = n;
  const ec = $("#edit-count"); ec.hidden = !S.doc.edits.length; ec.textContent = S.doc.edits.length;
  $("#btn-undo").disabled = !S.undo.length; $("#btn-redo").disabled = !S.redo.length;
}

// ------------------------------------------------------------------ panes
function showPane(name) {
  for (const p of $$(".pane")) p.hidden = p.id !== `pane-${name}`;
  const panel = $("#panel"); panel.classList.toggle("info-only", name === "info"); panel.classList.remove("collapsed");
  if (name !== "edit") { S.sel = null; $("#sel-layer").innerHTML = ""; }
}
function renderInfo() {
  let words = 0, lines = 0, review = 0, conf = [];
  for (const p of S.meta.pages) { const st = S.A[p.index]?.stats || {}; words += st.words || 0; lines += st.lines || 0; review += st.review_items || 0; if (st.mean_conf) conf.push(st.mean_conf); }
  const pm = S.meta.pages[S.page];
  const steps = (pm.steps || []).map((s) => s.step.replace(/_/g, " ")).join(", ");
  $("#doc-stats").innerHTML = `<div><b>${S.meta.pages.length}</b> page(s) · <b>${words}</b> words in <b>${lines}</b> lines</div>
    <div>Average OCR confidence <b>${conf.length ? Math.round(100 * conf.reduce((a, b) => a + b) / conf.length) : "–"}%</b> · <b>${review}</b> flagged for review</div>
    <div class="fine">Image steps: ${esc(steps || "none")}</div>`;
}

function renderReview() {
  const ul = $("#review-list"); ul.innerHTML = "";
  let open = 0;
  for (const p of S.meta.pages) for (const it of S.A[p.index]?.review || []) {
    const d = S.doc.review[it.id]; if (!d) open++;
    const li = document.createElement("li"); if (d) li.classList.add("done");
    const target = it.word_ids && it.word_ids.length === 1 ? it.word_ids[0] : it.region_id;
    const cur = ocrText(target, p.index);
    const alts = (it.alternatives || []).filter((a) => a.text && a.text !== cur);
    li.innerHTML = `<div class="why">${esc(it.type.replace(/_/g, " "))}${S.meta.pages.length > 1 ? ` · page ${p.index + 1}` : ""}</div>
      <div class="small">${esc(it.message)}</div><div class="txt">${esc(cur)}</div>
      ${alts.length ? `<div class="small muted">Other readings:</div><div class="alts">${alts.map((a, i) => `<button class="chip" data-alt="${i}">${esc(a.text)} <span class="muted">${a.engine}</span></button>`).join("")}</div>` : ""}
      <div class="acts">${d ? `<span class="small muted">${d.status === "corrected" ? "Corrected" : "Confirmed"}</span><button class="chip" data-undo>Reopen</button>` :
        `<button class="chip" data-ok><svg class="ic inline"><use href="#i-check"/></svg> Looks right</button><button class="chip" data-fix>Correct…</button>`}<button class="chip" data-show>Show on page</button></div>`;
    $$("[data-alt]", li).forEach((b) => b.onclick = () => commit(() => { S.doc.review[it.id] = { status: "corrected", text: alts[+b.dataset.alt].text, target }; }));
    $("[data-ok]", li) && ($("[data-ok]", li).onclick = () => commit(() => { S.doc.review[it.id] = { status: "accepted", target }; }));
    $("[data-fix]", li) && ($("[data-fix]", li).onclick = () => { const t = prompt("What does the original text actually say?", cur); if (t != null) commit(() => { S.doc.review[it.id] = { status: "corrected", text: t.trim(), target }; }); });
    $("[data-undo]", li) && ($("[data-undo]", li).onclick = () => commit(() => { delete S.doc.review[it.id]; }));
    $("[data-show]", li).onclick = async () => { if (p.index !== S.page) await loadPage(p.index); zoomTo(it.bbox); const wid = it.word_ids?.[0]; wid ? selectWord(wid, it.region_id) : selectLine(it.region_id); };
    ul.appendChild(li);
  }
  $("#review-intro").textContent = open ? `${open} reading(s) were flagged instead of guessed. Confirm or correct them — this fixes the recognised text only, never the page image.` : "Nothing left to review. Every flagged reading has been checked.";
}

function renderChanges() {
  const ul = $("#change-list"); ul.innerHTML = "";
  $("#changes-empty").hidden = !!S.doc.edits.length;
  for (const e of S.doc.edits) {
    const inf = S.info[e.page]?.[e.id] || {};
    const li = document.createElement("li");
    li.innerHTML = `<div class="small muted">${S.meta.pages.length > 1 ? `Page ${(e.page | 0) + 1} · ` : ""}${e.source === "manual" ? "Manual region" : (e.target_ids || []).length > 1 || /-l\d+(s\d+)?$/.test(e.target_ids?.[0] || "") ? "Line" : "Word"}</div>
      <div><span class="from">${esc(e.original_text || "—")}</span> → <span class="to">${e.text ? esc(e.text) : "<i>deleted</i>"}</span></div>
      ${(inf.warnings || []).map((w) => `<div class="warnline">${esc(w)}</div>`).join("")}
      ${inf.moved && inf.moved.length ? `<div class="fine muted">Following text on the line moved ${inf.moved[0].dx > 0 ? "right" : "left"} ${Math.abs(inf.moved[0].dx)} px (pixels kept exact).</div>` : ""}
      <div class="acts"><button class="chip" data-go>Show</button><button class="chip" data-rm>Undo this change</button></div>`;
    $("[data-go]", li).onclick = async () => { if ((e.page | 0) !== S.page) await loadPage(e.page | 0); zoomTo(e.bbox); const l = e.target_ids[0]; const line = allLines().find((r) => r.id === l || r.words.some((w) => w.id === l)); if (line) (line.id === l ? selectLine(l) : selectWord(l, line.id)); };
    $("[data-rm]", li).onclick = () => commit(() => { S.doc.edits = S.doc.edits.filter((x) => x !== e); });
    ul.appendChild(li);
  }
}

// ------------------------------------------------------------------ find & replace
function runFind() {
  const q = $("#find-q").value; const cs = $("#find-case").checked; S.find = { hits: [], i: -1 };
  if (!q) { $("#find-status").textContent = ""; markHits(); return; }
  const nq = cs ? q : q.toLowerCase();
  for (const p of S.meta.pages) {
    const n = p.index; const edited = new Set(S.doc.edits.filter((e) => (e.page | 0) === n).flatMap((e) => e.target_ids));
    for (const r of allLines(n)) {
      if (r.locked || edited.has(r.id)) continue;
      const words = r.words.filter((w) => !edited.has(w.id));
      let txt = "", spans = [];
      for (const w of words) { const t = ocrText(w.id, n); if (txt) txt += " "; spans.push([txt.length, txt.length + t.length, w]); txt += t; }
      const hay = cs ? txt : txt.toLowerCase();
      let k = hay.indexOf(nq);
      while (k >= 0) {
        const ws = spans.filter(([a, b]) => b > k && a < k + nq.length);
        if (ws.length) {
          const s0 = ws[0][0]; const s1 = ws[ws.length - 1][1];
          S.find.hits.push({ page: n, line: r.id, ids: ws.map((s) => s[2].id), bbox: unionBox(ws.map((s) => s[2].bbox)), full: txt.slice(s0, s1), from: k - s0, len: nq.length });
        }
        k = hay.indexOf(nq, k + Math.max(1, nq.length));
      }
    }
  }
  $("#find-status").textContent = S.find.hits.length ? `${S.find.hits.length} match(es)` : "No matches";
  markHits();
}
function markHits() {
  $$(".r.hit").forEach((d) => d.classList.remove("hit", "cur"));
  const cur = S.find.hits[S.find.i];
  for (const h of S.find.hits) if (h.page === S.page) for (const id of h.ids) { const d = $(`.r[data-w="${CSS.escape(id)}"]`); if (d) { d.classList.add("hit"); if (h === cur) d.classList.add("cur"); } }
}
async function gotoHit(step) {
  if (!S.find.hits.length) return; S.find.i = (S.find.i + step + S.find.hits.length) % S.find.hits.length;
  const h = S.find.hits[S.find.i]; if (h.page !== S.page) await loadPage(h.page);
  zoomTo(h.bbox); markHits(); $("#find-status").textContent = `${S.find.i + 1} of ${S.find.hits.length}`;
}
function replaceHits(hits) {
  const r = $("#find-r").value;
  if (!hits.length) return;
  commit(() => {
    for (const h of hits) {
      const text = h.full.slice(0, h.from) + r + h.full.slice(h.from + h.len);
      S.doc.edits.push({ id: "e" + Date.now().toString(36) + Math.random().toString(36).slice(2, 7), page: h.page, target_ids: h.ids, text: text.replace(/\s+/g, " ").trim(),
        original_text: h.full, bbox: h.bbox, source: "ocr", style: {}, created: Date.now() });
    }
  });
  toast(`Replaced ${hits.length} occurrence(s)`); runFind();
}
function initFind() {
  let t; $("#find-q").addEventListener("input", () => { clearTimeout(t); t = setTimeout(runFind, 150); });
  $("#find-case").onchange = runFind;
  $("#find-q").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); gotoHit(e.shiftKey ? -1 : 1); } });
  $("#find-next").onclick = () => gotoHit(1); $("#find-prev").onclick = () => gotoHit(-1);
  $("#find-one").onclick = () => { if (S.find.i < 0) gotoHit(1); const h = S.find.hits[S.find.i]; if (h) replaceHits([h]); };
  $("#find-all").onclick = () => replaceHits(S.find.hits);
}

// ------------------------------------------------------------------ compare
let cmpMode = "side";
async function openCompare() {
  await flushSave();
  $("#compare").hidden = false; renderCompare();
  const st = $("#cmp-stats"); st.textContent = "Measuring changes…";
  try {
    const f = await (await api(`/api/projects/${S.pid}/pages/${S.page}/fidelity`)).json();
    const pct = (f.changed_fraction * 100).toFixed(f.changed_fraction < 0.001 ? 3 : 2);
    st.innerHTML = f.edits ? `<b>${f.edits}</b> edit(s) on this page changed <b>${f.changed_pixels.toLocaleString()}</b> pixels (${pct}% of the page).
      ${f.outside_intended_pixels === 0 ? `<span class="ok">✓ Every other pixel is identical to the original.</span>` : `<span class="bad">⚠ ${f.outside_intended_pixels} pixels changed outside the edited areas.</span>`}`
      : `No edits on this page — it is identical to the original.`;
  } catch (e) { st.textContent = e.message; }
}
function renderCompare() {
  $$("#cmp-modes button").forEach((b) => b.classList.toggle("on", b.dataset.m === cmpMode));
  const v = Date.now(); const base = `/api/projects/${S.pid}/pages/${S.page}`;
  const orig = `${base}/image/canvas?max_side=2400`; const edit = `${base}/edited?max_side=2400&v=${v}`;
  const body = $("#cmp-body");
  if (cmpMode === "side") body.innerHTML = `<div class="cmp-side"><figure><figcaption>Original</figcaption><img src="${orig}" alt="Original"></figure><figure><figcaption>Edited</figcaption><img src="${edit}" alt="Edited"></figure></div>`;
  else if (cmpMode === "diff") body.innerHTML = `<div class="cmp-one"><div class="small muted" style="margin-bottom:6px">Red = changed pixels. Everything grey is untouched.</div><img src="${base}/diff.png?v=${v}" alt="Changed pixels"></div>`;
  else if (cmpMode === "overlay") {
    body.innerHTML = `<div class="cmp-controls">Edited opacity <input type="range" id="ov-op" min="0" max="100" value="50"> <button class="chip" id="ov-blink">Blink</button></div>
      <div class="cmp-one"><div class="overlay-wrap"><img src="${orig}" alt="Original"><img class="ov" src="${edit}" alt="Edited" style="opacity:.5"></div></div>`;
    const ov = $(".ov", body); $("#ov-op").oninput = (e) => (ov.style.opacity = e.target.value / 100);
    let bl = null; $("#ov-blink").onclick = (e) => { if (bl) { clearInterval(bl); bl = null; e.target.classList.remove("on"); } else { e.target.classList.add("on"); bl = setInterval(() => (ov.style.opacity = ov.style.opacity === "1" ? "0" : "1"), 600); } };
  } else {
    body.innerHTML = `<div class="cmp-one"><div class="swipe" id="swipe"><img src="${orig}" alt="Original"><div class="top"><img src="${edit}" alt="Edited"></div><div class="handle"></div></div><div class="small muted" style="text-align:center;margin-top:6px">Left: edited · Right: original — drag the handle</div></div>`;
    const sw = $("#swipe"); const top = $(".top", sw); const h = $(".handle", sw); const bimg = $("img", sw);
    const setX = (f) => { f = Math.min(1, Math.max(0, f)); top.style.width = f * 100 + "%"; h.style.left = `calc(${f * 100}% - 1px)`; $("img", top).style.width = bimg.clientWidth + "px"; };
    bimg.onload = () => setX(0.5); setX(0.5);
    let drag = false; sw.onpointerdown = (e) => { drag = true; sw.setPointerCapture(e.pointerId); setX((e.clientX - sw.getBoundingClientRect().left) / sw.clientWidth); };
    sw.onpointermove = (e) => { if (drag) setX((e.clientX - sw.getBoundingClientRect().left) / sw.clientWidth); };
    sw.onpointerup = () => (drag = false);
  }
}

// ------------------------------------------------------------------ export
function openExport() {
  $("#export").hidden = false; $("#exp-status").textContent = "";
  let n = 0; for (const p of S.meta.pages) for (const it of S.A[p.index]?.review || []) if (!S.doc.review[it.id]) n++;
  const w = $("#exp-review-warn"); w.hidden = !n;
  w.textContent = n ? `${n} uncertain OCR reading(s) haven't been reviewed. They don't change the page image, but check them if you rely on search/copy text in the PDF.` : "";
  const sync = () => { $("#exp-compact-row").hidden = $("input[name=fmt]:checked").value !== "pdf"; };
  $$("input[name=fmt]").forEach((r) => (r.onchange = sync)); sync();
}
async function doExport() {
  const fmt = $("input[name=fmt]:checked").value; const btn = $("#exp-go"); btn.disabled = true;
  $("#exp-status").textContent = "Preparing…";
  try {
    await flushSave();
    const r = await api(`/api/projects/${S.pid}/export`, { method: "POST", json: { format: fmt, quality: $("#exp-compact").checked ? "compact" : "lossless" } });
    const blob = await r.blob();
    const cd = r.headers.get("Content-Disposition") || ""; const name = (cd.match(/filename="([^"]+)"/) || [])[1] || `document.${fmt}`;
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 4000);
    $("#exp-status").textContent = `Downloaded ${name} (${(blob.size / 1e6).toFixed(1)} MB)`;
  } catch (e) { $("#exp-status").textContent = e.message; toast(`Export failed: ${e.message}`, "error"); }
  finally { btn.disabled = false; }
}

// ------------------------------------------------------------------ wiring
function initEditor() {
  initStage(); initEditPane(); initFind();
  $("#btn-home").onclick = goHome;
  $("#btn-undo").onclick = undo; $("#btn-redo").onclick = redo;
  $("#btn-find").onclick = () => { showPane("find"); setTimeout(() => $("#find-q").focus(), 30); };
  $("#btn-review").onclick = () => { showPane("review"); renderReview(); };
  $("#btn-changes").onclick = () => { showPane("changes"); renderChanges(); };
  $("#btn-compare").onclick = openCompare;
  $("#cmp-close").onclick = () => ($("#compare").hidden = true);
  $$("#cmp-modes button").forEach((b) => (b.onclick = () => { cmpMode = b.dataset.m; renderCompare(); }));
  $("#btn-export").onclick = openExport; $("#exp-close").onclick = () => ($("#export").hidden = true); $("#exp-go").onclick = doExport;
  $$("[data-close]").forEach((b) => (b.onclick = () => { clearSel(); showPane("info"); }));
  $("#pg-prev").onclick = () => loadPage(S.page - 1, true); $("#pg-next").onclick = () => loadPage(S.page + 1, true);
  $("#show-boxes").onchange = drawRegions;
  $("#panel-grip").onclick = () => $("#panel").classList.toggle("collapsed");
  // close a modal by tapping its backdrop — only when the press started on the backdrop itself
  // (on touch screens the tap that opened the modal is followed by a "ghost" click on it)
  for (const m of ["#compare", "#export"]) {
    let downOnBackdrop = false;
    $(m).addEventListener("pointerdown", (e) => { downOnBackdrop = e.target.id === m.slice(1); });
    $(m).addEventListener("click", (e) => { if (downOnBackdrop && e.target.id === m.slice(1)) $(m).hidden = true; downOnBackdrop = false; });
  }
  addEventListener("keydown", (e) => {
    if ($("#screen-editor").hidden) return;
    const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName);
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === "z" && !typing) { e.preventDefault(); e.shiftKey ? redo() : undo(); }
    else if (mod && e.key.toLowerCase() === "y" && !typing) { e.preventDefault(); redo(); }
    else if (mod && e.key.toLowerCase() === "f") { e.preventDefault(); $("#btn-find").click(); }
    else if (e.key === "Escape") { $("#compare").hidden = true; $("#export").hidden = true; if (!typing) clearSel(); }
    else if (!typing && !mod) {
      if ((e.key === "Delete" || e.key === "Backspace") && S.sel && !S.sel.locked) { e.preventDefault(); applyEdit(""); }
      else if (e.key === "v") setMode("select"); else if (e.key === "h") setMode("pan"); else if (e.key === "r") setMode("draw");
      else if (e.key === "+" || e.key === "=") $("#zoom-in").click(); else if (e.key === "-") $("#zoom-out").click(); else if (e.key === "0") fitView();
    }
  });
  addEventListener("beforeunload", () => { if (S.pid) navigator.sendBeacon && flushSave(); });
}

initUpload(); initEditor();
const m = location.hash.match(/p=([a-f0-9]{12})/);
if (m) openProject(m[1]); else show("screen-upload");
