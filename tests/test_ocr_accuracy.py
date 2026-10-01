"""Gate 3 — OCR accuracy & placement on realistic photographed documents (ground truth from the DOM)."""
import pytest

from helpers import RESULTS, save_result

KEY_FIELDS = ["first_name", "last_name", "item_desc", "reference", "order_no", "qty", "price", "amount",
              "subtotal", "tax", "total"]


@pytest.fixture(scope="module")
def evals():
    import eval_ocr

    out = {}
    for case in ["photo", "invoice", "clean", "rot90", "exif", "lowres"]:
        s, A = eval_ocr.run(case, overlay=str(RESULTS / f"ocr_overlay_{case}.jpg"))
        out[case] = (s, A)
        save_result(f"ocr_eval_{case}.json", s)
    return out


@pytest.mark.parametrize("case", ["photo", "invoice", "clean", "rot90", "exif"])
def test_word_accuracy_and_position(evals, case):
    s, A = evals[case]
    assert s["detection_recall"] >= 0.98, s["errors"]
    assert s["word_accuracy"] >= 0.97, s["errors"]
    assert s["median_center_err_px"] <= 4.0
    assert s["p95_center_err_px"] <= 10.0


@pytest.mark.parametrize("case", ["photo", "invoice", "rot90", "exif"])
def test_key_fields_exact(evals, case):
    s, _ = evals[case]
    bad = {f: v for f, v in s["fields_wrong"].items() if f in KEY_FIELDS}
    assert not bad, bad
    # small table labels
    labels = {e["gt"] for e in s["errors"]}
    for lab in ("QTY", "SKU", "LINE", "AMOUNT", "PRICE", "TOTAL"):
        assert lab not in labels


@pytest.mark.parametrize("case", ["photo", "invoice", "clean", "rot90", "exif", "lowres"])
def test_no_silent_errors(evals, case):
    """Every wrong reading must be flagged for review — never silently guessed."""
    s, _ = evals[case]
    assert s["silent_errors"] == [], s["silent_errors"]


def test_lowres_is_flagged_not_guessed(evals):
    s, A = evals["lowres"]
    assert A["stats"]["low_resolution"]
    assert any("low resolution" in w.lower() for w in A["warnings"])
    assert len(A["review"]) >= 0.5 * A["stats"]["lines"]


@pytest.mark.parametrize("case", ["photo", "invoice"])
def test_structure_detected(evals, case):
    s, A = evals[case]
    assert len(A["barcodes"]) == 1
    assert [g["kind"] for g in A["graphics"]] == ["logo"]
    assert any(t["rows"] >= 6 and t["cols"] >= 5 for t in A["tables"])
    roles = {r["role"] for r in A["regions"]}
    assert {"logo_text", "barcode_text"} <= roles
    # barcode bars never become editable text
    bc = A["barcodes"][0]["box"]
    for r in A["regions"]:
        b = r["bbox"]
        inter = max(0, min(b[2], bc[2]) - max(b[0], bc[0])) * max(0, min(b[3], bc[3]) - max(b[1], bc[1]))
        assert inter < 0.2 * (b[2] - b[0]) * (b[3] - b[1]), r["text"]


def test_region_metadata_complete(evals):
    _, A = evals["photo"]
    for r in A["regions"]:
        for k in ("text", "bbox", "conf", "page", "words", "style", "flags", "role"):
            assert k in r
        assert r["style"]["align"] in ("left", "right", "center")
        for w in r["words"]:
            st = w["style"]
            assert st["font_size_px"] and st["weight"] in ("regular", "bold") and st["color"].startswith("#")
            assert 0 <= w["conf"] <= 1


def test_table_cells_assigned(evals):
    _, A = evals["photo"]
    desc = [r for r in A["regions"] if r["text"] == "Organic Potting Soil 50 L"][0]
    qty = [r for r in A["regions"] if r["table"] and r["table"]["row"] == desc["table"]["row"]
           and r["table"]["col"] == 3][0]
    assert qty["text"] == "6" and qty["style"]["align"] == "right"


def test_weight_estimation(evals):
    s, _ = evals["photo"]
    assert s["weight_accuracy"] >= 0.9
