import io
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pyiqa
import requests
from flask import Flask, Response, request
from PIL import Image, ImageDraw, ImageFont, ImageOps

# smartcrop – optional (fallback retained)
try:
    import smartcrop as _smartcrop

    _SMARTCROP = _smartcrop.SmartCrop()
    _SMARTCROP_ERROR = None
except Exception as e:
    _smartcrop = None
    _SMARTCROP = None
    _SMARTCROP_ERROR = e

# ---------------------------------------------------------------------------
# Panel definition (Authoritative epd7in3e / E6 values)
# ---------------------------------------------------------------------------
PANEL_W, PANEL_H = 800, 480  # The panel itself is always landscape-native
FRAME_BYTES = PANEL_W * PANEL_H // 2  # 192000 bytes

WIDTH, HEIGHT = PANEL_W, PANEL_H

PALETTE_RGB = [
    (0, 0, 0),  # black
    (255, 255, 255),  # white
    (255, 243, 56),  # yellow
    (191, 35, 35),  # red
    (45, 65, 170),  # blue
    (35, 130, 75),  # green
]
CODE_LUT = np.array([0x0, 0x1, 0x2, 0x3, 0x5, 0x6], dtype=np.uint8)

# ---------------------------------------------------------------------------
# Configuration & Environment Setup
# ---------------------------------------------------------------------------
IMMICH_URL = os.environ.get("IMMICH_URL", "http://immich-server:2283").rstrip("/")
IMMICH_PUBLIC_URL = os.environ.get("IMMICH_PUBLIC_URL", IMMICH_URL).rstrip("/")
API_KEY = os.environ["IMMICH_API_KEY"]
ALBUM_ID = os.environ.get("IMMICH_ALBUM_ID", "").strip()
ROTATE = int(os.environ.get("FRAME_ROTATE", "0"))


def _csv_env(name: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


INCLUDE_PEOPLE = _csv_env("IMMICH_INCLUDE_PEOPLE")
EXCLUDE_PEOPLE = set(_csv_env("IMMICH_EXCLUDE_PEOPLE"))
REQUIRE_ALL_PEOPLE = os.environ.get("IMMICH_REQUIRE_ALL_PEOPLE", "false").lower() == "true"
DATE_AFTER = os.environ.get("IMMICH_DATE_AFTER", "").strip()
DATE_BEFORE = os.environ.get("IMMICH_DATE_BEFORE", "").strip()
SEARCH_BATCH = int(os.environ.get("IMMICH_SEARCH_BATCH", "250"))

DEVICE_ORIENTATION = os.environ.get("IMMICH_DEVICE_ORIENTATION", "landscape").strip().lower()
if DEVICE_ORIENTATION not in {"landscape", "portrait"}:
    raise SystemExit(f"IMMICH_DEVICE_ORIENTATION={DEVICE_ORIENTATION!r} must be landscape|portrait")

if DEVICE_ORIENTATION == "landscape":
    CANVAS_W, CANVAS_H = PANEL_W, PANEL_H
    SINGLE_SHAPE = "landscape"
    DUO_SHAPE = "portrait"
else:
    CANVAS_W, CANVAS_H = PANEL_H, PANEL_W
    SINGLE_SHAPE = "portrait"
    DUO_SHAPE = "landscape"

try:
    DUO_PROBABILITY = float(os.environ.get("IMMICH_DUO_PROBABILITY", "0.5"))
except ValueError:
    raise SystemExit("IMMICH_DUO_PROBABILITY must be a float between 0 and 1")
if not 0.0 <= DUO_PROBABILITY <= 1.0:
    raise SystemExit(f"IMMICH_DUO_PROBABILITY={DUO_PROBABILITY} must be between 0 and 1")

ASSET_ORIENTATION = os.environ.get("IMMICH_ASSET_ORIENTATION", "any").strip().lower()
if ASSET_ORIENTATION not in {"any", "landscape", "portrait", "square"}:
    raise SystemExit(f"IMMICH_ASSET_ORIENTATION={ASSET_ORIENTATION!r} must be any|landscape|portrait|square")

CROP_MODE = os.environ.get("IMMICH_CROP", "center").strip().lower()
if CROP_MODE not in {"center", "smart"}:
    raise SystemExit(f"IMMICH_CROP={CROP_MODE!r} must be center|smart")
if CROP_MODE == "smart" and _SMARTCROP is None:
    raise SystemExit(f"IMMICH_CROP=smart but smartcrop import failed: {_SMARTCROP_ERROR!r}")

RANKING_BATCH = int(os.environ.get("IMMICH_RANKING_BATCH", "5"))
QUALITY_ENABLED = os.environ.get("IMMICH_QUALITY_ENABLED", "true").lower() != "false"

SHOW_NAMES = os.environ.get("IMMICH_SHOW_NAMES", "true").lower() != "false"
LABEL_FONT_SIZE = int(os.environ.get("IMMICH_LABEL_FONT_SIZE", "18"))
LABEL_PADDING_X = int(os.environ.get("IMMICH_LABEL_PADDING_X", "4"))
LABEL_PADDING_Y = int(os.environ.get("IMMICH_LABEL_PADDING_Y", "6"))
LABEL_CORNER = os.environ.get("IMMICH_LABEL_CORNER", "bottom-right").strip().lower()

_VALID_CORNERS = {"top-left", "top-middle", "top-right", "bottom-left", "bottom-middle", "bottom-right"}
if LABEL_CORNER not in _VALID_CORNERS:
    raise SystemExit(f"IMMICH_LABEL_CORNER={LABEL_CORNER!r} must be one of {sorted(_VALID_CORNERS)}")

FIRST_NAME_ONLY = os.environ.get("IMMICH_FIRST_NAME_ONLY", "true").lower() != "false"
LABEL_DELIMITER = os.environ.get("IMMICH_LABEL_DELIMITER", " - ")
LABEL_FONT = os.environ.get("IMMICH_LABEL_FONT", "DejaVuSans-Bold").strip()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_level = getattr(logging, LOG_LEVEL, logging.INFO)

log = logging.getLogger("immich-epaper")
log.setLevel(_level)
log.propagate = False
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    log.addHandler(_h)

logging.getLogger("urllib3").setLevel(logging.WARNING)

if _SMARTCROP is None and _SMARTCROP_ERROR is not None:
    log.info("smartcrop unavailable (%r); IMMICH_CROP=smart will be rejected", _SMARTCROP_ERROR)

log.info("Quality scoring enabled (pyiqa NIMA, BRISQUE, sharpness)")


# ---------------------------------------------------------------------------
# Font Asset Loading
# ---------------------------------------------------------------------------
def _load_font(size: int):
    candidates = [LABEL_FONT, "DejaVuSans"]
    tried = []
    for name in candidates:
        path = f"/usr/share/fonts/truetype/dejavu/{name}.ttf"
        tried.append(path)
        try:
            font = ImageFont.truetype(path, size)
            if name != LABEL_FONT:
                log.warning("IMMICH_LABEL_FONT=%r not found; using %r instead", LABEL_FONT, name)
            return font
        except (OSError, IOError):
            continue
    log.warning("No DejaVu font found (tried %s); using Pillow default.", tried)
    return ImageFont.load_default()


_LABEL_FONT = _load_font(LABEL_FONT_SIZE)
HEADERS = {"x-api-key": API_KEY, "Accept": "application/json"}


# ---------------------------------------------------------------------------
# Helper Utility Functions
# ---------------------------------------------------------------------------
def _iso(d: str, end_of_day: bool) -> str:
    s = d.strip().lower()
    now = datetime.now(timezone.utc)
    edge = "T23:59:59.999Z" if end_of_day else "T00:00:00.000Z"

    if s in ("today", "now"):
        return now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    if s == "yesterday":
        return (now - timedelta(days=1)).strftime("%Y-%m-%d") + edge
    m = re.fullmatch(r"(\d+)\s*d(?:ays?)?(?:\s*ago)?", s)
    if m:
        return (now - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d") + edge
    if "T" in d:
        return d
    return d + edge


def _log_startup_config() -> None:
    masked = f"set(len={len(API_KEY)})" if API_KEY else "(MISSING)"
    items = [
        ("IMMICH_URL", IMMICH_URL),
        ("IMMICH_PUBLIC_URL", IMMICH_PUBLIC_URL),
        ("IMMICH_API_KEY", masked),
        ("IMMICH_ALBUM_ID", ALBUM_ID or "(unset -> library/search mode)"),
        ("IMMICH_INCLUDE_PEOPLE", INCLUDE_PEOPLE or "(none)"),
        ("IMMICH_REQUIRE_ALL_PEOPLE", REQUIRE_ALL_PEOPLE),
        ("IMMICH_EXCLUDE_PEOPLE", sorted(EXCLUDE_PEOPLE) or "(none)"),
        ("IMMICH_DATE_AFTER", DATE_AFTER or "(unset)"),
        ("IMMICH_DATE_BEFORE", DATE_BEFORE or "(unset)"),
        ("IMMICH_SEARCH_BATCH", SEARCH_BATCH),
        ("IMMICH_DEVICE_ORIENTATION", f"{DEVICE_ORIENTATION} ({CANVAS_W}x{CANVAS_H} canvas)"),
        ("IMMICH_DUO_PROBABILITY", DUO_PROBABILITY),
        ("IMMICH_ASSET_ORIENTATION", ASSET_ORIENTATION),
        ("IMMICH_CROP", CROP_MODE),
        ("IMMICH_SHOW_NAMES", SHOW_NAMES),
        ("IMMICH_LABEL_FONT_SIZE", LABEL_FONT_SIZE),
        ("IMMICH_LABEL_FONT", LABEL_FONT),
        ("IMMICH_LABEL_PADDING_X", LABEL_PADDING_X),
        ("IMMICH_LABEL_PADDING_Y", LABEL_PADDING_Y),
        ("IMMICH_LABEL_CORNER", LABEL_CORNER),
        ("IMMICH_FIRST_NAME_ONLY", FIRST_NAME_ONLY),
        ("IMMICH_LABEL_DELIMITER", repr(LABEL_DELIMITER)),
        ("FRAME_ROTATE", ROTATE),
        ("LOG_LEVEL", LOG_LEVEL),
        ("IMMICH_RANKING_BATCH", RANKING_BATCH),
        ("IMMICH_QUALITY_ENABLED", QUALITY_ENABLED),
    ]
    log.info("starting immich-epaper, resolved config:")
    for k, v in items:
        log.info("  %-26s = %s", k, v)
    if DATE_AFTER:
        log.info("  -> takenAfter resolves to  %s", _iso(DATE_AFTER, end_of_day=False))
    if DATE_BEFORE:
        log.info("  -> takenBefore resolves to %s", _iso(DATE_BEFORE, end_of_day=True))


_log_startup_config()

app = Flask(__name__)


def _build_palette_image() -> Image.Image:
    pal = Image.new("P", (1, 1))
    flat = []
    for rgb in PALETTE_RGB:
        flat.extend(rgb)
    flat.extend([0] * (768 - len(flat)))
    pal.putpalette(flat)
    return pal


_PAL_IMG = _build_palette_image()


# ---------------------------------------------------------------------------
# Image Geometry & Cropping Logic
# ---------------------------------------------------------------------------
def _center_fit(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    return ImageOps.fit(img, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _smart_fit(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    if img.width < target_w or img.height < target_h:
        log.info("crop: source %dx%d smaller than %dx%d, falling back to center", img.width, img.height, target_w, target_h)
        return _center_fit(img, target_w, target_h)
    try:
        assert _SMARTCROP is not None
        scale = min(1.0, 1024 / max(img.width, img.height))
        scored = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS) if scale < 1.0 else img

        result = _SMARTCROP.crop(scored, target_w, target_h)
        c = result["top_crop"]
        x, y, w, h = c["x"], c["y"], c["width"], c["height"]
        inv = 1.0 / scale
        box = (int(x * inv), int(y * inv), int((x + w) * inv), int((y + h) * inv))

        log.info("crop: smart picked %s -> %dx%d from %dx%d", box, target_w, target_h, img.width, img.height)
        return img.crop(box).resize((target_w, target_h), Image.Resampling.LANCZOS)
    except Exception as e:
        log.warning("crop: smartcrop failed (%s), falling back to center", e)
        return _center_fit(img, target_w, target_h)


def _fit_to_slot(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    return (_smart_fit if CROP_MODE == "smart" else _center_fit)(img, target_w, target_h)


# ---------------------------------------------------------------------------
# Immich Asset Validation & Processing
# ---------------------------------------------------------------------------
def _not_excluded(asset: dict) -> bool:
    if not EXCLUDE_PEOPLE:
        return True
    for p in asset.get("people", []):
        if p.get("id") in EXCLUDE_PEOPLE:
            return False
    return True


def _asset_display_dims(asset: dict) -> tuple[int, int] | None:
    exif = asset.get("exifInfo") or {}
    w, h = exif.get("exifImageWidth"), exif.get("exifImageHeight")
    if not w or not h:
        return None
    try:
        rot = int(exif.get("orientation") or 1)
    except (TypeError, ValueError):
        rot = 1
    if rot in (5, 6, 7, 8):
        w, h = h, w
    return (w, h)


def _asset_shape(asset: dict) -> str | None:
    dims = _asset_display_dims(asset)
    if dims is None:
        return None
    w, h = dims
    if max(w, h) / min(w, h) <= 1.05:
        return "square"
    return "landscape" if w > h else "portrait"


def _shape_matches(asset: dict, target: str) -> bool:
    if target == "any":
        return True
    shape = _asset_shape(asset)
    return False if shape is None else shape == target


def _asset_names(asset: dict) -> list[str]:
    seen = set()
    out = []
    for p in asset.get("people", []) or []:
        name = (p.get("name") or "").strip()
        if not name:
            continue
        if FIRST_NAME_ONLY:
            name = name.split()[0]
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _search_body() -> dict:
    body = {
        "type": "IMAGE",
        "size": max(1, min(SEARCH_BATCH, 1000)),
        "withArchived": False,
        "withPeople": True,
        "withExif": True,
    }
    if DATE_AFTER:
        body["takenAfter"] = _iso(DATE_AFTER, end_of_day=False)
    if DATE_BEFORE:
        body["takenBefore"] = _iso(DATE_BEFORE, end_of_day=True)
    if INCLUDE_PEOPLE:
        body["personIds"] = INCLUDE_PEOPLE if REQUIRE_ALL_PEOPLE else [random.choice(INCLUDE_PEOPLE)]
    return body


def _pick_asset(shape: str, exclude_ids: set[str] | None = None) -> dict:
    exclude_ids = exclude_ids or set()

    def _shape_filter(a: dict) -> bool:
        if not _shape_matches(a, shape):
            return False
        if ASSET_ORIENTATION != "any" and not _shape_matches(a, ASSET_ORIENTATION):
            return False
        return True

    if ALBUM_ID:
        r = requests.get(f"{IMMICH_URL}/api/albums/{ALBUM_ID}", headers=HEADERS, timeout=30)
        r.raise_for_status()
        raw = [a for a in r.json().get("assets", []) if a.get("type") == "IMAGE"]
        after_excl = [a for a in raw if _not_excluded(a) and a["id"] not in exclude_ids]
        assets = [a for a in after_excl if _shape_filter(a)]
        log.info("pick(%s): album mode, %d images, %d after exclusion, %d after shape", shape, len(raw), len(after_excl), len(assets))
        if not assets:
            raise RuntimeError(f"album has no matching {shape} assets")
        chosen = random.choice(assets)
        log.info(
            "pick(%s): chose %s (%s) %s names=%s  %s/photos/%s",
            shape,
            chosen["id"],
            chosen.get("originalFileName", "?"),
            _asset_display_dims(chosen),
            _asset_names(chosen) or "(none)",
            IMMICH_PUBLIC_URL,
            chosen["id"],
        )
        return chosen

    body = _search_body()
    log.info("pick(%s): search/random body=%s", shape, body)
    r = requests.post(f"{IMMICH_URL}/api/search/random", headers={**HEADERS, "Content-Type": "application/json"}, json=body, timeout=30)
    r.raise_for_status()
    data = r.json()
    raw = data if isinstance(data, list) else data.get("assets", {}).get("items", [])
    raw = [a for a in raw if a.get("type") == "IMAGE"]
    after_excl = [a for a in raw if _not_excluded(a) and a["id"] not in exclude_ids]
    assets = [a for a in after_excl if _shape_filter(a)]
    log.info("pick(%s): search returned %d, %d after exclusion, %d after shape", shape, len(raw), len(after_excl), len(assets))
    if not assets:
        raise RuntimeError(f"no {shape} assets matched the configured filters")
    chosen = random.choice(assets)
    log.info(
        "pick(%s): chose %s (%s) %s names=%s  %s/photos/%s",
        shape,
        chosen["id"],
        chosen.get("originalFileName", "?"),
        _asset_display_dims(chosen),
        _asset_names(chosen) or "(none)",
        IMMICH_PUBLIC_URL,
        chosen["id"],
    )
    return chosen


def _fetch_image(asset_id: str) -> Image.Image:
    t0 = time.monotonic()
    r = requests.get(f"{IMMICH_URL}/api/assets/{asset_id}/thumbnail", headers=HEADERS, params={"size": "preview"}, timeout=60)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    log.info("fetch: asset %s, %d bytes, %dx%d, %.0f ms", asset_id, len(r.content), img.width, img.height, (time.monotonic() - t0) * 1000)
    return img


def _normalize_source(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img) or img
    return img.convert("RGB")


def _pick_layout() -> str:
    if DUO_PROBABILITY <= 0.0 or ASSET_ORIENTATION != "any":
        return "single"
    if DUO_PROBABILITY >= 1.0:
        return "duo"
    return "duo" if random.random() < DUO_PROBABILITY else "single"


# ---------------------------------------------------------------------------
# Drawing & UI Compositions
# ---------------------------------------------------------------------------
def _draw_name_label(canvas: Image.Image, names: list[str], slot_x: int, slot_y: int, slot_w: int, slot_h: int) -> None:
    if not names or not SHOW_NAMES:
        return
    draw = ImageDraw.Draw(canvas)
    text = LABEL_DELIMITER.join(names)
    max_text_w = slot_w - 2 * LABEL_PADDING_X

    if draw.textlength(text, font=_LABEL_FONT) > max_text_w:
        ell = "…"
        while text and draw.textlength(text + ell, font=_LABEL_FONT) > max_text_w:
            text = text[:-1]
        text = text.rstrip(LABEL_DELIMITER) + ell

    tw = draw.textlength(text, font=_LABEL_FONT)
    metrics = getattr(_LABEL_FONT, "getmetrics", lambda: (12, 2))()
    th = metrics[0] + metrics[1]

    rect_w = int(tw) + 2 * LABEL_PADDING_X
    rect_h = th + 2 * LABEL_PADDING_Y

    if LABEL_CORNER.endswith("-left"):
        rect_x = slot_x
    elif LABEL_CORNER.endswith("-middle"):
        rect_x = slot_x + (slot_w - rect_w) // 2
    else:
        rect_x = slot_x + slot_w - rect_w

    if LABEL_CORNER.startswith("top-"):
        rect_y = slot_y
    else:
        rect_y = slot_y + slot_h - rect_h

    draw.rectangle((rect_x, rect_y, rect_x + rect_w, rect_y + rect_h), fill=(255, 255, 255))
    draw.text((rect_x + LABEL_PADDING_X, rect_y + LABEL_PADDING_Y - 1), text, fill=(0, 0, 0), font=_LABEL_FONT)


def _compose_canvas(sources: list[Image.Image], layout: str, assets: list[dict]) -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
    if layout == "single":
        assert len(sources) == 1 and len(assets) == 1
        canvas.paste(_fit_to_slot(sources[0], CANVAS_W, CANVAS_H), (0, 0))
        _draw_name_label(canvas, _asset_names(assets[0]), 0, 0, CANVAS_W, CANVAS_H)
        return canvas

    assert layout == "duo" and len(sources) == 2 and len(assets) == 2
    if DEVICE_ORIENTATION == "landscape":
        slot_w, slot_h = CANVAS_W // 2, CANVAS_H
        canvas.paste(_fit_to_slot(sources[0], slot_w, slot_h), (0, 0))
        canvas.paste(_fit_to_slot(sources[1], slot_w, slot_h), (slot_w, 0))
        _draw_name_label(canvas, _asset_names(assets[0]), 0, 0, slot_w, slot_h)
        _draw_name_label(canvas, _asset_names(assets[1]), slot_w, 0, slot_w, slot_h)
    else:
        slot_w, slot_h = CANVAS_W, CANVAS_H // 2
        canvas.paste(_fit_to_slot(sources[0], slot_w, slot_h), (0, 0))
        canvas.paste(_fit_to_slot(sources[1], slot_w, slot_h), (0, slot_h))
        _draw_name_label(canvas, _asset_names(assets[0]), 0, 0, slot_w, slot_h)
        _draw_name_label(canvas, _asset_names(assets[1]), 0, slot_h, slot_w, slot_h)
    return canvas


def _rotate_final_canvas(canvas: Image.Image) -> Image.Image:
    if ROTATE in (90, 180, 270):
        canvas = canvas.rotate(-ROTATE, expand=True)
    if DEVICE_ORIENTATION == "portrait":
        canvas = canvas.rotate(-90, expand=True)
    assert canvas.size == (PANEL_W, PANEL_H), (canvas.size, (PANEL_W, PANEL_H))
    return canvas


# ---------------------------------------------------------------------------
# Dynamic Quantization Packing Engine
# ---------------------------------------------------------------------------
def _pack_canvas_with_dither_mode(canvas: Image.Image, dither: bool) -> tuple[bytes, Image.Image]:
    if dither:
        quant = canvas.quantize(palette=_PAL_IMG, dither=Image.Dither.FLOYDSTEINBERG)
        preview_source = quant.convert("RGB")
    else:
        quant = canvas.quantize(palette=_PAL_IMG, dither=Image.Dither.NONE)
        preview_source = canvas

    idx = np.asarray(quant, dtype=np.uint8)
    codes = CODE_LUT[idx]
    hi = codes[:, 0::2] << 4
    lo = codes[:, 1::2]
    packed = (hi | lo).astype(np.uint8).tobytes()
    assert len(packed) == FRAME_BYTES, len(packed)

    return packed, preview_source


# ---------------------------------------------------------------------------
# Quality Scoring Analytics (using pyiqa)
# ---------------------------------------------------------------------------
_nima_model = None
_brisque_model = None


def _get_nima_score(img: Image.Image) -> float:
    global _nima_model
    try:
        if _nima_model is None:
            _nima_model = pyiqa.create_metric("nima")
            log.info("pyiqa NIMA model loaded")
        score_tensor = _nima_model(img)
        return max(1.0, min(10.0, score_tensor.cpu().detach().item()))
    except Exception as e:
        log.warning(f"NIMA scoring failed: {e}")
        return 5.0


def _get_brisque_score(img: Image.Image) -> float:
    global _brisque_model
    try:
        if _brisque_model is None:
            _brisque_model = pyiqa.create_metric("brisque")
            log.info("pyiqa BRISQUE model loaded")
        score_tensor = _brisque_model(img)
        return float(score_tensor.cpu().detach().item())
    except Exception as e:
        log.warning(f"BRISQUE scoring failed: {e}")
        return 50.0


def _get_sharpness_score(img: Image.Image) -> float:
    try:
        gray = np.array(img.convert("L"), dtype=np.uint8)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
    except Exception as e:
        log.warning(f"Sharpness scoring failed: {e}")
        return 0.0


def _composite_score(img: Image.Image) -> float:
    nima = _get_nima_score(img)
    brisque = _get_brisque_score(img)
    sharp = _get_sharpness_score(img)

    nima_norm = max(0.0, min(1.0, (nima - 1.0) / 9.0))
    brisque_norm = max(0.0, min(1.0, brisque / 100.0))
    sharp_norm = max(0.0, min(1.0, sharp / 2000.0))

    composite = nima_norm * 0.6 + sharp_norm * 0.3 - brisque_norm * 0.1
    return max(0.0, min(1.0, composite))


# ---------------------------------------------------------------------------
# Core Layout Matrix Builder (Returns clean unquantized RGB canvas)
# ---------------------------------------------------------------------------
def _build_frame() -> tuple[str, str, Image.Image]:
    """Generates the optimal layout and returns a crisp unquantized final composition.
    Returns tuple of (composite_id, layout_type, final_unquantized_canvas_object)."""
    layout = _pick_layout()
    log.info("build: layout=%s (device=%s, duo_prob=%.2f)", layout, DEVICE_ORIENTATION, DUO_PROBABILITY)

    if not QUALITY_ENABLED:
        if layout == "duo":
            a1 = _pick_asset(DUO_SHAPE)
            try:
                a2 = _pick_asset(DUO_SHAPE, exclude_ids={a1["id"]})
            except RuntimeError as e:
                log.warning("duo failed (%s), falling back to single", e)
                layout = "single"
                assets = [_pick_asset(SINGLE_SHAPE)]
            else:
                assets = [a1, a2]
        else:
            assets = [_pick_asset(SINGLE_SHAPE)]

        sources = [_fetch_image(a["id"]) for a in assets]
        sources = [_normalize_source(s) for s in sources]
        canvas = _compose_canvas(sources, layout, assets)
        canvas = _rotate_final_canvas(canvas)
        return "+".join(a["id"] for a in assets), layout, canvas

    candidates = []
    for i in range(RANKING_BATCH):
        try:
            if layout == "duo":
                a1 = _pick_asset(DUO_SHAPE)
                try:
                    a2 = _pick_asset(DUO_SHAPE, exclude_ids={a1["id"]})
                except RuntimeError:
                    log.warning("candidate %d: duo failed, skipping", i + 1)
                    continue
                assets = [a1, a2]
            else:
                assets = [_pick_asset(SINGLE_SHAPE)]

            sources = [_fetch_image(a["id"]) for a in assets]
            canvas = _compose_canvas([_normalize_source(s) for s in sources], layout, assets)
            canvas = _rotate_final_canvas(canvas)

            score = _composite_score(canvas)
            log.info("candidate %d: score=%.4f (assets: %s)", i + 1, score, [a["id"] for a in assets])
            candidates.append((score, assets, canvas))
        except Exception as e:
            log.warning("candidate %d generation failed: %s", i + 1, e)
            continue

    if not candidates:
        log.error("No candidates could be generated; falling back to random single pick")
        assets = [_pick_asset(SINGLE_SHAPE)]
        sources = [_fetch_image(a["id"]) for a in assets]
        canvas = _compose_canvas([_normalize_source(s) for s in sources], "single", assets)
        canvas = _rotate_final_canvas(canvas)
        return assets[0]["id"], "single", canvas

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_assets, best_canvas = candidates[0]
    log.info("best candidate score=%.4f, assets=%s", best_score, [a["id"] for a in best_assets])

    return "+".join(a["id"] for a in best_assets), layout, best_canvas


def _no_store(headers: dict) -> dict:
    return {**headers, "Cache-Control": "no-store, max-age=0"}


# ---------------------------------------------------------------------------
# API Routing Endpoints (Now enforcing format-specific dithering)
# ---------------------------------------------------------------------------
@app.get("/frame.bin")
def frame_bin():
    log.info("request: GET /frame.bin from %s", request.remote_addr)
    t0 = time.monotonic()
    try:
        asset_id, _, canvas = _build_frame()
        # Default hardware fallback dither choice: true
        packed, _ = _pack_canvas_with_dither_mode(canvas, dither=True)
    except Exception as e:
        log.exception("frame.bin failed: %s", e)
        return Response(f"frame generation failed: {e}\n", status=503, mimetype="text/plain")
    log.info("request: served /frame.bin asset %s in %.0f ms", asset_id, (time.monotonic() - t0) * 1000)
    return Response(
        packed,
        mimetype="application/octet-stream",
        headers=_no_store({"X-Immich-Asset-Id": asset_id, "Content-Length": str(FRAME_BYTES)}),
    )


@app.get("/frame.png")
def frame_png():
    log.info("request: GET /frame.png from %s", request.remote_addr)
    try:
        asset_id, _, canvas = _build_frame()
        # PNG explicitly forces dither evaluation
        _, preview_source = _pack_canvas_with_dither_mode(canvas, dither=True)
        png_buf = io.BytesIO()
        preview_source.save(png_buf, format="PNG")
        png_data = png_buf.getvalue()
    except Exception as e:
        log.exception("frame.png failed: %s", e)
        return Response(f"frame generation failed: {e}\n", status=503, mimetype="text/plain")
    return Response(png_data, mimetype="image/png", headers=_no_store({"X-Immich-Asset-Id": asset_id}))


@app.get("/frame.jpg")
def frame_jpg():
    log.info("request: GET /frame.jpg from %s", request.remote_addr)
    try:
        asset_id, _, canvas = _build_frame()
        # JPG explicitly drops dithering rules
        _, preview_source = _pack_canvas_with_dither_mode(canvas, dither=False)
        jpg_buf = io.BytesIO()
        preview_source.save(jpg_buf, format="JPEG", quality=85, optimize=True)
        jpg_data = jpg_buf.getvalue()
    except Exception as e:
        log.exception("frame.jpg failed: %s", e)
        return Response(f"frame generation failed: {e}\n", status=503, mimetype="text/plain")
    return Response(jpg_data, mimetype="image/jpeg", headers=_no_store({"X-Immich-Asset-Id": asset_id}))


@app.get("/healthz")
def healthz():
    return {"ok": True, "album_mode": bool(ALBUM_ID)}


# ---------------------------------------------------------------------------
# Initialization Pre-loading
# ---------------------------------------------------------------------------
def _warmup_quality_models() -> None:
    if not QUALITY_ENABLED:
        return
    t0 = time.monotonic()
    log.info("warmup: loading pyiqa NIMA + BRISQUE models...")
    try:
        dummy = Image.new("RGB", (224, 224), (128, 128, 128))
        _get_nima_score(dummy)
        _get_brisque_score(dummy)
        log.info("warmup: models ready, took %.1f s", time.monotonic() - t0)
    except Exception as e:
        log.warning("warmup: pyiqa preload failed (%s); models will load lazily", e)


_warmup_quality_models()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
