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

# ---------------------------------------------------------------------------
# Global & Hardware Panel Constants
# ---------------------------------------------------------------------------
PANEL_W, PANEL_H = 800, 480  # The panel itself is always landscape-native
FRAME_BYTES = PANEL_W * PANEL_H // 2  # 192000 bytes
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
# Environment & App Configuration
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
DEFAULT_LOG_LEVEL = getattr(logging, LOG_LEVEL, logging.INFO)

HEADERS = {"x-api-key": API_KEY, "Accept": "application/json"}

# ---------------------------------------------------------------------------
# Logging Infrastructure Setup
# ---------------------------------------------------------------------------
log = logging.getLogger("immich-epaper")
log.setLevel(DEFAULT_LOG_LEVEL)
log.propagate = False
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    log.addHandler(_h)

logging.getLogger("urllib3").setLevel(logging.WARNING)

log.info("Quality scoring enabled (pyiqa NIMA, BRISQUE, sharpness)")

# ---------------------------------------------------------------------------
# Flask Initialization
# ---------------------------------------------------------------------------
app = Flask(__name__)


# ---------------------------------------------------------------------------
# Internal Core Asset & Image State Cache Variables
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


def _build_palette_image() -> Image.Image:
    pal = Image.new("P", (1, 1))
    flat = []
    for rgb in PALETTE_RGB:
        flat.extend(rgb)
    flat.extend([0] * (768 - len(flat)))
    pal.putpalette(flat)
    return pal


_QUANTIZATION_PALETTE = _build_palette_image()


# ---------------------------------------------------------------------------
# Environment & Core Helper Utilities
# ---------------------------------------------------------------------------
def _resolve_api_date(d: str, end_of_day: bool) -> str:
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
        log.info("  -> takenAfter resolves to  %s", _resolve_api_date(DATE_AFTER, end_of_day=False))
    if DATE_BEFORE:
        log.info("  -> takenBefore resolves to %s", _resolve_api_date(DATE_BEFORE, end_of_day=True))


_log_startup_config()


def _add_no_store_headers(headers: dict) -> dict:
    return {**headers, "Cache-Control": "no-store, max-age=0"}


# ---------------------------------------------------------------------------
# Immich API Client & Validation Handlers
# ---------------------------------------------------------------------------
def _is_asset_allowed(asset: dict) -> bool:
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


def _build_search_payload() -> dict:
    body = {
        "type": "IMAGE",
        "size": max(1, min(SEARCH_BATCH, 1000)),
        "withArchived": False,
        "withPeople": True,
        "withExif": True,
    }
    if DATE_AFTER:
        body["takenAfter"] = _resolve_api_date(DATE_AFTER, end_of_day=False)
    if DATE_BEFORE:
        body["takenBefore"] = _resolve_api_date(DATE_BEFORE, end_of_day=True)
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

    # 1. Album Mode
    if ALBUM_ID:
        r = requests.get(f"{IMMICH_URL}/api/albums/{ALBUM_ID}", headers=HEADERS, timeout=30)
        r.raise_for_status()
        raw = [a for a in r.json().get("assets", []) if a.get("type") == "IMAGE"]
        after_excl = [a for a in raw if _is_asset_allowed(a) and a["id"] not in exclude_ids]
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

    # 2. Search Mode (with automatic fallback tiering)
    body = _build_search_payload()

    for pass_num in ("strict", "fallback"):
        if pass_num == "fallback":
            log.warning("pick(%s): Strict filters returned 0 assets. Dropping constraints for a wide safety sweep.", shape)
            body = {"type": "IMAGE", "size": 50, "withArchived": False, "withPeople": True, "withExif": True}

        log.info("pick(%s): executing search (mode=%s)", shape, pass_num)
        r = requests.post(f"{IMMICH_URL}/api/search/random", headers={**HEADERS, "Content-Type": "application/json"}, json=body, timeout=30)
        r.raise_for_status()

        data = r.json()
        raw = data if isinstance(data, list) else data.get("assets", {}).get("items", [])
        raw = [a for a in raw if a.get("type") == "IMAGE"]

        if raw:
            break
    else:
        raise RuntimeError("Immich instance returned zero total image assets during fallback search sweep.")

    # 3. Apply Filters & Layout Shape Recovery
    after_excl = [a for a in raw if _is_asset_allowed(a) and a["id"] not in exclude_ids]
    assets = [a for a in after_excl if _shape_filter(a)]

    log.info("pick(%s): search returned %d, %d after exclusion, %d after shape.", shape, len(raw), len(after_excl), len(assets))

    if not assets:
        log.warning("pick(%s): No assets matched shape constraint. Disregarding shape rules to preserve layout cycle.", shape)
        chosen = random.choice(after_excl) if after_excl else random.choice(raw)
    else:
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


# ---------------------------------------------------------------------------
# Image Geometry & Face-Aware Cropping Logic
# ---------------------------------------------------------------------------
def _center_fit(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    log.info("crop: center fit (%dx%d -> %dx%d)", img.width, img.height, target_w, target_h)
    return ImageOps.fit(img, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _get_asset_faces(asset_id: str) -> list[dict]:
    try:
        r = requests.get(f"{IMMICH_URL}/api/assets/{asset_id}", headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        extracted_faces = []
        for person in data.get("people", []) or []:
            faces = person.get("faces", []) or []
            extracted_faces.extend(faces)
        unassigned = data.get("unassignedFaces", []) or []
        extracted_faces.extend(unassigned)
        log.info("found %d face(s) for asset %s", len(extracted_faces), asset_id)
        return extracted_faces
    except Exception as e:
        log.warning("Failed to fetch face details for asset %s: %s", asset_id, e)
        return []


def _smart_face_fit(img: Image.Image, asset_id: str, target_w: int, target_h: int) -> Image.Image:
    faces = _get_asset_faces(asset_id)
    if not faces:
        log.info("crop: no faces returned for %s, falling back to center", asset_id)
        return _center_fit(img, target_w, target_h)

    img_w, img_h = img.size
    x_coords: list[float] = []
    y_coords: list[float] = []

    for face in faces:
        x1 = face.get("boundingBoxX1")
        y1 = face.get("boundingBoxY1")
        x2 = face.get("boundingBoxX2")
        y2 = face.get("boundingBoxY2")
        face_img_w = face.get("imageWidth")
        face_img_h = face.get("imageHeight")

        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue
        if not face_img_w or not face_img_h:
            continue

        scale_x = img_w / float(face_img_w)
        scale_y = img_h / float(face_img_h)

        x_coords.extend([float(x1) * scale_x, float(x2) * scale_x])
        y_coords.extend([float(y1) * scale_y, float(y2) * scale_y])

    if not x_coords or not y_coords:
        log.info("crop: no coordinates found for %s, falling back to center", asset_id)
        return _center_fit(img, target_w, target_h)

    f_min_x, f_max_x = min(x_coords), max(x_coords)
    f_min_y, f_max_y = min(y_coords), max(y_coords)

    f_w = f_max_x - f_min_x
    f_h = f_max_y - f_min_y
    f_center_x = f_min_x + (f_w / 2)
    f_center_y = f_min_y + (f_h / 2)

    target_aspect = target_w / target_h
    PADDING_MULTIPLIER = 2.5

    desired_w = f_w * PADDING_MULTIPLIER
    desired_h = f_h * PADDING_MULTIPLIER

    if desired_w / target_aspect > desired_h:
        desired_h = desired_w / target_aspect
    else:
        desired_w = desired_h * target_aspect

    scale = min(1.0, img_w / desired_w, img_h / desired_h)
    crop_w = int(desired_w * scale)
    crop_h = int(desired_h * scale)

    min_short_edge = min(img_w, img_h) * 0.60
    if crop_w / target_aspect < min_short_edge or crop_h < min_short_edge:
        if img_w / target_aspect > img_h:
            crop_h = int(img_h)
            crop_w = int(crop_h * target_aspect)
        else:
            crop_w = int(img_w)
            crop_h = int(crop_w / target_aspect)

    left = int(f_center_x - (crop_w / 2))
    top = int(f_center_y - (crop_h / 2))

    left = max(0, min(left, img_w - crop_w))
    top = max(0, min(top, img_h - crop_h))

    log.info("crop: aesthetic face-aware crop frame picked (%d, %d, %d, %d) for asset %s", left, top, left + crop_w, top + crop_h, asset_id)

    return img.crop((left, top, left + crop_w, top + crop_h)).resize((target_w, target_h), Image.Resampling.LANCZOS)


def _fit_to_slot(img: Image.Image, asset_id: str, target_w: int, target_h: int) -> Image.Image:
    if CROP_MODE == "smart":
        return _smart_face_fit(img, asset_id, target_w, target_h)
    return _center_fit(img, target_w, target_h)


# ---------------------------------------------------------------------------
# Drawing, Text & Canvas UI Layout Compositions
# ---------------------------------------------------------------------------
def _pick_layout() -> str:
    if DUO_PROBABILITY <= 0.0 or ASSET_ORIENTATION != "any":
        return "single"
    if DUO_PROBABILITY >= 1.0:
        return "duo"
    return "duo" if random.random() < DUO_PROBABILITY else "single"


def _get_year(asset: dict) -> str:
    dt_str = asset.get("fileCreatedAt") or asset.get("exifInfo", {}).get("dateTimeOriginal") or ""
    return dt_str[:4] if len(dt_str) >= 4 and dt_str[:4].isdigit() else ""


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
        canvas.paste(_fit_to_slot(sources[0], assets[0]["id"], CANVAS_W, CANVAS_H), (0, 0))

        label_elements = _asset_names(assets[0])
        year = _get_year(assets[0])
        if year:
            label_elements.append(year)

        _draw_name_label(canvas, label_elements, 0, 0, CANVAS_W, CANVAS_H)
        return canvas

    assert layout == "duo" and len(sources) == 2 and len(assets) == 2
    if DEVICE_ORIENTATION == "landscape":
        slot_w, slot_h = CANVAS_W // 2, CANVAS_H
        canvas.paste(_fit_to_slot(sources[0], assets[0]["id"], slot_w, slot_h), (0, 0))
        canvas.paste(_fit_to_slot(sources[1], assets[1]["id"], slot_w, slot_h), (slot_w, 0))

        labels_0 = _asset_names(assets[0])
        year_0 = _get_year(assets[0])
        if year_0:
            labels_0.append(year_0)

        labels_1 = _asset_names(assets[1])
        year_1 = _get_year(assets[1])
        if year_1:
            labels_1.append(year_1)

        _draw_name_label(canvas, labels_0, 0, 0, slot_w, slot_h)
        _draw_name_label(canvas, labels_1, slot_w, 0, slot_w, slot_h)
    else:
        slot_w, slot_h = CANVAS_W, CANVAS_H // 2
        canvas.paste(_fit_to_slot(sources[0], assets[0]["id"], slot_w, slot_h), (0, 0))
        canvas.paste(_fit_to_slot(sources[1], assets[1]["id"], slot_w, slot_h), (0, slot_h))

        labels_0 = _asset_names(assets[0])
        year_0 = _get_year(assets[0])
        if year_0:
            labels_0.append(year_0)

        labels_1 = _asset_names(assets[1])
        year_1 = _get_year(assets[1])
        if year_1:
            labels_1.append(year_1)

        _draw_name_label(canvas, labels_0, 0, 0, slot_w, slot_h)
        _draw_name_label(canvas, labels_1, 0, slot_h, slot_w, slot_h)

    return canvas


def _rotate_final_canvas(canvas: Image.Image) -> Image.Image:
    if ROTATE in (90, 180, 270):
        canvas = canvas.rotate(-ROTATE, expand=True)
    if DEVICE_ORIENTATION == "portrait":
        canvas = canvas.rotate(-90, expand=True)
    assert canvas.size == (PANEL_W, PANEL_H), (canvas.size, (PANEL_W, PANEL_H))
    return canvas


# ---------------------------------------------------------------------------
# Pyiqa Image Quality Scoring Analytics
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
# Quantization Packing & Frame Assembly Engines
# ---------------------------------------------------------------------------
def _quantize_and_pack_frame(canvas: Image.Image, enable_dither: bool) -> tuple[bytes, Image.Image]:
    if enable_dither:
        quant = canvas.quantize(palette=_QUANTIZATION_PALETTE, dither=Image.Dither.FLOYDSTEINBERG)
        preview_source = quant.convert("RGB")
    else:
        quant = canvas.quantize(palette=_QUANTIZATION_PALETTE, dither=Image.Dither.NONE)
        preview_source = canvas

    idx = np.asarray(quant, dtype=np.uint8)
    codes = CODE_LUT[idx]
    hi = codes[:, 0::2] << 4
    lo = codes[:, 1::2]
    packed = (hi | lo).astype(np.uint8).tobytes()
    assert len(packed) == FRAME_BYTES, len(packed)

    return packed, preview_source


def _build_frame() -> tuple[str, str, Image.Image]:
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


# ---------------------------------------------------------------------------
# Flask Routing Endpoints
# ---------------------------------------------------------------------------
@app.get("/frame.bin")
def frame_bin():
    log.info("request: GET /frame.bin from %s", request.remote_addr)
    t0 = time.monotonic()
    try:
        asset_id, _, canvas = _build_frame()
        packed, _ = _quantize_and_pack_frame(canvas, enable_dither=True)
    except Exception as e:
        log.exception("frame.bin failed: %s", e)
        return Response(f"frame generation failed: {e}\n", status=503, mimetype="text/plain")
    log.info("request: served /frame.bin asset %s in %.0f ms", asset_id, (time.monotonic() - t0) * 1000)
    return Response(
        packed,
        mimetype="application/octet-stream",
        headers=_add_no_store_headers({"X-Immich-Asset-Id": asset_id, "Content-Length": str(FRAME_BYTES)}),
    )


@app.get("/frame.png")
def frame_png():
    log.info("request: GET /frame.png from %s", request.remote_addr)
    try:
        asset_id, _, canvas = _build_frame()
        _, preview_source = _quantize_and_pack_frame(canvas, enable_dither=True)
        png_buf = io.BytesIO()
        preview_source.save(png_buf, format="PNG")
        png_data = png_buf.getvalue()
    except Exception as e:
        log.exception("frame.png failed: %s", e)
        return Response(f"frame generation failed: {e}\n", status=503, mimetype="text/plain")
    return Response(png_data, mimetype="image/png", headers=_add_no_store_headers({"X-Immich-Asset-Id": asset_id}))


@app.get("/frame.jpg")
def frame_jpg():
    log.info("request: GET /frame.jpg from %s", request.remote_addr)
    try:
        asset_id, _, canvas = _build_frame()
        _, preview_source = _quantize_and_pack_frame(canvas, enable_dither=False)
        jpg_buf = io.BytesIO()
        preview_source.save(jpg_buf, format="JPEG", quality=85, optimize=True)
        jpg_data = jpg_buf.getvalue()
    except Exception as e:
        log.exception("frame.jpg failed: %s", e)
        return Response(f"frame generation failed: {e}\n", status=503, mimetype="text/plain")
    return Response(jpg_data, mimetype="image/jpeg", headers=_add_no_store_headers({"X-Immich-Asset-Id": asset_id}))


@app.get("/healthz")
def healthz():
    return {"ok": True, "album_mode": bool(ALBUM_ID)}


# ---------------------------------------------------------------------------
# Application Server & Pre-warm Startup Process
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
