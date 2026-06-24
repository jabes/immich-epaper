"""
immich-epaper: serve a pre-dithered, pre-packed framebuffer for a
Waveshare 7.3" Spectra 6 (E6 / epd7in3e) panel, sourced from Immich.

Endpoints:
  GET /frame.bin   -> 192000 bytes, the exact buffer EPD_7IN3E_Display() wants
                      (800x480, 4 bits/pixel, 2 pixels/byte, high nibble = left pixel)
  GET /frame.png   -> the same image as PNG, lossless. Best for visual inspection.
  GET /frame.jpg   -> the same image as JPEG (quality 85). Much smaller than PNG;
                      preferred for memory-constrained clients (e.g. the ESP32
                      photoframe firmware in URL-fetch mode).
  GET /healthz     -> liveness

Query params (both image endpoints):
  ?fresh=1         -> bypass the once-per-day cache and pick a new image now

Config via environment:
  IMMICH_URL       e.g. http://immich-server:2283   (no trailing /api)
  IMMICH_API_KEY   key with at least asset.read (+ album.read if you use an album)
  IMMICH_ALBUM_ID  optional; if set, pick randomly from this album instead of the
                   whole library. This also sidesteps the server-side /search/random
                   "same asset every call" caching bug seen in some Immich versions,
                   because the random choice happens here.
  FRAME_ROTATE     0|90|180|270, default 0 (rotate the final image before packing)
  IMMICH_DITHER    true (default) | false. Set false if your client (e.g.
                   aitjcize/esp32-photoframe firmware in URL-fetch mode) does its
                   own dithering on-device. When false, /frame.png is the cropped
                   RGB image with no palette quantization, and /frame.bin uses
                   nearest-colour mapping without dither.

  Layout — how the panel is mounted and whether to compose multiple assets:
  IMMICH_DEVICE_ORIENTATION   landscape (default) | portrait
                              Affects both the canvas dimensions and the duo layout
                              direction: landscape = 800x480, two portraits placed
                              side-by-side; portrait = 480x800, two landscapes
                              stacked top/bottom.
  IMMICH_DUO_PROBABILITY      0.0..1.0, default 0.5. Each refresh, this is the
                              probability of composing two opposite-orientation
                              assets into the frame instead of using one matching
                              the device orientation. 0 disables duo entirely;
                              1.0 forces duo every time.
  IMMICH_ASSET_ORIENTATION    any | landscape | portrait | square. A manual
                              override that filters every pick to assets of this
                              shape. Default 'any', which lets the duo logic do
                              its job. Set this only if you want to force "always
                              landscape" or similar — it bypasses the duo logic.

  Kiosk-style filters (apply to the search path, i.e. when IMMICH_ALBUM_ID is unset;
  EXCLUDE_PEOPLE also applies in album mode):
  IMMICH_INCLUDE_PEOPLE     comma-separated person UUIDs to include
  IMMICH_REQUIRE_ALL_PEOPLE true  -> asset must contain ALL of them (personIds is AND)
                            false -> any of them (picks one at random per fetch)
  IMMICH_EXCLUDE_PEOPLE    comma-separated person UUIDs to drop (filtered locally;
                            Immich has no native exclude filter)
  IMMICH_DATE_AFTER         e.g. 2021-01-01 (-> takenAfter)
  IMMICH_DATE_BEFORE        e.g. 2024-12-31, or the literal "today" (-> takenBefore)
  IMMICH_SEARCH_BATCH       candidates pulled per fetch for local filtering (default 250)
"""

import io
import logging
import os
import random
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests
from flask import Flask, Response, request
from PIL import Image, ImageOps

try:
    import smartcrop as _smartcrop  # type: ignore[import-not-found]

    _SMARTCROP = _smartcrop.SmartCrop()
    _SMARTCROP_ERROR = None
except Exception as e:  # pragma: no cover
    _smartcrop = None
    _SMARTCROP = None
    _SMARTCROP_ERROR = e

# ---------------------------------------------------------------------------
# Panel definition. These are the authoritative epd7in3e (E6) values.
# Palette order below is fixed; CODE_LUT maps each palette index to the
# panel's 4-bit colour code. Note the gap: 0x4 is unused on the E6 (it was
# orange on the 7-colour "F" panel), so blue/green are 0x5/0x6.
# ---------------------------------------------------------------------------
PANEL_W, PANEL_H = 800, 480  # The panel itself is always landscape-native.
FRAME_BYTES = PANEL_W * PANEL_H // 2  # 192000

# WIDTH/HEIGHT are kept for legacy callers but track the *canvas* we compose to.
# When DEVICE_ORIENTATION=portrait, the canvas is 480x800 and the final image is
# rotated 90 degrees before packing so the panel still receives an 800x480 buffer.
WIDTH, HEIGHT = PANEL_W, PANEL_H

# RGB targets used for the nearest-colour mapping. These are deliberately a bit
# muted vs pure primaries because the panel pigments are not saturated; tune
# them against your actual unit by viewing /frame.png.
PALETTE_RGB = [
    (0, 0, 0),  # black
    (255, 255, 255),  # white
    (255, 243, 56),  # yellow
    (191, 35, 35),  # red
    (45, 65, 170),  # blue
    (35, 130, 75),  # green
]
CODE_LUT = np.array([0x0, 0x1, 0x2, 0x3, 0x5, 0x6], dtype=np.uint8)

IMMICH_URL = os.environ.get("IMMICH_URL", "http://immich-server:2283").rstrip("/")
# Used only for building human-clickable links in the logs (e.g. your public
# https://immich.example.com). Falls back to IMMICH_URL, which is fine if you
# also reach Immich at that address from a browser; not fine if IMMICH_URL is
# the internal docker-network address (http://immich-server:2283), in which
# case set this to whatever you type into your browser.
IMMICH_PUBLIC_URL = os.environ.get("IMMICH_PUBLIC_URL", IMMICH_URL).rstrip("/")
API_KEY = os.environ["IMMICH_API_KEY"]
ALBUM_ID = os.environ.get("IMMICH_ALBUM_ID", "").strip()
ROTATE = int(os.environ.get("FRAME_ROTATE", "0"))
# Set IMMICH_CACHE=false to recompute (new random pick + fetch + dither) on every
# request instead of serving one stable image per calendar day.
CACHE_ENABLED = os.environ.get("IMMICH_CACHE", "true").lower() != "false"
# Set IMMICH_DITHER=false when the client (e.g. aitjcize/esp32-photoframe)
# applies its own dithering. The crop still happens server-side; what changes is
# whether we quantize+dither to the 6-colour panel palette before serving.
DITHER_ENABLED = os.environ.get("IMMICH_DITHER", "true").lower() != "false"


def _csv_env(name: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


# Kiosk-style filters (search/random path only; album mode is its own curation,
# matching Kiosk's "filter_date only applies to people and random assets").
INCLUDE_PEOPLE = _csv_env("IMMICH_INCLUDE_PEOPLE")  # person UUIDs to include
EXCLUDE_PEOPLE = set(_csv_env("IMMICH_EXCLUDE_PEOPLE"))  # person UUIDs to drop
REQUIRE_ALL_PEOPLE = (
    os.environ.get("IMMICH_REQUIRE_ALL_PEOPLE", "false").lower() == "true"
)
DATE_AFTER = os.environ.get("IMMICH_DATE_AFTER", "").strip()  # e.g. 2021-01-01
DATE_BEFORE = os.environ.get("IMMICH_DATE_BEFORE", "").strip()  # date, or "today"
# How many candidates to pull per fetch so local exclusion + random pick have
# something to work with (1..1000).
SEARCH_BATCH = int(os.environ.get("IMMICH_SEARCH_BATCH", "250"))

# Device orientation determines the canvas shape and the duo direction.
DEVICE_ORIENTATION = (
    os.environ.get("IMMICH_DEVICE_ORIENTATION", "landscape").strip().lower()
)
if DEVICE_ORIENTATION not in {"landscape", "portrait"}:
    raise SystemExit(
        f"IMMICH_DEVICE_ORIENTATION={DEVICE_ORIENTATION!r} must be landscape|portrait"
    )

# Canvas dimensions match the device orientation; we rotate at the end so the
# panel always receives an 800x480 packed buffer.
if DEVICE_ORIENTATION == "landscape":
    CANVAS_W, CANVAS_H = PANEL_W, PANEL_H  # 800x480
    SINGLE_SHAPE = "landscape"
    DUO_SHAPE = "portrait"
else:
    CANVAS_W, CANVAS_H = PANEL_H, PANEL_W  # 480x800
    SINGLE_SHAPE = "portrait"
    DUO_SHAPE = "landscape"

# Probability per refresh of composing two DUO_SHAPE assets instead of one
# SINGLE_SHAPE asset. 0 disables the feature; 1.0 always composes a duo.
try:
    DUO_PROBABILITY = float(os.environ.get("IMMICH_DUO_PROBABILITY", "0.5"))
except ValueError:
    raise SystemExit("IMMICH_DUO_PROBABILITY must be a float between 0 and 1")
if not 0.0 <= DUO_PROBABILITY <= 1.0:
    raise SystemExit(
        f"IMMICH_DUO_PROBABILITY={DUO_PROBABILITY} must be between 0 and 1"
    )

# Manual override that forces every pick to a single shape. 'any' (default)
# lets the duo logic operate normally. Set to 'landscape' if you want to
# disable portraits entirely, etc. Setting this to anything other than 'any'
# effectively disables duo composition.
ASSET_ORIENTATION = os.environ.get("IMMICH_ASSET_ORIENTATION", "any").strip().lower()
if ASSET_ORIENTATION not in {"any", "landscape", "portrait", "square"}:
    raise SystemExit(
        f"IMMICH_ASSET_ORIENTATION={ASSET_ORIENTATION!r} must be any|landscape|portrait|square"
    )

# center | smart. smart uses smartcrop.py (edge/saturation/skin heuristics) to
# pick the most "interesting" 800x480 window from the source, then falls back to
# center-crop on any error or when the source is smaller than the target.
CROP_MODE = os.environ.get("IMMICH_CROP", "center").strip().lower()
if CROP_MODE not in {"center", "smart"}:
    raise SystemExit(f"IMMICH_CROP={CROP_MODE!r} must be center|smart")
if CROP_MODE == "smart" and _SMARTCROP is None:
    raise SystemExit(
        f"IMMICH_CROP=smart but smartcrop import failed: {_SMARTCROP_ERROR!r}"
    )

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_level = getattr(logging, LOG_LEVEL, logging.INFO)

# Don't rely on logging.basicConfig(): under gunicorn it's effectively a no-op
# (gunicorn has already configured the logging machinery), so app logs never
# reach `docker logs` and you only see gunicorn's own lines. Instead attach our
# own stdout handler to this logger and turn off propagation so there are no
# duplicates regardless of what gunicorn did to the root logger.
log = logging.getLogger("immich-epaper")
log.setLevel(_level)
log.propagate = False
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    log.addHandler(_h)

# requests/urllib3 are noisy at DEBUG; keep them at WARNING unless explicitly wanted.
logging.getLogger("urllib3").setLevel(logging.WARNING)

# If smartcrop was wanted but failed to import, note it so a startup install
# issue (vs an intentional absence) is debuggable from the logs.
if _SMARTCROP is None and _SMARTCROP_ERROR is not None:
    log.info(
        "smartcrop unavailable (%r); IMMICH_CROP=smart will be rejected",
        _SMARTCROP_ERROR,
    )

HEADERS = {"x-api-key": API_KEY, "Accept": "application/json"}


def _iso(d: str, end_of_day: bool) -> str:
    """Resolve a date spec to ISO-8601 Z.

    Accepts: 'today'/'now' (current instant), 'yesterday', 'N days ago' / 'Nd',
    a plain date ('2021-01-01'), or a full ISO string (passed through).
    Relative day keywords anchor to the start of day (for *After*) or end of day
    (for *Before*), per end_of_day.
    """
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
        (
            "IMMICH_DEVICE_ORIENTATION",
            f"{DEVICE_ORIENTATION} ({CANVAS_W}x{CANVAS_H} canvas)",
        ),
        ("IMMICH_DUO_PROBABILITY", DUO_PROBABILITY),
        ("IMMICH_ASSET_ORIENTATION", ASSET_ORIENTATION),
        ("IMMICH_CROP", CROP_MODE),
        ("IMMICH_DITHER", DITHER_ENABLED),
        ("IMMICH_CACHE", CACHE_ENABLED),
        ("FRAME_ROTATE", ROTATE),
        ("LOG_LEVEL", LOG_LEVEL),
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

# date_str -> {"id": str, "bin": bytes, "png": bytes, "jpg": bytes}
# Keyed by UTC date (the container's clock is almost always UTC; the cache
# would flip at local midnight if TZ is set, which is fine but not what callers
# would intuit). The day boundary is mostly cosmetic — most consumers care about
# "the same image for one wake-up cycle", not strict calendar alignment.
_cache: dict[str, dict] = {}


def _build_palette_image() -> Image.Image:
    pal = Image.new("P", (1, 1))
    flat: list[int] = []
    for rgb in PALETTE_RGB:
        flat.extend(rgb)
    flat.extend([0] * (768 - len(flat)))  # pad to 256 entries
    pal.putpalette(flat)
    return pal


_PAL_IMG = _build_palette_image()


def _center_fit(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Cover-crop to exactly target_w x target_h, centred. The fallback for both
    smart-crop failure and as the default crop mode."""
    return ImageOps.fit(
        img, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
    )


def _smart_fit(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """target_w x target_h crop chosen by smartcrop.py, then resampled.

    Falls back to center-fit on any failure (source too small, smartcrop error)
    so a bad image can never wedge the whole pipeline.
    """
    if img.width < target_w or img.height < target_h:
        log.info(
            "crop: source %dx%d smaller than %dx%d, falling back to center",
            img.width,
            img.height,
            target_w,
            target_h,
        )
        return _center_fit(img, target_w, target_h)
    try:
        assert _SMARTCROP is not None
        scale = min(1.0, 1024 / max(img.width, img.height))
        if scale < 1.0:
            scored = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.Resampling.LANCZOS,
            )
        else:
            scored = img
        result = _SMARTCROP.crop(scored, target_w, target_h)
        c = result["top_crop"]
        x, y, w, h = c["x"], c["y"], c["width"], c["height"]
        inv = 1.0 / scale
        box = (int(x * inv), int(y * inv), int((x + w) * inv), int((y + h) * inv))
        cropped = img.crop(box)
        log.info(
            "crop: smart picked %s -> %dx%d from %dx%d",
            box,
            target_w,
            target_h,
            img.width,
            img.height,
        )
        return cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
    except Exception as e:
        log.warning("crop: smartcrop failed (%s), falling back to center", e)
        return _center_fit(img, target_w, target_h)


def _fit_to_slot(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Apply the configured crop mode at the requested slot dimensions."""
    return (_smart_fit if CROP_MODE == "smart" else _center_fit)(
        img, target_w, target_h
    )


def _not_excluded(asset: dict) -> bool:
    if not EXCLUDE_PEOPLE:
        return True
    for p in asset.get("people", []):
        if p.get("id") in EXCLUDE_PEOPLE:
            return False
    return True


def _asset_display_dims(asset: dict) -> tuple[int, int] | None:
    """Return (display_width, display_height) for an asset, accounting for EXIF
    orientation. Returns None if EXIF dims are missing.

    NB: exifImageWidth/Height are the *sensor* dimensions. A portrait phone photo
    is stored 4032x3024 (landscape sensor) with EXIF Orientation tag = 6/8,
    meaning "rotate 90°/270° for display". Values 5-8 all rotate by a quarter
    turn, so we have to swap W/H to get the display dims.
    """
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
    """Return one of 'landscape' | 'portrait' | 'square', or None if unclassifiable."""
    dims = _asset_display_dims(asset)
    if dims is None:
        return None
    w, h = dims
    longer, shorter = max(w, h), min(w, h)
    if longer / shorter <= 1.05:
        return "square"
    return "landscape" if w > h else "portrait"


def _shape_matches(asset: dict, target: str) -> bool:
    """True if asset matches target shape. Assets without EXIF dims are dropped
    when target is anything other than 'any' — safer than guessing wrong."""
    if target == "any":
        return True
    shape = _asset_shape(asset)
    if shape is None:
        return False  # no EXIF dims, refuse to classify
    return shape == target


def _search_body() -> dict:
    body: dict = {
        "type": "IMAGE",
        "size": max(1, min(SEARCH_BATCH, 1000)),
        "withArchived": False,
        # needed so we can apply exclude_people locally
        "withPeople": bool(EXCLUDE_PEOPLE),
        # Always request EXIF: even when ASSET_ORIENTATION='any', the duo picker
        # needs display dims to classify candidates.
        "withExif": True,
    }
    if DATE_AFTER:
        body["takenAfter"] = _iso(DATE_AFTER, end_of_day=False)
    if DATE_BEFORE:
        body["takenBefore"] = _iso(DATE_BEFORE, end_of_day=True)
    if INCLUDE_PEOPLE:
        if REQUIRE_ALL_PEOPLE:
            body["personIds"] = INCLUDE_PEOPLE  # AND: asset must contain all
        else:
            body["personIds"] = [random.choice(INCLUDE_PEOPLE)]  # OR: one per fetch
    return body


def _pick_asset(shape: str, exclude_ids: set[str] | None = None) -> dict:
    """Pick one asset matching the requested shape ('landscape'/'portrait'/'square'/'any').

    Returns the full asset dict (so the caller has the id, filename, exif, etc).
    exclude_ids is honoured to prevent picking the same asset twice in a duo
    composition.

    Honours ASSET_ORIENTATION as an additional filter: if it's set to something
    other than 'any', the picked asset must satisfy BOTH the requested shape AND
    the configured override. In practice setting ASSET_ORIENTATION makes duo
    layouts impossible — by design, it's the "force single shape" escape hatch.
    """
    exclude_ids = exclude_ids or set()

    def _shape_filter(a: dict) -> bool:
        if not _shape_matches(a, shape):
            return False
        # Manual override: AND with shape.
        if ASSET_ORIENTATION != "any" and not _shape_matches(a, ASSET_ORIENTATION):
            return False
        return True

    if ALBUM_ID:
        r = requests.get(
            f"{IMMICH_URL}/api/albums/{ALBUM_ID}", headers=HEADERS, timeout=30
        )
        r.raise_for_status()
        raw = [a for a in r.json().get("assets", []) if a.get("type") == "IMAGE"]
        after_excl = [a for a in raw if _not_excluded(a) and a["id"] not in exclude_ids]
        assets = [a for a in after_excl if _shape_filter(a)]
        log.info(
            "pick(%s): album mode, %d images, %d after exclusion, %d after shape",
            shape,
            len(raw),
            len(after_excl),
            len(assets),
        )
        if not assets:
            raise RuntimeError(f"album has no matching {shape} assets")
        chosen = random.choice(assets)
        log.info(
            "pick(%s): chose %s (%s) %s  %s/photos/%s",
            shape,
            chosen["id"],
            chosen.get("originalFileName", "?"),
            _asset_display_dims(chosen),
            IMMICH_PUBLIC_URL,
            chosen["id"],
        )
        return chosen

    body = _search_body()
    log.info("pick(%s): search/random body=%s", shape, body)
    r = requests.post(
        f"{IMMICH_URL}/api/search/random",
        headers={**HEADERS, "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    raw = data if isinstance(data, list) else data.get("assets", {}).get("items", [])
    raw = [a for a in raw if a.get("type") == "IMAGE"]
    after_excl = [a for a in raw if _not_excluded(a) and a["id"] not in exclude_ids]
    assets = [a for a in after_excl if _shape_filter(a)]
    log.info(
        "pick(%s): search returned %d, %d after exclusion, %d after shape",
        shape,
        len(raw),
        len(after_excl),
        len(assets),
    )
    if not assets:
        raise RuntimeError(f"no {shape} assets matched the configured filters")
    chosen = random.choice(assets)
    log.info(
        "pick(%s): chose %s (%s) %s  %s/photos/%s",
        shape,
        chosen["id"],
        chosen.get("originalFileName", "?"),
        _asset_display_dims(chosen),
        IMMICH_PUBLIC_URL,
        chosen["id"],
    )
    return chosen


def _fetch_image(asset_id: str) -> Image.Image:
    # "preview" is always a JPEG and large enough (long edge ~1440px), so we
    # never have to deal with HEIC decoding on this side.
    t0 = time.monotonic()
    r = requests.get(
        f"{IMMICH_URL}/api/assets/{asset_id}/thumbnail",
        headers=HEADERS,
        params={"size": "preview"},
        timeout=60,
    )
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    log.info(
        "fetch: asset %s, %d bytes, %dx%d, %.0f ms",
        asset_id,
        len(r.content),
        img.width,
        img.height,
        (time.monotonic() - t0) * 1000,
    )
    return img


def _normalize_source(img: Image.Image) -> Image.Image:
    """Apply EXIF transpose and convert to RGB. Always do this before any
    geometric work so per-photo rotation is honoured."""
    img = ImageOps.exif_transpose(img) or img
    return img.convert("RGB")


def _pick_layout() -> str:
    """Decide whether this refresh is a 'single' (one SINGLE_SHAPE asset) or
    'duo' (two DUO_SHAPE assets composed)."""
    if DUO_PROBABILITY <= 0.0 or ASSET_ORIENTATION != "any":
        # Forced single: either duo is disabled, or the manual override is on
        # (and would make duo impossible anyway).
        return "single"
    if DUO_PROBABILITY >= 1.0:
        return "duo"
    return "duo" if random.random() < DUO_PROBABILITY else "single"


def _compose_canvas(sources: list[Image.Image], layout: str) -> Image.Image:
    """Build the final RGB canvas (CANVAS_W x CANVAS_H) before quantization.

    layout='single': one source filling the entire canvas.
    layout='duo': two sources split along the canvas's *long* axis. For a
    landscape canvas (800x480), portraits sit side-by-side at 400x480 each.
    For a portrait canvas (480x800), landscapes stack at 480x400 each.
    """
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
    if layout == "single":
        assert len(sources) == 1
        canvas.paste(_fit_to_slot(sources[0], CANVAS_W, CANVAS_H), (0, 0))
        return canvas

    assert layout == "duo" and len(sources) == 2
    if DEVICE_ORIENTATION == "landscape":
        # Two portrait slots side-by-side. Splitting 800 in two gives 400-wide
        # slots; if you ever want a gap between, subtract a margin here.
        slot_w, slot_h = CANVAS_W // 2, CANVAS_H
        canvas.paste(_fit_to_slot(sources[0], slot_w, slot_h), (0, 0))
        canvas.paste(_fit_to_slot(sources[1], slot_w, slot_h), (slot_w, 0))
    else:
        # Two landscape slots stacked vertically.
        slot_w, slot_h = CANVAS_W, CANVAS_H // 2
        canvas.paste(_fit_to_slot(sources[0], slot_w, slot_h), (0, 0))
        canvas.paste(_fit_to_slot(sources[1], slot_w, slot_h), (0, slot_h))
    return canvas


def _process(sources: list[Image.Image], layout: str) -> tuple[bytes, bytes, bytes]:
    """Return (packed_192000_bytes, png_bytes, jpg_bytes).

    Takes the raw source images and a layout choice, composes them onto a canvas
    matching the device orientation, then rotates to panel-native 800x480 if
    needed, then quantizes/packs.

    DITHER_ENABLED controls quantization:
      true  -> Floyd-Steinberg dither to the 6-colour panel palette. Best
               output when the consumer is the panel directly (/frame.bin)
               or a client that won't re-process.
      false -> /frame.png and /frame.jpg are the cropped RGB image (no
               quantization at all), and /frame.bin uses nearest-colour
               mapping with no dither. Use when the client (e.g. aitjcize
               firmware) handles dither.
    """
    t0 = time.monotonic()
    sources = [_normalize_source(s) for s in sources]
    canvas = _compose_canvas(sources, layout)

    # FRAME_ROTATE applies to the final composed canvas (legacy behavior).
    if ROTATE in (90, 180, 270):
        canvas = canvas.rotate(-ROTATE, expand=True)

    # If the device is portrait, the canvas is 480x800. The panel still wants
    # 800x480 — so rotate 90° CCW so the top of the canvas becomes the left of
    # the panel. (Mount the panel with its native bottom-edge on the right.)
    if DEVICE_ORIENTATION == "portrait":
        canvas = canvas.rotate(-90, expand=True)  # 480x800 -> 800x480

    assert canvas.size == (PANEL_W, PANEL_H), (canvas.size, (PANEL_W, PANEL_H))

    if DITHER_ENABLED:
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

    png_buf = io.BytesIO()
    preview_source.save(png_buf, format="PNG")
    jpg_buf = io.BytesIO()
    preview_source.save(jpg_buf, format="JPEG", quality=85, optimize=True)

    log.info(
        "process: layout=%s sources=%d packed=%d png=%d jpg=%d, crop=%s, dither=%s, rotate=%s, %.0f ms",
        layout,
        len(sources),
        len(packed),
        len(png_buf.getvalue()),
        len(jpg_buf.getvalue()),
        CROP_MODE,
        DITHER_ENABLED,
        ROTATE,
        (time.monotonic() - t0) * 1000,
    )
    return packed, png_buf.getvalue(), jpg_buf.getvalue()


def _get_today(fresh: bool) -> dict:
    use_cache = CACHE_ENABLED and not fresh
    key = date.today().isoformat()
    if use_cache and key in _cache:
        log.info("get: cache hit for %s (assets %s)", key, _cache[key]["id"])
        return _cache[key]
    log.info("get: recomputing (cache=%s, fresh=%s)", CACHE_ENABLED, fresh)

    layout = _pick_layout()
    log.info(
        "get: layout=%s (device=%s, duo_prob=%.2f)",
        layout,
        DEVICE_ORIENTATION,
        DUO_PROBABILITY,
    )

    if layout == "duo":
        # Pick two DUO_SHAPE assets, second one excluding the first to avoid
        # showing the same photo twice.
        a1 = _pick_asset(DUO_SHAPE)
        try:
            a2 = _pick_asset(DUO_SHAPE, exclude_ids={a1["id"]})
        except RuntimeError as e:
            # Not enough DUO_SHAPE assets in the pool to fill two slots. Fall
            # back to a single SINGLE_SHAPE asset rather than failing the whole
            # refresh.
            log.warning("get: duo failed (%s), falling back to single", e)
            layout = "single"
            assets = [_pick_asset(SINGLE_SHAPE)]
        else:
            assets = [a1, a2]
    else:
        assets = [_pick_asset(SINGLE_SHAPE)]

    sources = [_fetch_image(a["id"]) for a in assets]
    packed, png, jpg = _process(sources, layout)
    # Use a compound id so the cache header / log line reflects the duo.
    composite_id = "+".join(a["id"] for a in assets)
    entry = {
        "id": composite_id,
        "bin": packed,
        "png": png,
        "jpg": jpg,
        "layout": layout,
    }
    if use_cache:
        _cache.clear()
        _cache[key] = entry
    return entry


def _no_store(headers: dict) -> dict:
    # Stop the browser (and any intermediary) from serving a cached frame on
    # reload. The device path doesn't cache, but browsers will cache an image at
    # a stable URL without this. Returns a new dict — does not mutate the input.
    return {**headers, "Cache-Control": "no-store, max-age=0"}


@app.get("/frame.bin")
def frame_bin():
    fresh = request.args.get("fresh") == "1"
    log.info("request: GET /frame.bin from %s (fresh=%s)", request.remote_addr, fresh)
    t0 = time.monotonic()
    try:
        entry = _get_today(fresh)
    except Exception as e:
        log.exception("frame.bin failed: %s", e)
        return Response(
            f"frame generation failed: {e}\n", status=503, mimetype="text/plain"
        )
    log.info(
        "request: served /frame.bin asset %s in %.0f ms",
        entry["id"],
        (time.monotonic() - t0) * 1000,
    )
    return Response(
        entry["bin"],
        mimetype="application/octet-stream",
        headers=_no_store(
            {"X-Immich-Asset-Id": entry["id"], "Content-Length": str(FRAME_BYTES)}
        ),
    )


@app.get("/frame.png")
def frame_png():
    fresh = request.args.get("fresh") == "1"
    log.info("request: GET /frame.png from %s (fresh=%s)", request.remote_addr, fresh)
    try:
        entry = _get_today(fresh)
    except Exception as e:
        log.exception("frame.png failed: %s", e)
        return Response(
            f"frame generation failed: {e}\n", status=503, mimetype="text/plain"
        )
    return Response(
        entry["png"],
        mimetype="image/png",
        headers=_no_store({"X-Immich-Asset-Id": entry["id"]}),
    )


@app.get("/frame.jpg")
def frame_jpg():
    fresh = request.args.get("fresh") == "1"
    log.info("request: GET /frame.jpg from %s (fresh=%s)", request.remote_addr, fresh)
    try:
        entry = _get_today(fresh)
    except Exception as e:
        log.exception("frame.jpg failed: %s", e)
        return Response(
            f"frame generation failed: {e}\n", status=503, mimetype="text/plain"
        )
    return Response(
        entry["jpg"],
        mimetype="image/jpeg",
        headers=_no_store({"X-Immich-Asset-Id": entry["id"]}),
    )


@app.get("/healthz")
def healthz():
    return {"ok": True, "album_mode": bool(ALBUM_ID)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
