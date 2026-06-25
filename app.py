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

Every request picks fresh assets from Immich; rotation cadence (hourly, daily,
etc) is the firmware's responsibility, not the server's.

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

  Labels (drawn on top of each slot, before quantization so the text stays crisp):
  IMMICH_SHOW_NAMES         true (default) | false. Draw person names in a
                            corner of each slot in black text on a white
                            rectangle. Requires Immich face recognition to
                            have run.
  IMMICH_LABEL_FONT_SIZE    px, default 18.
  IMMICH_LABEL_FONT         font filename (no path, no extension), default
                            "DejaVuSans-Bold". Looked up under
                            /usr/share/fonts/truetype/dejavu/. With
                            fonts-dejavu installed you also get
                            DejaVuSans-{Oblique,BoldOblique,ExtraLight},
                            DejaVuSansCondensed-*, DejaVuSansMono-*, and the
                            DejaVuSerif-* family. Falls back to Bold then
                            Regular if the requested font isn't found.
  IMMICH_LABEL_CORNER       top-left | top-middle | top-right |
                            bottom-left | bottom-middle | bottom-right (default).
                            In duo layouts, the position is relative to the SLOT,
                            not the canvas.
  IMMICH_FIRST_NAME_ONLY    true (default) | false. Split each name on whitespace
                            and use the first token only ("Henry Bull" -> "Henry").
  IMMICH_LABEL_DELIMITER    String between names. Default " - ". Whitespace is
                            preserved as written, so " - " gives "Henry - Maya"
                            while "-" gives "Henry-Maya".

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

  Quality ranking:
  IMMICH_RANKING_BATCH      how many candidates to fetch and score per request (default 5)
  IMMICH_QUALITY_ENABLED    true (default) | false. If false, picks randomly (no scoring).
"""

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

# Quality scoring libraries – assumed available at build time
import pyiqa
import requests
from flask import Flask, Response, request
from PIL import Image, ImageDraw, ImageFont, ImageOps

# smartcrop – optional (we keep the existing fallback)
try:
    import smartcrop as _smartcrop

    _SMARTCROP = _smartcrop.SmartCrop()
    _SMARTCROP_ERROR = None
except Exception as e:
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

# Quality ranking settings
RANKING_BATCH = int(os.environ.get("IMMICH_RANKING_BATCH", "5"))
QUALITY_ENABLED = os.environ.get("IMMICH_QUALITY_ENABLED", "true").lower() != "false"

# Name-label settings. Labels are drawn into each slot on the RGB canvas BEFORE
# quantization so the text renders crisply (black/white quantize cleanly to the
# panel's actual black/white; if drawn after dithering, the text edges would
# pick up rainbow speckles from surrounding pixels).
SHOW_NAMES = os.environ.get("IMMICH_SHOW_NAMES", "true").lower() != "false"
LABEL_FONT_SIZE = int(os.environ.get("IMMICH_LABEL_FONT_SIZE", "18"))
LABEL_PADDING = 6  # px around text inside the white pill
# Corner of each slot to anchor labels in. Note: in duo layouts this is the
# corner of the SLOT, not the canvas — so e.g. "bottom-right" in a landscape
# duo places one label at the bottom-right of the left photo and another at
# the bottom-right of the right photo, not both crammed into one corner.
LABEL_CORNER = os.environ.get("IMMICH_LABEL_CORNER", "bottom-right").strip().lower()
_VALID_CORNERS = {
    "top-left",
    "top-middle",
    "top-right",
    "bottom-left",
    "bottom-middle",
    "bottom-right",
}
if LABEL_CORNER not in _VALID_CORNERS:
    raise SystemExit(
        f"IMMICH_LABEL_CORNER={LABEL_CORNER!r} must be one of {sorted(_VALID_CORNERS)}"
    )
# First name only: split on whitespace, take token 0. Useful when your Immich
# face tags use full names ("Henry Bull" -> "Henry") and you don't want the
# label to take up half the photo.
FIRST_NAME_ONLY = os.environ.get("IMMICH_FIRST_NAME_ONLY", "true").lower() != "false"
# Separator between names. Default " - "; common alternatives: ", " or " · ".
# Any string is accepted; surrounding whitespace is preserved verbatim from
# the env var so you can control padding (e.g. "-" with no spaces).
LABEL_DELIMITER = os.environ.get("IMMICH_LABEL_DELIMITER", " - ")
# Font filename (no path, no extension). Looked up under the standard DejaVu
# install dir on Debian. Some valid values:
#   DejaVuSans, DejaVuSans-Bold, DejaVuSans-BoldOblique, DejaVuSans-ExtraLight,
#   DejaVuSans-Oblique, DejaVuSansCondensed, DejaVuSansCondensed-Bold, ...
#   DejaVuSansMono, DejaVuSansMono-Bold, ...
#   DejaVuSerif, DejaVuSerif-Bold, DejaVuSerif-BoldItalic, ...
LABEL_FONT = os.environ.get("IMMICH_LABEL_FONT", "DejaVuSans-Bold").strip()

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

log.info("Quality scoring enabled (pyiqa NIMA, BRISQUE, sharpness)")


def _load_font(size: int):
    """Load LABEL_FONT from the standard DejaVu install dir, falling back to
    DejaVuSans-Bold, then DejaVuSans, then Pillow's bitmap default. The cascade
    means a typo in IMMICH_LABEL_FONT logs a warning and keeps running with
    something reasonable rather than rejecting the request."""
    candidates = [LABEL_FONT, "DejaVuSans"]
    tried: list[str] = []
    for name in candidates:
        path = f"/usr/share/fonts/truetype/dejavu/{name}.ttf"
        tried.append(path)
        try:
            font = ImageFont.truetype(path, size)
            if name != LABEL_FONT:
                log.warning(
                    "IMMICH_LABEL_FONT=%r not found; using %r instead",
                    LABEL_FONT,
                    name,
                )
            return font
        except (OSError, IOError):
            continue
    log.warning(
        "No DejaVu font found (tried %s); using Pillow default (text will look bad). "
        "Add `fonts-dejavu-core` (or `fonts-dejavu` for the full set) to the Dockerfile.",
        tried,
    )
    return ImageFont.load_default()


_LABEL_FONT = _load_font(LABEL_FONT_SIZE)

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
        ("IMMICH_SHOW_NAMES", SHOW_NAMES),
        ("IMMICH_LABEL_FONT_SIZE", LABEL_FONT_SIZE),
        ("IMMICH_LABEL_FONT", LABEL_FONT),
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


def _asset_names(asset: dict) -> list[str]:
    """Return non-empty person names from the asset, deduped, preserving the
    order Immich returns them. Immich includes unnamed face clusters with empty
    name strings; skip those rather than rendering "Anna, , , Henry".

    When FIRST_NAME_ONLY, the first whitespace-separated token is used. Dedup
    happens AFTER trimming so "Henry Bull" and "Henry James" collapse to one
    "Henry" entry — desirable if you mostly have one Henry in your library,
    accept the corner case otherwise.
    """
    seen: set[str] = set()
    out: list[str] = []
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
    body: dict = {
        "type": "IMAGE",
        "size": max(1, min(SEARCH_BATCH, 1000)),
        "withArchived": False,
        # Always request people: needed to draw name labels on slots, and the
        # cost is negligible (Immich already has them indexed).
        "withPeople": True,
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


def _draw_name_label(
    canvas: Image.Image,
    names: list[str],
    slot_x: int,
    slot_y: int,
    slot_w: int,
    slot_h: int,
) -> None:
    """Draw "Name, Name, Name" anchored to LABEL_CORNER of a slot. Black text
    on a white rectangle, padded LABEL_PADDING. Truncates with an ellipsis if
    the joined string doesn't fit in slot_w. No-op if names is empty or
    SHOW_NAMES is off.
    """
    if not names or not SHOW_NAMES:
        return
    draw = ImageDraw.Draw(canvas)
    text = LABEL_DELIMITER.join(names)
    max_text_w = slot_w - 2 * LABEL_PADDING

    # Truncate with ellipsis if the joined names overflow the slot. Strip any
    # characters from the delimiter when tidying the tail so we don't end up
    # with a dangling separator before the ellipsis.
    if draw.textlength(text, font=_LABEL_FONT) > max_text_w:
        ell = "…"
        while text and draw.textlength(text + ell, font=_LABEL_FONT) > max_text_w:
            text = text[:-1]
        text = text.rstrip(LABEL_DELIMITER) + ell

    tw = draw.textlength(text, font=_LABEL_FONT)
    # ascent+descent gives a tighter, more predictable label box than textbbox
    # for short labels. Pillow's bitmap default font lacks getmetrics(), so use
    # getattr to handle both the FreeTypeFont and ImageFont base cases without
    # tripping the type checker on a union-typed attribute access.
    metrics = getattr(_LABEL_FONT, "getmetrics", lambda: (12, 2))()
    th = metrics[0] + metrics[1]

    rect_w = int(tw) + 2 * LABEL_PADDING
    rect_h = th + 2 * LABEL_PADDING

    # Pin the label rectangle to the requested corner of the slot. No outside
    # margin — the label sits flush against the slot edge.
    if LABEL_CORNER.endswith("-left"):
        rect_x = slot_x
    elif LABEL_CORNER.endswith("-middle"):
        rect_x = slot_x + (slot_w - rect_w) // 2
    else:  # -right
        rect_x = slot_x + slot_w - rect_w
    if LABEL_CORNER.startswith("top-"):
        rect_y = slot_y
    else:  # bottom-
        rect_y = slot_y + slot_h - rect_h

    draw.rectangle(
        (rect_x, rect_y, rect_x + rect_w, rect_y + rect_h),
        fill=(255, 255, 255),
    )
    draw.text(
        (rect_x + LABEL_PADDING, rect_y + LABEL_PADDING - 1),
        text,
        fill=(0, 0, 0),
        font=_LABEL_FONT,
    )


def _compose_canvas(
    sources: list[Image.Image], layout: str, assets: list[dict]
) -> Image.Image:
    """Build the final RGB canvas (CANVAS_W x CANVAS_H) before rotation and
    quantization. The `assets` argument is parallel to `sources` and supplies
    the names used for per-slot labels.

    layout='single': one source filling the entire canvas.
    layout='duo': two sources split along the canvas's *long* axis. For a
    landscape canvas (800x480), portraits sit side-by-side at 400x480 each.
    For a portrait canvas (480x800), landscapes stack at 480x400 each.
    """
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
    if layout == "single":
        assert len(sources) == 1 and len(assets) == 1
        canvas.paste(_fit_to_slot(sources[0], CANVAS_W, CANVAS_H), (0, 0))
        _draw_name_label(canvas, _asset_names(assets[0]), 0, 0, CANVAS_W, CANVAS_H)
        return canvas

    assert layout == "duo" and len(sources) == 2 and len(assets) == 2
    if DEVICE_ORIENTATION == "landscape":
        # Two portrait slots side-by-side. Splitting 800 in two gives 400-wide
        # slots; if you ever want a gap between, subtract a margin here.
        slot_w, slot_h = CANVAS_W // 2, CANVAS_H
        canvas.paste(_fit_to_slot(sources[0], slot_w, slot_h), (0, 0))
        canvas.paste(_fit_to_slot(sources[1], slot_w, slot_h), (slot_w, 0))
        _draw_name_label(canvas, _asset_names(assets[0]), 0, 0, slot_w, slot_h)
        _draw_name_label(canvas, _asset_names(assets[1]), slot_w, 0, slot_w, slot_h)
    else:
        # Two landscape slots stacked vertically.
        slot_w, slot_h = CANVAS_W, CANVAS_H // 2
        canvas.paste(_fit_to_slot(sources[0], slot_w, slot_h), (0, 0))
        canvas.paste(_fit_to_slot(sources[1], slot_w, slot_h), (0, slot_h))
        _draw_name_label(canvas, _asset_names(assets[0]), 0, 0, slot_w, slot_h)
        _draw_name_label(canvas, _asset_names(assets[1]), 0, slot_h, slot_w, slot_h)
    return canvas


def _rotate_final_canvas(canvas: Image.Image) -> Image.Image:
    """Apply FRAME_ROTATE and device‑orientation rotation to get the final
    panel‑native 800x480 RGB canvas."""
    if ROTATE in (90, 180, 270):
        canvas = canvas.rotate(-ROTATE, expand=True)
    if DEVICE_ORIENTATION == "portrait":
        canvas = canvas.rotate(-90, expand=True)  # 480x800 -> 800x480
    assert canvas.size == (PANEL_W, PANEL_H), (canvas.size, (PANEL_W, PANEL_H))
    return canvas


def _pack_canvas(canvas: Image.Image) -> tuple[bytes, bytes, bytes]:
    """Quantize (with or without dither) and pack the canvas into the 192k buffer.
    Also return PNG and JPEG of the preview source."""
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
    return packed, png_buf.getvalue(), jpg_buf.getvalue()


def _process(
    sources: list[Image.Image], assets: list[dict], layout: str
) -> tuple[bytes, bytes, bytes]:
    """Compose, rotate, pack, and return (packed, png, jpg)."""
    t0 = time.monotonic()
    sources = [_normalize_source(s) for s in sources]
    canvas = _compose_canvas(sources, layout, assets)
    canvas = _rotate_final_canvas(canvas)
    packed, png, jpg = _pack_canvas(canvas)

    log.info(
        "process: layout=%s sources=%d packed=%d png=%d jpg=%d, crop=%s, dither=%s, rotate=%s, %.0f ms",
        layout,
        len(sources),
        len(packed),
        len(png),
        len(jpg),
        CROP_MODE,
        DITHER_ENABLED,
        ROTATE,
        (time.monotonic() - t0) * 1000,
    )
    return packed, png, jpg


# ---------------------------------------------------------------------------
# Quality scoring functions – using pyiqa for both NIMA and BRISQUE
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
        score = score_tensor.cpu().detach().item()
        return max(1.0, min(10.0, score))
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
        score = score_tensor.cpu().detach().item()
        return float(score)
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
    """Return a normalised composite score in [0,1] (higher = better)."""
    nima = _get_nima_score(img)  # ~5-8
    brisque = _get_brisque_score(img)  # ~20-60
    sharp = _get_sharpness_score(img)  # ~100-2000 for 800x480

    # Normalise each
    nima_norm = max(0.0, min(1.0, (nima - 1.0) / 9.0))  # 1-10 -> 0-1
    brisque_norm = max(0.0, min(1.0, brisque / 100.0))  # assume 0-100
    sharp_norm = max(0.0, min(1.0, sharp / 2000.0))  # cap at 2000

    # Combine: NIMA and sharpness are positive, BRISQUE is negative
    composite = nima_norm * 0.6 + sharp_norm * 0.3 - brisque_norm * 0.1
    return max(0.0, min(1.0, composite))


# ---------------------------------------------------------------------------
# Main frame builder – with ranking
# ---------------------------------------------------------------------------
def _build_frame() -> dict:
    """Pick the best candidate by quality ranking (or random if disabled)."""
    layout = _pick_layout()
    log.info(
        "build: layout=%s (device=%s, duo_prob=%.2f)",
        layout,
        DEVICE_ORIENTATION,
        DUO_PROBABILITY,
    )

    # If quality scoring is disabled, just pick the first candidate
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
        packed, png, jpg = _process(sources, assets, layout)
        composite_id = "+".join(a["id"] for a in assets)
        return {
            "id": composite_id,
            "bin": packed,
            "png": png,
            "jpg": jpg,
            "layout": layout,
        }

    # --- Quality ranking path ---
    candidates = []
    for i in range(RANKING_BATCH):
        try:
            if layout == "duo":
                a1 = _pick_asset(DUO_SHAPE)
                try:
                    a2 = _pick_asset(DUO_SHAPE, exclude_ids={a1["id"]})
                except RuntimeError:
                    # Not enough duo assets; fall back to single for this candidate
                    log.warning("candidate %d: duo failed, skipping", i + 1)
                    continue
                assets = [a1, a2]
            else:
                assets = [_pick_asset(SINGLE_SHAPE)]
            # Fetch and compose
            sources = [_fetch_image(a["id"]) for a in assets]
            # Compose (with labels) and rotate to final canvas, before quantization
            canvas = _compose_canvas(
                [_normalize_source(s) for s in sources], layout, assets
            )
            canvas = _rotate_final_canvas(canvas)
            # Score the RGB canvas. NB: scoring runs against the labelled canvas,
            # so quality models see the white label pills too. This shifts scores
            # a touch (NIMA may like the clean whitespace; BRISQUE may not love
            # the hard rectangle edges) but the bias is consistent across all
            # candidates, so the ranking order is preserved.
            score = _composite_score(canvas)
            log.info(
                "candidate %d: score=%.4f (assets: %s)",
                i + 1,
                score,
                [a["id"] for a in assets],
            )
            candidates.append((score, assets, canvas))
        except Exception as e:
            log.warning("candidate %d generation failed: %s", i + 1, e)
            continue

    if not candidates:
        log.error(
            "No candidates could be generated; falling back to random single pick"
        )
        # Ultimate fallback: pick one random asset (single)
        assets = [_pick_asset(SINGLE_SHAPE)]
        sources = [_fetch_image(a["id"]) for a in assets]
        packed, png, jpg = _process(sources, assets, "single")
        composite_id = assets[0]["id"]
        return {
            "id": composite_id,
            "bin": packed,
            "png": png,
            "jpg": jpg,
            "layout": "single",
        }

    # Sort by score descending, pick the best
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_assets, best_canvas = candidates[0]
    log.info(
        "best candidate score=%.4f, assets=%s",
        best_score,
        [a["id"] for a in best_assets],
    )

    # Pack the best canvas
    packed, png, jpg = _pack_canvas(best_canvas)
    composite_id = "+".join(a["id"] for a in best_assets)
    return {"id": composite_id, "bin": packed, "png": png, "jpg": jpg, "layout": layout}


def _no_store(headers: dict) -> dict:
    # Stop the browser (and any intermediary) from serving a cached frame on
    # reload. The device path doesn't cache, but browsers will cache an image at
    # a stable URL without this. Returns a new dict — does not mutate the input.
    return {**headers, "Cache-Control": "no-store, max-age=0"}


@app.get("/frame.bin")
def frame_bin():
    log.info("request: GET /frame.bin from %s", request.remote_addr)
    t0 = time.monotonic()
    try:
        entry = _build_frame()
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
    log.info("request: GET /frame.png from %s", request.remote_addr)
    try:
        entry = _build_frame()
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
    log.info("request: GET /frame.jpg from %s", request.remote_addr)
    try:
        entry = _build_frame()
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
