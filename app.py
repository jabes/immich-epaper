"""
immich-epaper: serve a pre-dithered, pre-packed framebuffer for a
Waveshare 7.3" Spectra 6 (E6 / epd7in3e) panel, sourced from Immich.

Endpoints:
  GET /frame.bin   -> 192000 bytes, the exact buffer EPD_7IN3E_Display() wants
                      (800x480, 4 bits/pixel, 2 pixels/byte, high nibble = left pixel)
  GET /frame.png   -> the same dithered image as PNG, for eyeballing in a browser
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
except Exception:  # pragma: no cover
    _smartcrop = None
    _SMARTCROP = None

# ---------------------------------------------------------------------------
# Panel definition. These are the authoritative epd7in3e (E6) values.
# Palette order below is fixed; CODE_LUT maps each palette index to the
# panel's 4-bit colour code. Note the gap: 0x4 is unused on the E6 (it was
# orange on the 7-colour "F" panel), so blue/green are 0x5/0x6.
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 800, 480
FRAME_BYTES = WIDTH * HEIGHT // 2  # 192000

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
# any | landscape | portrait | square. Immich has no native orientation filter
# (open feature request: discussions #19478 / #24216 / #9098), so we set
# withExif and drop non-matching candidates locally before the random pick.
ORIENTATION = os.environ.get("IMMICH_ORIENTATION", "any").strip().lower()
if ORIENTATION not in {"any", "landscape", "portrait", "square"}:
    raise SystemExit(
        f"IMMICH_ORIENTATION={ORIENTATION!r} must be any|landscape|portrait|square"
    )

# center | smart. smart uses smartcrop.py (edge/saturation/skin heuristics) to
# pick the most "interesting" 800x480 window from the source, then falls back to
# center-crop on any error or when the source is smaller than the target.
CROP_MODE = os.environ.get("IMMICH_CROP", "center").strip().lower()
if CROP_MODE not in {"center", "smart"}:
    raise SystemExit(f"IMMICH_CROP={CROP_MODE!r} must be center|smart")
if CROP_MODE == "smart" and _SMARTCROP is None:
    raise SystemExit(
        "IMMICH_CROP=smart requires the 'smartcrop' package (add it to requirements.txt)"
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

HEADERS = {"x-api-key": API_KEY, "Accept": "application/json"}


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
        ("IMMICH_ORIENTATION", ORIENTATION),
        ("IMMICH_CROP", CROP_MODE),
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


_log_startup_config()

app = Flask(__name__)

# date_str -> {"id": str, "bin": bytes, "png": bytes}
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


def _not_excluded(asset: dict) -> bool:
    if not EXCLUDE_PEOPLE:
        return True
    for p in asset.get("people", []):
        if p.get("id") in EXCLUDE_PEOPLE:
            return False
    return True


def _orientation_ok(asset: dict) -> bool:
    """True if the asset matches IMMICH_ORIENTATION. 'any' always passes; 'square'
    allows a small tolerance because real photos are rarely pixel-perfect square."""
    if ORIENTATION == "any":
        return True
    exif = asset.get("exifInfo") or {}
    w, h = exif.get("exifImageWidth"), exif.get("exifImageHeight")
    if not w or not h:
        # No EXIF dims -> can't classify. Drop it; it's safer than risking a portrait
        # being shown when the user asked for landscape, and there's almost always
        # plenty of other candidates in the batch.
        return False
    if ORIENTATION == "landscape":
        return w > h
    if ORIENTATION == "portrait":
        return h > w
    # square
    longer, shorter = max(w, h), min(w, h)
    return (longer / shorter) <= 1.05


def _search_body() -> dict:
    body: dict = {
        "type": "IMAGE",
        "size": max(1, min(SEARCH_BATCH, 1000)),
        "withArchived": False,
        # needed so we can apply exclude_people locally
        "withPeople": bool(EXCLUDE_PEOPLE),
        # needed so we can filter by orientation locally (exifImageWidth/Height)
        "withExif": ORIENTATION != "any",
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


def _pick_asset_id() -> str:
    if ALBUM_ID:
        r = requests.get(
            f"{IMMICH_URL}/api/albums/{ALBUM_ID}", headers=HEADERS, timeout=30
        )
        r.raise_for_status()
        raw = [a for a in r.json().get("assets", []) if a.get("type") == "IMAGE"]
        after_excl = [a for a in raw if _not_excluded(a)]
        assets = [a for a in after_excl if _orientation_ok(a)]
        log.info(
            "pick: album mode, %d images, %d after exclusion, %d after orientation=%s",
            len(raw),
            len(after_excl),
            len(assets),
            ORIENTATION,
        )
        if not assets:
            raise RuntimeError("album has no matching image assets")
        a = random.choice(assets)
        chosen = a["id"]
        log.info(
            "pick: chose %s (%s)  %s/photos/%s",
            chosen,
            a.get("originalFileName", "?"),
            IMMICH_PUBLIC_URL,
            chosen,
        )
        return chosen

    body = _search_body()
    log.info("pick: search/random body=%s", body)
    r = requests.post(
        f"{IMMICH_URL}/api/search/random",
        headers={**HEADERS, "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    # current API returns a bare list; tolerate the paged shape too
    raw = data if isinstance(data, list) else data.get("assets", {}).get("items", [])
    after_excl = [a for a in raw if _not_excluded(a)]
    assets = [a for a in after_excl if _orientation_ok(a)]
    log.info(
        "pick: search returned %d, %d after exclusion, %d after orientation=%s",
        len(raw),
        len(after_excl),
        len(assets),
        ORIENTATION,
    )
    if not assets:
        raise RuntimeError("no assets matched the configured filters")
    a = random.choice(assets)
    chosen = a["id"]
    log.info(
        "pick: chose %s (%s)  %s/photos/%s",
        chosen,
        a.get("originalFileName", "?"),
        IMMICH_PUBLIC_URL,
        chosen,
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


def _smart_fit(img: Image.Image) -> Image.Image:
    """800x480 crop chosen by smartcrop.py, then resampled to the target.

    Returns a center-fit on any failure (source too small, smartcrop error) so
    a bad image can never wedge the whole pipeline.
    """
    if img.width < WIDTH or img.height < HEIGHT:
        log.info(
            "crop: source %dx%d smaller than %dx%d, falling back to center",
            img.width,
            img.height,
            WIDTH,
            HEIGHT,
        )
        return ImageOps.fit(
            img, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
        )
    try:
        assert _SMARTCROP is not None  # CROP_MODE validation guarantees this
        # smartcrop scoring is content-blind to resolution but it's slow on huge
        # images. Downscale the long edge to ~1024 before scoring; the chosen
        # region scales linearly back to the full-res image, so quality of the
        # final crop is unaffected.
        scale = min(1.0, 1024 / max(img.width, img.height))
        if scale < 1.0:
            scored = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.Resampling.LANCZOS,
            )
        else:
            scored = img
        result = _SMARTCROP.crop(scored, WIDTH, HEIGHT)
        c = result["top_crop"]
        x, y, w, h = c["x"], c["y"], c["width"], c["height"]
        # Map back to full-res coordinates.
        inv = 1.0 / scale
        box = (int(x * inv), int(y * inv), int((x + w) * inv), int((y + h) * inv))
        cropped = img.crop(box)
        log.info("crop: smart picked %s from %dx%d", box, img.width, img.height)
        return cropped.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    except Exception as e:
        log.warning("crop: smartcrop failed (%s), falling back to center", e)
        return ImageOps.fit(
            img, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
        )


def _process(img: Image.Image) -> tuple[bytes, bytes]:
    """Return (packed_192000_bytes, png_preview_bytes)."""
    t0 = time.monotonic()
    # Split across two statements so Pylance can narrow the type: the stub for
    # exif_transpose declares Image | None (because of its in-place overload),
    # and `.convert` on None would warn. `or img` is a safe fallback if a
    # future Pillow really does return None here.
    img = ImageOps.exif_transpose(img) or img
    img = img.convert("RGB")
    if ROTATE in (90, 180, 270):
        img = img.rotate(-ROTATE, expand=True)
    if CROP_MODE == "smart":
        img = _smart_fit(img)
    else:
        img = ImageOps.fit(
            img, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
        )

    quant = img.quantize(palette=_PAL_IMG, dither=Image.Dither.FLOYDSTEINBERG)

    idx = np.asarray(quant, dtype=np.uint8)  # (480, 800), values 0..5
    codes = CODE_LUT[idx]  # (480, 800), panel nibble codes
    hi = codes[:, 0::2] << 4  # even columns -> high nibble
    lo = codes[:, 1::2]  # odd columns  -> low nibble
    packed = (hi | lo).astype(np.uint8).tobytes()  # row-major -> 192000 bytes
    assert len(packed) == FRAME_BYTES, len(packed)

    png_buf = io.BytesIO()
    quant.convert("RGB").save(png_buf, format="PNG")
    log.info(
        "process: dithered+packed %d bytes, crop=%s, rotate=%s, %.0f ms",
        len(packed),
        CROP_MODE,
        ROTATE,
        (time.monotonic() - t0) * 1000,
    )
    return packed, png_buf.getvalue()


def _get_today(fresh: bool) -> dict:
    use_cache = CACHE_ENABLED and not fresh
    key = date.today().isoformat()
    if use_cache and key in _cache:
        log.info("get: cache hit for %s (asset %s)", key, _cache[key]["id"])
        return _cache[key]
    log.info("get: recomputing (cache=%s, fresh=%s)", CACHE_ENABLED, fresh)
    asset_id = _pick_asset_id()
    packed, png = _process(_fetch_image(asset_id))
    entry = {"id": asset_id, "bin": packed, "png": png}
    if use_cache:
        _cache.clear()  # only keep the current day
        _cache[key] = entry
    return entry


def _no_store(headers: dict) -> dict:
    # Stop the browser (and any intermediary) from serving a cached frame on
    # reload. The device path doesn't cache, but browsers will cache an image at
    # a stable URL without this.
    headers["Cache-Control"] = "no-store, max-age=0"
    return headers


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


@app.get("/healthz")
def healthz():
    return {"ok": True, "album_mode": bool(ALBUM_ID)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
