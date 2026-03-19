from __future__ import annotations

import argparse
import base64
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from camoufox.sync_api import Camoufox
from playwright.sync_api import Frame, Page, TimeoutError as PlaywrightTimeoutError

from gradient_highlight import find_best_match_on_gradients


LOG = logging.getLogger("collect_solve_camoufox")

# Hard-pinned window/viewport to keep layout stable across runs
# Use a larger size to reduce responsive layout changes/cropping.
CAMOUFOX_WINDOW = (1920, 1080)
CAMOUFOX_VIEWPORT = {"width": 1920, "height": 1080}
CAMOUFOX_SCREEN = {"width": 1920, "height": 1080}

STEP_WAIT_S = 3.0
DRAG_STEP_DELAY_S = 0.012


TAB_CONTAINER_XPATH = "//div[contains(@class,'tab-container')]"
TAB1_XPATH = "//div[@class='tab-item tab-item-1']"
TAB3_XPATH = "//div[@class='tab-item tab-item-3']"
TRACK_XPATH = "//div[contains(@class,'geetest_track_')]"
BTN_XPATH = "//div[contains(@class,'geetest_btn')][ancestor::div[contains(@class,'geetest_slider')]]"


URL_RE = re.compile(r"url\\([\"']?(.*?)[\"']?\\)")


@dataclass(frozen=True)
class CaptchaImages:
    bg_url: str
    puzzle_url: str
    bg_bgr: np.ndarray
    puzzle_bgr: np.ndarray
    puzzle_mask_u8: np.ndarray | None


def _sleep_action(min_s: float, max_s: float, *, label: str | None = None) -> None:
    # Fixed timing mode: always sleep exactly STEP_WAIT_S.
    # min_s/max_s/label are ignored intentionally for determinism.
    time.sleep(STEP_WAIT_S)


def _urls_from_style_background_image(bg_value: str) -> list[str]:
    if not bg_value:
        return []
    m = URL_RE.search(bg_value)
    if not m:
        return []
    url = (m.group(1) or "").strip()
    return [url] if url else []


def _unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _download_bytes(page: Page, url: str, *, timeout_ms: int = 20000) -> bytes:
    resp = page.context.request.get(url, timeout=timeout_ms)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status} while downloading: {url}")
    return resp.body()


def _download_bytes_via_frame_fetch(frame: Frame, url: str, *, timeout_ms: int = 20000) -> bytes:
    """
    Fallback for cases where `context.request.get(url)` can't access the resource
    (e.g. cookies/headers/origin restrictions). Fetches inside the page context.
    Returns raw bytes.
    """
    b64: str = frame.evaluate(
        """
        async ({ url, timeoutMs }) => {
          const ctrl = new AbortController();
          const t = setTimeout(() => ctrl.abort(), timeoutMs);
          try {
            const resp = await fetch(url, { signal: ctrl.signal, credentials: "include" });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const buf = await resp.arrayBuffer();
            let binary = "";
            const bytes = new Uint8Array(buf);
            const chunk = 0x8000;
            for (let i = 0; i < bytes.length; i += chunk) {
              binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
            }
            return btoa(binary);
          } finally {
            clearTimeout(t);
          }
        }
        """,
        {"url": url, "timeoutMs": int(timeout_ms)},
    )
    return base64.b64decode(b64)


def _decode_image_with_optional_alpha(buf: bytes) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to decode image bytes (imdecode returned None)")

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return bgr, None

    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        mask_u8 = (alpha > 0).astype(np.uint8) * 255
        return bgr, mask_u8

    return img[:, :, :3], None


def _open_geetest_demo_captcha(page: Page, *, explicit_wait_s: float = 15.0) -> None:
    """
    Best-effort opener for the Geetest demo layout used in the existing RL script:
    - click tab container
    - click tab-item-1
    - click "Click to verify" button
    """
    try:
        LOG.info("Waiting %.1fs", STEP_WAIT_S)
        time.sleep(STEP_WAIT_S)

        tab1 = page.wait_for_selector(f"xpath={TAB1_XPATH}", timeout=int(explicit_wait_s * 1000))
        page.evaluate("el => el.click()", tab1)
        LOG.info("Clicked tab-item-1; waiting %.1fs", STEP_WAIT_S)
        time.sleep(STEP_WAIT_S)
    except PlaywrightTimeoutError:
        LOG.info("tab-item-1 not found; continuing")

    try:
        btn_verify = page.wait_for_selector(
            "css=div.geetest_btn_click[aria-label='Click to verify']",
            timeout=int(explicit_wait_s * 1000),
        )
        btn_verify.click()
        LOG.info('Clicked "Click to verify"; waiting %.1fs', STEP_WAIT_S)
        time.sleep(STEP_WAIT_S)
    except PlaywrightTimeoutError:
        LOG.info("verify button not found; continuing")


def _dismiss_cookie_banners(page: Page) -> None:
    """
    Best-effort: click common cookie consent buttons if present.
    """
    candidates = [
        "button:has-text('Accept')",
        "button:has-text('I accept')",
        "button:has-text('Agree')",
        "button:has-text('OK')",
        "button:has-text('Got it')",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0 and loc.is_visible():
                loc.click(timeout=1500)
                time.sleep(0.25)
                break
        except Exception:
            continue


def _extract_geetest_bg_puzzle_urls(frame: Frame) -> tuple[str | None, str | None]:
    """
    Try to read Geetest bg/slice URLs from known elements/classes.
    Returns (bg_url, puzzle_url) if found.
    """
    out: list[str | None] = frame.evaluate(
        """
        () => {
          const pickUrl = (el) => {
            if (!el) return null;
            const st = window.getComputedStyle(el);
            const v = st && st.backgroundImage ? st.backgroundImage : "";
            const m = v.match(/url\\(["']?(.*?)["']?\\)/);
            return m ? m[1] : null;
          };

          // Geetest v4 (as provided by user): bg and slice are divs with background-image
          //   //div[contains(@class, 'geetest_bg_')]
          //   //div[contains(@class, 'geetest_slice_bg_')]
          const bg =
            document.querySelector("div[class*='geetest_bg_']") ||
            document.querySelector("div.geetest_bg");
          const slice =
            document.querySelector("div[class*='geetest_slice_bg_']") ||
            document.querySelector("div.geetest_slice_bg");

          const bgUrl = pickUrl(bg);
          const sliceUrl = pickUrl(slice);
          return [bgUrl, sliceUrl];
        }
        """,
    )
    return out[0], out[1]


def _wait_for_geetest_open(page: Page, *, timeout_s: float) -> Frame:
    """
    Wait until a frame (page or iframe) contains Geetest slider elements.
    Returns the frame to use for extraction.
    """
    deadline = time.time() + max(timeout_s, 1.0)
    last_err: Exception | None = None
    while time.time() < deadline:
        for fr in page.frames:
            try:
                has = fr.evaluate(
                    """
                    () => !!(
                      document.querySelector("div[class*='geetest_slider']") ||
                      document.querySelector("div[class*='geetest_track_']") ||
                      document.querySelector("div.geetest_btn_click") ||
                      document.querySelector("div[class*='geetest_bg_']") ||
                      document.querySelector("div[class*='geetest_slice_bg_']")
                    )
                    """
                )
                if has:
                    return fr
            except Exception as exc:
                last_err = exc
                continue
        time.sleep(0.25)
    raise RuntimeError(f"Geetest elements not found in any frame within {timeout_s:.1f}s") from last_err


def _get_geetest_bg_display_width_px(frame: Frame) -> float | None:
    try:
        w: float | None = frame.evaluate(
            """
            () => {
              const el =
                document.querySelector("div[class*='geetest_bg_']") ||
                document.querySelector("div.geetest_bg");
              if (!el) return null;
              const r = el.getBoundingClientRect();
              return r && r.width ? r.width : null;
            }
            """
        )
        return w
    except Exception:
        return None


def _read_slider_handle_viewport_x(frame: Frame) -> float | None:
    """
    Read current slider handle X in *frame viewport* coordinates (best-effort).
    Used to detect whether drag actually moved the handle.
    """
    try:
        x: float | None = frame.evaluate(
            """
            () => {
              const el =
                document.querySelector("div[class*='geetest_btn']") ||
                document.querySelector("div.geetest_btn");
              if (!el) return null;
              const r = el.getBoundingClientRect();
              return r && typeof r.x === "number" ? r.x : null;
            }
            """
        )
        return x
    except Exception:
        return None


def _drag_geetest_slider(page: Page, frame: Frame, *, distance_px: float) -> None:
    """
    Drag the Geetest slider by a given distance in page pixels, using Playwright's mouse
    (Camoufox internal cursor).
    """
    handle = frame.wait_for_selector(f"xpath={BTN_XPATH}", timeout=10_000)
    box = handle.bounding_box()
    if not box:
        raise RuntimeError("Slider button bounding box is not available")

    pre_handle_x = _read_slider_handle_viewport_x(frame)

    def _perform_drag(start_x: float, start_y: float) -> None:
        end_x = start_x + float(distance_px)
        steps = 36
        page.mouse.move(start_x, start_y)
        time.sleep(DRAG_STEP_DELAY_S * 10)
        page.mouse.down()
        time.sleep(DRAG_STEP_DELAY_S * 8)

        for i in range(1, steps + 1):
            t = i / steps
            eased = 1 - (1 - t) * (1 - t)  # ease-out quadratic
            x = start_x + (end_x - start_x) * eased
            y = start_y + (0.35 if (i % 7 == 0) else 0.0)
            page.mouse.move(x, y, steps=1)
            time.sleep(DRAG_STEP_DELAY_S)

        time.sleep(DRAG_STEP_DELAY_S * 12)
        page.mouse.up()

    # Attempt #1: use handle.bounding_box() coordinates directly.
    start1_x = box["x"] + box["width"] * 0.5
    start1_y = box["y"] + box["height"] * 0.5
    _perform_drag(start1_x, start1_y)

    # If the handle did not move, we might be in an iframe and coordinates are relative.
    post_handle_x = _read_slider_handle_viewport_x(frame)
    moved = (
        pre_handle_x is not None
        and post_handle_x is not None
        and abs(float(post_handle_x) - float(pre_handle_x)) >= 2.0
    )
    if moved:
        return

    try:
        iframe_el = frame.frame_element()
        iframe_box = iframe_el.bounding_box() if iframe_el else None
    except Exception:
        iframe_box = None

    if not iframe_box:
        LOG.warning("Drag attempt did not move slider; iframe box unavailable to offset coordinates")
        return

    LOG.info("Retrying drag with iframe offset correction")
    start2_x = iframe_box["x"] + start1_x
    start2_y = iframe_box["y"] + start1_y
    _perform_drag(start2_x, start2_y)


def _collect_bg_puzzle_images(page: Page, *, timeout_s: float) -> CaptchaImages:
    """
    Ensures captcha is opened, then collects bg/puzzle images.
    Strategy:
    - find the frame with Geetest
    - attempt to extract bg/slice URLs from Geetest elements
    - if not found, fallback to scanning candidates within that frame only
    """
    fr = _wait_for_geetest_open(page, timeout_s=timeout_s)

    bg_url, puzzle_url = _extract_geetest_bg_puzzle_urls(fr)
    if bg_url and puzzle_url:
        LOG.info("Extracted geetest urls directly")
        try:
            bg_bytes = _download_bytes(page, bg_url)
        except Exception:
            bg_bytes = _download_bytes_via_frame_fetch(fr, bg_url)
        try:
            puzzle_bytes = _download_bytes(page, puzzle_url)
        except Exception:
            puzzle_bytes = _download_bytes_via_frame_fetch(fr, puzzle_url)

        bg_bgr, _ = _decode_image_with_optional_alpha(bg_bytes)
        puzzle_bgr, puzzle_mask = _decode_image_with_optional_alpha(puzzle_bytes)
        return CaptchaImages(
            bg_url=bg_url,
            puzzle_url=puzzle_url,
            bg_bgr=bg_bgr,
            puzzle_bgr=puzzle_bgr,
            puzzle_mask_u8=puzzle_mask,
        )

    # Fallback: scan image urls inside geetest frame only
    LOG.info("Direct geetest urls not found; scanning frame images")
    urls: list[str] = fr.evaluate(
        """
        () => {
          const out = [];
          for (const img of Array.from(document.querySelectorAll("img"))) {
            const src = img.getAttribute("src") || "";
            if (src) out.push(src);
          }
          for (const el of Array.from(document.querySelectorAll("*"))) {
            const style = window.getComputedStyle(el);
            const bg = style && style.backgroundImage ? style.backgroundImage : "";
            if (bg && bg.includes("url(")) out.push(bg);
          }
          return out;
        }
        """
    )
    extracted: list[str] = []
    for u in urls:
        if "url(" in u:
            extracted.extend(_urls_from_style_background_image(u))
        else:
            extracted.append(u)
    extracted = [u for u in extracted if u.startswith("http")]
    extracted = _unique_preserve_order(extracted)

    if len(extracted) < 2:
        raise RuntimeError("Not enough candidate image URLs found after captcha open")

    # Download/decode using frame fetch fallback
    decoded: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for url in extracted:
        try:
            try:
                buf = _download_bytes(page, url)
            except Exception:
                buf = _download_bytes_via_frame_fetch(fr, url)
            bgr, mask = _decode_image_with_optional_alpha(buf)
            h, w = bgr.shape[:2]
            if w < 40 or h < 40:
                continue
            decoded.append((url, bgr, mask))
        except Exception:
            continue

    if len(decoded) < 2:
        raise RuntimeError("Failed to download/decode at least 2 images after captcha open")

    decoded.sort(key=lambda x: (x[1].shape[0] * x[1].shape[1]), reverse=True)
    bg_url2, bg_bgr2, _ = decoded[0]
    puzzle_url2, puzzle_bgr2, puzzle_mask2 = decoded[1]
    return CaptchaImages(
        bg_url=bg_url2,
        puzzle_url=puzzle_url2,
        bg_bgr=bg_bgr2,
        puzzle_bgr=puzzle_bgr2,
        puzzle_mask_u8=puzzle_mask2,
    )


def _extract_candidate_image_urls(page: Page) -> list[str]:
    urls: list[str] = page.evaluate(
        """
        () => {
          const out = [];

          // <img src=...>
          for (const img of Array.from(document.querySelectorAll("img"))) {
            const src = img.getAttribute("src") || "";
            if (src) out.push(src);
          }

          // background-image: url(...)
          for (const el of Array.from(document.querySelectorAll("*"))) {
            const style = window.getComputedStyle(el);
            const bg = style && style.backgroundImage ? style.backgroundImage : "";
            if (bg && bg.includes("url(")) out.push(bg);
          }

          return out;
        }
        """
    )

    # convert `background-image` values -> urls
    extracted: list[str] = []
    for u in urls:
        if "url(" in u:
            extracted.extend(_urls_from_style_background_image(u))
        else:
            extracted.append(u)

    # crude filter: keep likely image URLs
    extracted = [u for u in extracted if u.startswith("http") and any(x in u.lower() for x in [".png", ".jpg", ".jpeg", "image"])]
    return _unique_preserve_order(extracted)


def _pick_bg_and_puzzle_by_size(
    page: Page,
    candidate_urls: list[str],
    *,
    min_candidates: int = 2,
) -> CaptchaImages:
    if len(candidate_urls) < min_candidates:
        raise RuntimeError("Not enough candidate image URLs found on page")

    decoded: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for url in candidate_urls:
        try:
            buf = _download_bytes(page, url)
            bgr, mask = _decode_image_with_optional_alpha(buf)
            h, w = bgr.shape[:2]
            # ignore tiny icons/sprites
            if w < 40 or h < 40:
                continue
            decoded.append((url, bgr, mask))
        except Exception:
            continue

    if len(decoded) < 2:
        raise RuntimeError("Failed to download/decode at least 2 images from page candidates")

    # bg обычно существенно больше puzzle по площади
    decoded.sort(key=lambda x: (x[1].shape[0] * x[1].shape[1]), reverse=True)
    bg_url, bg_bgr, _ = decoded[0]
    puzzle_url, puzzle_bgr, puzzle_mask = decoded[1]
    return CaptchaImages(
        bg_url=bg_url,
        puzzle_url=puzzle_url,
        bg_bgr=bg_bgr,
        puzzle_bgr=puzzle_bgr,
        puzzle_mask_u8=puzzle_mask,
    )


def _save_optional(out_dir: Path, imgs: CaptchaImages) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "bg.png"), imgs.bg_bgr)
    cv2.imwrite(str(out_dir / "puzzle.png"), imgs.puzzle_bgr)
    if imgs.puzzle_mask_u8 is not None:
        cv2.imwrite(str(out_dir / "puzzle_mask.png"), imgs.puzzle_mask_u8)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open a page via Camoufox, collect Geetest bg/puzzle images, estimate puzzle x-offset by gradient matching."
    )
    parser.add_argument(
        "--url",
        default="https://www.geetest.com/en/demo",
        help="Target page URL (default: Geetest demo)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional output dir to save collected bg/puzzle")
    parser.add_argument("--method", choices=["sobel", "laplacian"], default="sobel", help="Gradient method")
    parser.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel size (odd >=3). 0 disables.")
    parser.add_argument("--timeout", type=float, default=25.0, help="Overall wait for captcha images (seconds)")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode (no window)")
    parser.add_argument(
        "--drag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After estimating x-offset, drag the Geetest slider via Camoufox internal cursor (default: enabled).",
    )
    parser.add_argument(
        "--drag-fudge",
        type=float,
        default=0.0,
        help="Constant adjustment (px) added to computed drag distance. Useful for calibration.",
    )
    parser.add_argument(
        "--post-drag-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after releasing the slider (for verification animation/network).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with Camoufox(humanize=False, headless=args.headless, window=CAMOUFOX_WINDOW) as browser:
        context = browser.new_context(
            viewport=CAMOUFOX_VIEWPORT,
            screen=CAMOUFOX_SCREEN,
            device_scale_factor=1.0,
            is_mobile=False,
        )
        page = context.new_page()
        page.set_viewport_size(CAMOUFOX_VIEWPORT)

        LOG.info("Opening %s", args.url)
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=int(args.timeout * 1000))
        except PlaywrightTimeoutError:
            LOG.warning("page.goto timeout; continuing")

        _dismiss_cookie_banners(page)
        _open_geetest_demo_captcha(page)
        imgs = _collect_bg_puzzle_images(page, timeout_s=args.timeout)
        LOG.info("bg_url=%s", imgs.bg_url)
        LOG.info("puzzle_url=%s", imgs.puzzle_url)

        if args.out_dir is not None:
            _save_optional(args.out_dir, imgs)
            LOG.info("Saved collected images to %s", args.out_dir)

        (x, y), score = find_best_match_on_gradients(
            imgs.bg_bgr,
            imgs.puzzle_bgr,
            puzzle_mask_u8=imgs.puzzle_mask_u8,
            method=args.method,
            blur_ksize=args.blur,
        )

        LOG.info("match score=%.4f", score)
        LOG.info("match top_left x=%d y=%d", x, y)

        if args.drag:
            fr = _wait_for_geetest_open(page, timeout_s=args.timeout)
            bg_w = _get_geetest_bg_display_width_px(fr)
            img_w = float(imgs.bg_bgr.shape[1])
            if bg_w is None or bg_w <= 0:
                LOG.warning("Could not read bg display width from DOM; using image width ratio=1.0")
                ratio = 1.0
            else:
                ratio = float(bg_w) / img_w

            drag_distance = float(x) * ratio + float(args.drag_fudge)
            LOG.info("Dragging slider by %.2f px (x=%d, ratio=%.4f, fudge=%.2f)", drag_distance, x, ratio, args.drag_fudge)
            _drag_geetest_slider(page, fr, distance_px=drag_distance)
            time.sleep(max(0.0, float(args.post_drag_wait)))

        print(x)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

