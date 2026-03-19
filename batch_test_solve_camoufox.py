from __future__ import annotations

import argparse
import logging
import time

from camoufox.sync_api import Camoufox
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from collect_solve_camoufox import (
    CAMOUFOX_SCREEN,
    CAMOUFOX_VIEWPORT,
    CAMOUFOX_WINDOW,
    _collect_bg_puzzle_images,
    _dismiss_cookie_banners,
    _drag_geetest_slider,
    _get_geetest_bg_display_width_px,
    _open_geetest_demo_captcha,
    _wait_for_geetest_open,
)
from gradient_highlight import find_best_match_on_gradients


LOG = logging.getLogger("batch_test_solve_camoufox")

SUCCESS_XPATH = "//*[text()='Verification Success']"


def _refresh_captcha(page: Page, *, explicit_wait_s: float, after_captcha_open_delay_s: float) -> None:
    try:
        tab3 = page.wait_for_selector(
            "//div[@class='tab-item tab-item-3']",
            timeout=int(explicit_wait_s * 1000),
        )
        page.evaluate("el => el.click()", tab3)
        time.sleep(0.5)
    except PlaywrightTimeoutError:
        print("[demo] Failed to click tab-item-3, continuing...")

    try:
        tab1 = page.wait_for_selector(
            "//div[@class='tab-item tab-item-1']",
            timeout=int(explicit_wait_s * 1000),
        )
        page.evaluate("el => el.click()", tab1)
        time.sleep(0.5)
    except PlaywrightTimeoutError:
        print("[demo] Failed to click tab-item-1, continuing...")

    try:
        btn_verify = page.wait_for_selector(
            "css=div.geetest_btn_click[aria-label='Click to verify']",
            timeout=int(explicit_wait_s * 1000),
        )
        btn_verify.click()
        time.sleep(after_captcha_open_delay_s)
    except PlaywrightTimeoutError as exc:
        print(f"[demo] Failed to open next CAPTCHA: {exc}")

    time.sleep(1.0)


def _solve_current_captcha(
    page: Page,
    *,
    timeout_s: float,
    drag_fudge_px: float,
    post_drag_wait_s: float,
    method: str,
    blur_ksize: int,
) -> None:
    imgs = _collect_bg_puzzle_images(page, timeout_s=timeout_s)
    (x, _y), score = find_best_match_on_gradients(
        imgs.bg_bgr,
        imgs.puzzle_bgr,
        puzzle_mask_u8=imgs.puzzle_mask_u8,
        method=method,
        blur_ksize=blur_ksize,
    )
    LOG.info("match score=%.4f, x=%d", float(score), int(x))

    fr = _wait_for_geetest_open(page, timeout_s=timeout_s)
    bg_w = _get_geetest_bg_display_width_px(fr)
    img_w = float(imgs.bg_bgr.shape[1])
    ratio = (float(bg_w) / img_w) if (bg_w is not None and bg_w > 0) else 1.0
    drag_distance = float(x) * ratio + float(drag_fudge_px)

    _drag_geetest_slider(page, fr, distance_px=drag_distance)
    time.sleep(max(0.0, float(post_drag_wait_s)))


def _format_stats(ok: int, bad: int) -> str:
    total = ok + bad
    rate = (ok / total * 100.0) if total else 0.0
    return f"ok={ok} bad={bad} total={total} success={rate:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run N solve attempts and report success/fail statistics.")
    parser.add_argument("--n", type=int, default=100, help="Number of tests (default: 100)")
    parser.add_argument("--url", default="https://www.geetest.com/en/demo", help="Target page URL")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--timeout", type=float, default=25.0, help="Captcha open/images timeout (seconds)")
    parser.add_argument("--explicit-wait", type=float, default=15.0, help="UI wait timeout for refresh clicks (seconds)")
    parser.add_argument("--after-open-delay", type=float, default=2.0, help="Delay after opening captcha (seconds)")
    parser.add_argument("--post-drag-wait", type=float, default=2.0, help="Wait after releasing slider (seconds)")
    parser.add_argument("--drag-fudge", type=float, default=0.0, help="Constant px adjustment added to drag distance")
    parser.add_argument("--method", choices=["sobel", "laplacian"], default="sobel", help="Gradient method")
    parser.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel size (odd >=3). 0 disables.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ok = 0
    bad = 0

    with Camoufox(humanize=False, headless=bool(args.headless), window=CAMOUFOX_WINDOW) as browser:
        context = browser.new_context(
            viewport=CAMOUFOX_VIEWPORT,
            screen=CAMOUFOX_SCREEN,
            device_scale_factor=1.0,
            is_mobile=False,
        )
        page = context.new_page()
        page.set_viewport_size(CAMOUFOX_VIEWPORT)

        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=int(args.timeout * 1000))
        except PlaywrightTimeoutError:
            LOG.warning("page.goto timeout; continuing")

        _dismiss_cookie_banners(page)
        _open_geetest_demo_captcha(page, explicit_wait_s=float(args.explicit_wait))

        for i in range(1, int(args.n) + 1):
            print(f"\n[test] ===== Attempt #{i}/{args.n} =====")
            attempt_ok = False

            try:
                _solve_current_captcha(
                    page,
                    timeout_s=float(args.timeout),
                    drag_fudge_px=float(args.drag_fudge),
                    post_drag_wait_s=float(args.post_drag_wait),
                    method=str(args.method),
                    blur_ksize=int(args.blur),
                )

                try:
                    page.wait_for_selector(SUCCESS_XPATH, timeout=3000)
                    attempt_ok = True
                except PlaywrightTimeoutError:
                    attempt_ok = False
            except Exception as exc:
                LOG.exception("Solve failed: %s", exc)
                attempt_ok = False

            if attempt_ok:
                ok += 1
                print(f"[test] ✅ success ({_format_stats(ok, bad)})")
            else:
                bad += 1
                print(f"[test] ❌ fail ({_format_stats(ok, bad)})")

            if i != int(args.n):
                _refresh_captcha(
                    page,
                    explicit_wait_s=float(args.explicit_wait),
                    after_captcha_open_delay_s=float(args.after_open_delay),
                )

        print(f"\n[test] DONE: {_format_stats(ok, bad)}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

