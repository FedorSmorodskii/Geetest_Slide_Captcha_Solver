from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _read_bgr_with_alpha_mask(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Reads image and returns:
      - bgr: (H,W,3)
      - mask_u8: (H,W) uint8 in {0,255} if alpha exists, else None
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return bgr, None
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        mask = (alpha > 0).astype(np.uint8) * 255
        return bgr, mask
    return img[:, :, :3], None


def _to_uint8_heatmap(mag: np.ndarray) -> np.ndarray:
    mag = np.asarray(mag, dtype=np.float32)
    if mag.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    mmin = float(mag.min())
    mmax = float(mag.max())
    if mmax <= mmin + 1e-12:
        return np.zeros(mag.shape[:2], dtype=np.uint8)
    mag_norm = (mag - mmin) / (mmax - mmin)
    return np.clip(mag_norm * 255.0, 0.0, 255.0).astype(np.uint8)


def gradient_maps(
    bgr: np.ndarray,
    *,
    method: str = "sobel",
    blur_ksize: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - heatmap_bgr: colorized gradient intensity (BGR)
      - mag_u8: gradient magnitude normalized to uint8 [0..255]
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    method = method.lower().strip()
    if method == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
    elif method == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        mag = np.abs(lap)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sobel' or 'laplacian'.")

    mag_u8 = _to_uint8_heatmap(mag)
    heatmap_bgr = cv2.applyColorMap(mag_u8, cv2.COLORMAP_TURBO)
    return heatmap_bgr, mag_u8


def overlay_heatmap(
    base_bgr: np.ndarray,
    heatmap_bgr: np.ndarray,
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    alpha = float(alpha)
    alpha = 0.0 if alpha < 0.0 else 1.0 if alpha > 1.0 else alpha
    return cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)


def find_best_match_on_gradients(
    bg_bgr: np.ndarray,
    puzzle_bgr: np.ndarray,
    *,
    puzzle_mask_u8: np.ndarray | None = None,
    method: str = "sobel",
    blur_ksize: int = 3,
) -> tuple[tuple[int, int], float]:
    """
    Matches puzzle inside bg using gradient magnitude images.
    Returns (top_left_xy, score). Higher score is better.
    """
    _, bg_mag_u8 = gradient_maps(bg_bgr, method=method, blur_ksize=blur_ksize)
    _, puz_mag_u8 = gradient_maps(puzzle_bgr, method=method, blur_ksize=blur_ksize)

    bg_mag = bg_mag_u8.astype(np.float32) / 255.0
    puz_mag = puz_mag_u8.astype(np.float32) / 255.0

    match_method = cv2.TM_CCORR_NORMED
    if puzzle_mask_u8 is not None:
        mask = (puzzle_mask_u8 > 0).astype(np.uint8)
        res = cv2.matchTemplate(bg_mag, puz_mag, match_method, mask=mask)
    else:
        res = cv2.matchTemplate(bg_mag, puz_mag, match_method)

    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return (int(max_loc[0]), int(max_loc[1])), float(max_val)


def paint_match_region(
    bg_bgr: np.ndarray,
    top_left: tuple[int, int],
    wh: tuple[int, int],
    *,
    color_bgr: tuple[int, int, int] = (0, 255, 255),
    fill_alpha: float = 0.35,
    border_thickness: int = 2,
) -> np.ndarray:
    x, y = top_left
    w, h = wh
    x2 = max(x + w, x)
    y2 = max(y + h, y)

    out = bg_bgr.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), color_bgr, thickness=-1)
    out = cv2.addWeighted(overlay, fill_alpha, out, 1.0 - fill_alpha, 0.0)
    cv2.rectangle(out, (x, y), (x2, y2), (0, 0, 0), thickness=border_thickness + 2)
    cv2.rectangle(out, (x, y), (x2, y2), color_bgr, thickness=border_thickness)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Highlight gradients (Sobel/Laplacian) on an image and save overlay."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/bg.png"),
        help="Input image path (default: data/bg.png)",
    )
    parser.add_argument(
        "--also-puzzle",
        type=Path,
        default=Path("data/puzzle.png"),
        help="Optional second image to highlight too (default: data/puzzle.png). Set to empty to disable.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--method",
        choices=["sobel", "laplacian"],
        default="sobel",
        help="Gradient method",
    )
    parser.add_argument(
        "--blur",
        type=int,
        default=3,
        help="Gaussian blur kernel size (odd >=3). Use 0 to disable.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Overlay alpha in [0..1] (default: 0.55)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show windows (press any key to close).",
    )
    parser.add_argument(
        "--match-puzzle",
        type=Path,
        default=None,
        help="Optional puzzle template path to localize on the input image.",
    )
    parser.add_argument(
        "--match-out",
        type=Path,
        default=None,
        help="Optional output png path for match visualization (defaults to out-dir/match_location.png).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def _process_one(img_path: Path, *, name_suffix: str | None) -> tuple[np.ndarray, np.ndarray]:
        bgr2, mask_u8 = _read_bgr_with_alpha_mask(img_path)
        heatmap2, _ = gradient_maps(bgr2, method=args.method, blur_ksize=args.blur)
        overlay2 = overlay_heatmap(bgr2, heatmap2, alpha=args.alpha)
        if mask_u8 is not None:
            keep = mask_u8 > 0
            heatmap2 = heatmap2.copy()
            overlay2 = overlay2.copy()
            heatmap2[~keep] = 0
            overlay2[~keep] = 0

        if name_suffix is None:
            heatmap_path2 = args.out_dir / "gradient_heatmap.png"
            overlay_path2 = args.out_dir / "gradient_overlay.png"
        else:
            heatmap_path2 = args.out_dir / f"gradient_heatmap_{name_suffix}.png"
            overlay_path2 = args.out_dir / f"gradient_overlay_{name_suffix}.png"

        cv2.imwrite(str(heatmap_path2), heatmap2)
        cv2.imwrite(str(overlay_path2), overlay2)
        print(f"Saved: {heatmap_path2}")
        print(f"Saved: {overlay_path2}")
        return heatmap2, overlay2

    primary_path = Path(args.input)
    heatmap_bgr, overlay = _process_one(primary_path, name_suffix=None)

    also_path = args.also_puzzle
    extra_heatmap = None
    extra_overlay = None
    if also_path is not None:
        try:
            also_path = Path(also_path)
            if also_path.exists() and also_path.resolve() != primary_path.resolve():
                extra_heatmap, extra_overlay = _process_one(also_path, name_suffix=also_path.stem)
        except OSError:
            pass

    if args.match_puzzle is not None:
        puz_bgr, puz_mask = _read_bgr_with_alpha_mask(args.match_puzzle)
        top_left, score = find_best_match_on_gradients(
            _read_bgr(primary_path),
            puz_bgr,
            puzzle_mask_u8=puz_mask,
            method=args.method,
            blur_ksize=args.blur,
        )
        h, w = puz_bgr.shape[:2]
        match_vis = paint_match_region(_read_bgr(primary_path), top_left, (w, h))
        out_path = args.match_out or (args.out_dir / "match_location.png")
        cv2.imwrite(str(out_path), match_vis)
        print(f"Match score: {score:.4f} at x={top_left[0]} y={top_left[1]}")
        print(f"Saved: {out_path}")

    if args.show:
        cv2.imshow("gradient heatmap", heatmap_bgr)
        cv2.imshow("gradient overlay", overlay)
        if extra_heatmap is not None and extra_overlay is not None:
            cv2.imshow("gradient heatmap (also)", extra_heatmap)
            cv2.imshow("gradient overlay (also)", extra_overlay)
        if args.match_puzzle is not None:
            cv2.imshow("match location", match_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

