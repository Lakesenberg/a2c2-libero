import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def detect_outer_black_rectangle(hsv_image: np.ndarray, debug_image: np.ndarray):
    """黒い枠の外側長方形輪郭（4頂点, Nx1x2）を返す。見つからなければNone。"""
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])

    mask = cv2.inRange(hsv_image, lower_black, upper_black)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        approx = np.int32(box).reshape(-1, 1, 2)

    cv2.drawContours(debug_image, [approx], -1, (0, 255, 0), 2)
    return approx, mask


def detect_yellow_block(hsv_image: np.ndarray, debug_image: np.ndarray):
    """黄色ブロックの重心座標と輪郭を返す。見つからなければNone。"""
    lower_yellow = np.array([15, 90, 90])
    upper_yellow = np.array([35, 255, 255])

    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, mask

    target = max(contours, key=cv2.contourArea)
    M = cv2.moments(target)
    if M["m00"] == 0:
        return None, None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    cv2.circle(debug_image, (cx, cy), 2, (0, 0, 255), -1)
    return (cx, cy), target, mask


def _fit_line_from_segments(segments: np.ndarray):
    """HoughSegments(?,4) -> robustな直線をcv2.fitLineでフィットして返す。"""
    if segments is None or len(segments) == 0:
        return None
    points = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        points.append([x1, y1])
        points.append([x2, y2])
    pts = np.array(points, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    return float(vx), float(vy), float(x0), float(y0)


def _intersection_of_lines(l1, l2):
    """2つの無限直線の交点を返す。各直線は(vx,vy,x0,y0)。交点が無ければNone。"""
    if l1 is None or l2 is None:
        return None
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    # 連立: [vx1, -vx2] [t] = x2-x1, [vy1, -vy2] [s] = y2-y1
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    t, s = np.linalg.solve(A, b)
    xi = x1 + vx1 * t
    yi = y1 + vy1 * t
    return int(round(xi)), int(round(yi))


def order_rectangle_points(points_xy: np.ndarray) -> np.ndarray:
    """4点を (tl, tr, br, bl) に並べ替える。points_xy: (4,2)。"""
    pts = points_xy.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    return ordered


def uv_to_xy_on_quad(quad_points_tl_tr_br_bl: np.ndarray, u: float, v: float) -> tuple[int, int]:
    """単位正方形上の(u,v) (0..1) を与え、四角形上の画素座標(x,y)を返す。"""
    u = float(min(max(u, 0.0), 1.0))
    v = float(min(max(v, 0.0), 1.0))
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    dst = quad_points_tl_tr_br_bl.astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    pt = np.array([[[u, v]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, M)[0, 0]
    return int(round(out[0])), int(round(out[1]))


def parse_uv_arg(x_uv_args):
    """--x-uv の入力を柔軟に解析し (u, v) を返す。
    例: ["0.5", "0.5"] / ["0.5,0.5"] / ["0.5;0.5"]
    """
    if x_uv_args is None:
        return None
    # 2トークン (u v)
    if isinstance(x_uv_args, (list, tuple)) and len(x_uv_args) == 2:
        try:
            return float(x_uv_args[0]), float(x_uv_args[1])
        except ValueError:
            return None
    # 1トークン ("u,v")
    if isinstance(x_uv_args, (list, tuple)) and len(x_uv_args) == 1:
        token = str(x_uv_args[0]).replace(",", " ").replace(";", " ")
        parts = token.split()
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                return None
    return None


def main():
    parser = argparse.ArgumentParser(description="矩形枠と黄色ブロックの位置検出（画像/動画対応）")
    parser.add_argument("input", type=str, help="入力ファイルのパス（画像 or 動画）")
    parser.add_argument("--show-masks", action="store_true", help="マスクウィンドウも表示")
    parser.add_argument("--debug", action="store_true", help="詳細ログと可視化を表示")
    parser.add_argument("--out", type=str, help="結果画像の保存先パス（例: out/result.png）")
    parser.add_argument("--out-video", type=str, help="結果動画の保存先パス（例: out/result.mp4）")
    parser.add_argument("--size-mm", type=float, nargs=2, metavar=("W", "H"),
                        default=[236.0, 333.0],
                        help="枠の実寸(mm)。デフォルトはA4短辺x長辺 (210 297) + 18mmのテープ が上下左右に18mmずつ出ている つまり、236x333")
    parser.add_argument("--x-uv", nargs="+", metavar=("U", "V"),
                        help="内側矩形に対するバツ印の規格化位置。'u v' または 'u,v' 形式をサポート")
    parser.add_argument("--crop", nargs="+", metavar="CROP",
                        help="処理する矩形領域。'x y w h' または 'x,y,w,h'。0..1なら割合、>1はpx",
                        default=["0.35,0.47,0.40,0.30"])  # 左上x,y, 幅, 高さ (割合)
    parser.add_argument("--center-gate", type=float, default=0.15,
                        help="青クロスが画像中心からこの割合(短辺×比)以内のときだけ距離を評価")
    parser.add_argument("--hit-dir", type=str,
                        help="center gate を満たしたフレームを保存するディレクトリ",default="hit")
    parser.add_argument("--jsonl", type=str,
                        help="結果を書き出すJSONLファイル。各行に {file|video, distance_mm, ...}")

    args = parser.parse_args()
    input_path = Path(args.input)

    def _parse_crop_arg(arg):
        if arg is None:
            return None
        if isinstance(arg, (list, tuple)) and len(arg) == 4:
            try:
                return [float(a) for a in arg]
            except ValueError:
                pass
        if isinstance(arg, (list, tuple)) and len(arg) == 1:
            token = str(arg[0]).replace(",", " ")
            parts = token.split()
            if len(parts) == 4:
                try:
                    return [float(p) for p in parts]
                except ValueError:
                    pass
        return None

    def analyze_frame(image):
        # 任意クロップが優先。未指定なら中心クロップを適用。
        crop = _parse_crop_arg(args.crop)
        if crop is not None:
            x, y, w, h = crop
            H, W = image.shape[:2]
            # 0..1 の場合は割合として解釈
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
                x = int(round(x * W))
                y = int(round(y * H))
                w = int(round(w * W))
                h = int(round(h * H))
            else:
                x = int(round(x))
                y = int(round(y))
                w = int(round(w))
                h = int(round(h))
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))
            w = max(1, min(W - x, w))
            h = max(1, min(H - y, h))
            image = image[y:y + h, x:x + w].copy()
        else:
            s = float(max(0.05, min(1.0, args.center_crop)))
            if s < 1.0:
                h, w = image.shape[:2]
                ch, cw = int(h * s), int(w * s)
                y0 = (h - ch) // 2
                x0 = (w - cw) // 2
                image = image[y0:y0 + ch, x0:x0 + cw].copy()

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        rect_poly, black_mask = detect_outer_black_rectangle(hsv, image)
        yellow_center, yellow_contour, yellow_mask = detect_yellow_block(hsv, image)

        ordered = None
        if rect_poly is not None:
            ordered = order_rectangle_points(rect_poly.reshape(-1, 2))
            if args.debug:
                print("枠の4頂点(時計回り):", ordered.tolist())
        else:
            if args.debug:
                print("黒い長方形が見つかりませんでした。")

        if yellow_center is None and args.debug:
            print("黄色いブロックが見つかりませんでした。")

        cross_center = None
        if ordered is not None and args.x_uv is not None:
            parsed = parse_uv_arg(args.x_uv)
            if parsed is None:
                if args.debug:
                    print("--x-uv の形式が不正です。'u v' または 'u,v' で指定してください。")
            else:
                u, v = parsed
                cross_center = uv_to_xy_on_quad(ordered, u, v)
                # バツ印の位置を常に描画
                marker_size = max(3, 3)
                cv2.drawMarker(image, cross_center, (255, 0, 0),
                               markerType=cv2.MARKER_TILTED_CROSS,
                               markerSize=marker_size, thickness=2)

        d_mm = None
        show_gate = False
        if ordered is not None and yellow_center is not None and cross_center is not None:
            # 中心ゲート: 青クロスがフレーム中心近傍のときのみ評価
            Hc, Wc = image.shape[:2]
            cx_img, cy_img = Wc / 2.0, Hc / 2.0
            gate_radius = min(Wc, Hc) * float(max(0.0, args.center_gate))
            dist_center = np.hypot(cross_center[0] - cx_img, cross_center[1] - cy_img)
            if dist_center <= gate_radius:
                show_gate = True
                dx = yellow_center[0] - cross_center[0]
                dy = yellow_center[1] - cross_center[1]
                d_px = float(np.hypot(dx, dy))
                w_px = np.linalg.norm(ordered[1] - ordered[0])
                h_px = np.linalg.norm(ordered[3] - ordered[0])
                w_mm, h_mm = float(args.size_mm[0]), float(args.size_mm[1])
                sx = w_mm / w_px
                sy = h_mm / h_px
                s_mm_per_px = (sx + sy) / 2.0
                d_mm = d_px * s_mm_per_px
                if args.debug:
                    print(f"黄色ブロック中心とバツ印中心の距離: {d_mm:.2f} mm (scale {s_mm_per_px:.3f} mm/px)")
            else:
                # 中心から外れているときは評価しない
                pass

        return image, (black_mask if rect_poly is not None else None), (yellow_mask if yellow_center is not None else None), show_gate, d_mm

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

    def process_video(video_path: Path, out_dir_for_video: Path | None, hit_dir_root: Path | None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"エラー: 動画ファイル '{video_path}' を開けませんでした。")
            return None

        writer = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        distances_mm = []
        frame_idx = 0
        hit_dir = None
        if hit_dir_root is not None:
            hit_dir = hit_dir_root / video_path.stem
            hit_dir.mkdir(parents=True, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, black_mask, yellow_mask, show_gate, d_mm = analyze_frame(frame)
            if d_mm is not None:
                distances_mm.append(d_mm)

            if out_dir_for_video is not None:
                if writer is None:
                    out_dir_for_video.mkdir(parents=True, exist_ok=True)
                    h_out, w_out = annotated.shape[:2]
                    out_path = out_dir_for_video / f"{video_path.stem}_result.mp4"
                    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_out, h_out))
                writer.write(annotated)

            if show_gate and hit_dir is not None:
                saved_path = hit_dir / f"hit_{frame_idx:06d}.png"
                cv2.imwrite(str(saved_path), annotated)

            if args.show_masks and black_mask is not None and yellow_mask is not None and frame_idx == 0:
                cv2.imshow("black_mask", black_mask)
                cv2.imshow("yellow_mask", yellow_mask)
            if args.debug and show_gate:
                cv2.imshow("Result", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()
        if args.debug or args.show_masks:
            cv2.destroyAllWindows()

        if len(distances_mm) > 0:
            median_mm = float(np.median(np.array(distances_mm)))
            print(f"Median distance for '{video_path.name}': {median_mm:.2f} mm (N={len(distances_mm)})")
            if args.jsonl is not None:
                jp = Path(args.jsonl)
                jp.parent.mkdir(parents=True, exist_ok=True)
                with open(jp, "a", encoding="utf-8") as f:
                    json.dump({"video": video_path.name, "median_mm": median_mm, "n": int(len(distances_mm))}, f, ensure_ascii=False)
                    f.write("\n")
            return median_mm
        else:
            print(f"No gated samples for '{video_path.name}'.")
            return None

    if input_path.is_dir():
        out_dir_for_video = Path(args.out_video) if args.out_video is not None else None
        hit_dir_root = Path(args.hit_dir) if args.hit_dir is not None else None
        video_files = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in video_exts])
        if len(video_files) == 0:
            print(f"ディレクトリ '{input_path}' に動画が見つかりませんでした。")
            return
        medians = []
        for vp in video_files:
            m = process_video(vp, out_dir_for_video, hit_dir_root)
            if m is not None:
                medians.append(m)
        if len(medians) > 0:
            overall_median = float(np.median(np.array(medians)))
            print(f"Overall median across {len(medians)} videos: {overall_median:.2f} mm")
        return
    elif input_path.suffix.lower() in video_exts:
        out_dir_for_video = None
        if args.out_video is not None:
            out_dir_for_video = Path(args.out_video)
            if out_dir_for_video.suffix:
                out_dir_for_video = out_dir_for_video.parent
        hit_dir_root = Path(args.hit_dir) if args.hit_dir is not None else None
        process_video(input_path, out_dir_for_video, hit_dir_root)
    else:
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"エラー: 画像ファイル '{input_path}' を読み込めませんでした。")
            return
        annotated, black_mask, yellow_mask, show_gate, d_mm = analyze_frame(image)
        if d_mm is not None:
            print(f"Median distance for image: {d_mm:.2f} mm (N=1)")
            if args.jsonl is not None:
                jp = Path(args.jsonl)
                jp.parent.mkdir(parents=True, exist_ok=True)
                with open(jp, "a", encoding="utf-8") as f:
                    json.dump({"file": str(input_path), "distance_mm": float(d_mm)}, f, ensure_ascii=False)
                    f.write("\n")

        # 画像の保存
        if args.out is not None:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), annotated)
            if args.debug:
                print(f"結果画像を保存しました: {out_path}")

        if args.show_masks:
            if black_mask is not None:
                cv2.imshow("black_mask", black_mask)
            if yellow_mask is not None:
                cv2.imshow("yellow_mask", yellow_mask)
        if args.debug or args.show_masks:
            cv2.imshow("Result", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()