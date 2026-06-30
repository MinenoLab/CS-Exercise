from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

try:
    from experiment_paths import DEFAULT_EXPERIMENTS_DIR, ensure_experiment_dirs, experiment_dir, find_default_model, format_float_for_name, resolve_experiment_name, write_json
except ModuleNotFoundError:
    from scripts.experiment_paths import DEFAULT_EXPERIMENTS_DIR, ensure_experiment_dirs, experiment_dir, find_default_model, format_float_for_name, resolve_experiment_name, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "custom_yolo"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0
    conf: float | None = None


@dataclass
class ImageErrorReport:
    image_path: Path
    label_path: Path
    gt_count: int
    pred_count: int
    tp: int
    fp: int
    fn: int
    max_iou: float
    mean_matched_iou: float
    max_conf: float
    status: str

    @property
    def has_error(self) -> bool:
        return self.fp > 0 or self.fn > 0 or self.status == "missing_label"


@dataclass
class MatchResult:
    matched_gt: set[int]
    matched_pred: set[int]
    matched_ious: list[float]
    max_iou: float

    @property
    def tp(self) -> int:
        return len(self.matched_ious)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO推論結果と正解ラベルを比較し、間違えた画像リストを出力します。")
    parser.add_argument("--model", default=str(find_default_model()))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--experiment", default="", help="実験名。未指定ならモデルパスから推定します。")
    parser.add_argument("--output-dir", default="", help="分析結果の保存先を直接指定します。")
    parser.add_argument("--name", default="", help="error_analysis内の出力名。")
    parser.add_argument("--save-images", action="store_true", help="間違えた画像に正解枠と検出枠を描画して保存します。")
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise SystemExit(f"[ERROR] 画像ディレクトリが見つかりません: {images_dir}")
    return [path for path in sorted(images_dir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def read_yolo_labels(label_path: Path, image_width: int, image_height: int) -> list[Box]:
    boxes: list[Box] = []
    if not label_path.exists():
        return boxes
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            print(f"[WARN] 不正なラベル行をスキップします: {label_path}:{line_number}")
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            print(f"[WARN] 数値変換できないラベル行をスキップします: {label_path}:{line_number}")
            continue
        box_width = width * image_width
        box_height = height * image_height
        center_x = x_center * image_width
        center_y = y_center * image_height
        boxes.append(Box(center_x - box_width / 2, center_y - box_height / 2, center_x + box_width / 2, center_y + box_height / 2, class_id=class_id))
    return boxes


def read_predictions(model: YOLO, image_path: Path, args: argparse.Namespace) -> list[Box]:
    results = model.predict(source=str(image_path), conf=args.conf, iou=args.nms_iou, imgsz=args.imgsz, device=args.device, verbose=False)
    if not results or results[0].boxes is None:
        return []
    result_boxes = results[0].boxes
    predictions: list[Box] = []
    for xyxy, conf, class_id in zip(result_boxes.xyxy.cpu().tolist(), result_boxes.conf.cpu().tolist(), result_boxes.cls.cpu().tolist(), strict=True):
        predictions.append(Box(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), class_id=int(class_id), conf=float(conf)))
    return predictions


def box_iou(a: Box, b: Box) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union_area = area_a + area_b - inter_area
    return 0.0 if union_area <= 0 else inter_area / union_area


def match_boxes(ground_truths: list[Box], predictions: list[Box], iou_threshold: float) -> MatchResult:
    candidates: list[tuple[float, int, int]] = []
    max_iou = 0.0
    for gt_index, gt_box in enumerate(ground_truths):
        for pred_index, pred_box in enumerate(predictions):
            if gt_box.class_id != pred_box.class_id:
                continue
            iou = box_iou(gt_box, pred_box)
            max_iou = max(max_iou, iou)
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))
    candidates.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matched_ious: list[float] = []
    for iou, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matched_ious.append(iou)
    return MatchResult(matched_gt, matched_pred, matched_ious, max_iou)


def make_status(fp: int, fn: int, missing_label: bool) -> str:
    if missing_label:
        return "missing_label"
    if fp > 0 and fn > 0:
        return "false_positive+false_negative"
    if fp > 0:
        return "false_positive"
    if fn > 0:
        return "false_negative"
    return "ok"


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def analyze_image(model: YOLO, image_path: Path, labels_dir: Path, args: argparse.Namespace) -> tuple[ImageErrorReport, list[Box], list[Box], MatchResult]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"画像を読み込めません: {image_path}")
    image_height, image_width = image.shape[:2]
    label_path = labels_dir / f"{image_path.stem}.txt"
    missing_label = not label_path.exists()
    ground_truths = read_yolo_labels(label_path, image_width, image_height)
    predictions = read_predictions(model, image_path, args)
    match_result = match_boxes(ground_truths, predictions, args.match_iou)
    fp = len(predictions) - match_result.tp
    fn = len(ground_truths) - match_result.tp
    mean_iou = sum(match_result.matched_ious) / len(match_result.matched_ious) if match_result.matched_ious else 0.0
    max_conf = max((box.conf or 0.0 for box in predictions), default=0.0)
    report = ImageErrorReport(image_path, label_path, len(ground_truths), len(predictions), match_result.tp, fp, fn, match_result.max_iou, mean_iou, max_conf, make_status(fp, fn, missing_label))
    return report, ground_truths, predictions, match_result


def draw_boxes(image_path: Path, ground_truths: list[Box], predictions: list[Box], match_result: MatchResult, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for index, box in enumerate(ground_truths):
        color = (0, 180, 0) if index in match_result.matched_gt else (0, 220, 255)
        label = "TP" if index in match_result.matched_gt else "FN"
        pt1 = (int(round(box.x1)), int(round(box.y1)))
        pt2 = (int(round(box.x2)), int(round(box.y2)))
        cv2.rectangle(image, pt1, pt2, color, 2)
        cv2.putText(image, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for index, box in enumerate(predictions):
        color = (0, 180, 0) if index in match_result.matched_pred else (0, 0, 220)
        prefix = "TP" if index in match_result.matched_pred else "FP"
        label = f"{prefix} {box.conf:.2f}" if box.conf is not None else prefix
        pt1 = (int(round(box.x1)), int(round(box.y1)))
        pt2 = (int(round(box.x2)), int(round(box.y2)))
        cv2.rectangle(image, pt1, pt2, color, 2)
        cv2.putText(image, label, (pt1[0], max(0, pt1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def write_reports(reports: list[ImageErrorReport], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "label", "gt_count", "pred_count", "tp", "fp", "fn", "max_iou", "mean_matched_iou", "max_conf", "status"]

    def row(report: ImageErrorReport) -> dict[str, str | int]:
        return {
            "image": relative_path(report.image_path),
            "label": relative_path(report.label_path),
            "gt_count": report.gt_count,
            "pred_count": report.pred_count,
            "tp": report.tp,
            "fp": report.fp,
            "fn": report.fn,
            "max_iou": f"{report.max_iou:.6f}",
            "mean_matched_iou": f"{report.mean_matched_iou:.6f}",
            "max_conf": f"{report.max_conf:.6f}",
            "status": report.status,
        }

    subsets = {
        "all_images": reports,
        "mistake_images": [report for report in reports if report.has_error],
        "false_positive_images": [report for report in reports if report.fp > 0],
        "false_negative_images": [report for report in reports if report.fn > 0],
    }
    for name, subset in subsets.items():
        with (output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row(report) for report in subset)
        (output_dir / f"{name}.txt").write_text("\n".join(relative_path(report.image_path) for report in subset) + ("\n" if subset else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / "images" / args.split
    labels_dir = dataset_dir / "labels" / args.split
    if not model_path.exists():
        raise SystemExit(f"[ERROR] モデルが見つかりません: {model_path}")
    if not labels_dir.exists():
        raise SystemExit(f"[ERROR] ラベルディレクトリが見つかりません: {labels_dir}")
    images = list_images(images_dir)
    if not images:
        raise SystemExit(f"[ERROR] 画像が見つかりません: {images_dir}")

    experiment_name = resolve_experiment_name(args.experiment, model_path)
    exp_dir = experiment_dir(Path(args.experiments_dir), experiment_name)
    ensure_experiment_dirs(exp_dir)
    output_name = args.name or f"{args.split}_conf{format_float_for_name(args.conf)}_iou{format_float_for_name(args.match_iou)}"
    output_parent = Path(args.output_dir) if args.output_dir else exp_dir / "error_analysis"
    output_dir = output_parent / output_name

    model = YOLO(str(model_path))
    reports: list[ImageErrorReport] = []
    for image_path in tqdm(images, desc="analyze"):
        report, ground_truths, predictions, match_result = analyze_image(model, image_path, labels_dir, args)
        reports.append(report)
        if args.save_images and report.has_error:
            draw_boxes(image_path, ground_truths, predictions, match_result, output_dir / "mistake_images" / image_path.name)
        if args.save_images and report.fp > 0:
            draw_boxes(image_path, ground_truths, predictions, match_result, output_dir / "false_positive_images" / image_path.name)
        if args.save_images and report.fn > 0:
            draw_boxes(image_path, ground_truths, predictions, match_result, output_dir / "false_negative_images" / image_path.name)

    write_reports(reports, output_dir)
    write_json(output_dir / "analysis_metadata.json", {"experiment": experiment_name, "model": str(model_path), "dataset_dir": str(dataset_dir), "split": args.split, "conf": args.conf, "match_iou": args.match_iou, "nms_iou": args.nms_iou, "imgsz": args.imgsz, "device": args.device, "output_dir": str(output_dir)})
    print(f"[DONE] 分析結果を保存しました: {output_dir}")
    print(f"[DONE] 間違えた画像数: {sum(1 for report in reports if report.has_error)} / {len(reports)}")
    print(f"[DONE] FP合計: {sum(report.fp for report in reports)}")
    print(f"[DONE] FN合計: {sum(report.fn for report in reports)}")


if __name__ == "__main__":
    main()
