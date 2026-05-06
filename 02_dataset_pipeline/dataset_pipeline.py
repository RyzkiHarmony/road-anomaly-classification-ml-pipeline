from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_CANDIDATE_COLS = {
    "event_id",
    "trip_id",
    "time_s",
    "lat",
    "lon",
    "peak_mag",
    "peak_vertical",
    "peak_gyro_mag",
    "vert_jrk",
}
REQUIRED_GT_COLS = {"event_id", "trip_id", "label"}


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_repo_path(repo_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs.replace("\\", "/"))
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def normalize_label_series(
    labels: pd.Series,
    classes: Iterable[str],
    aliases: Dict[str, str],
) -> Tuple[pd.Series, List[str]]:
    class_lookup = {c.lower(): c for c in classes}
    alias_lookup = {k.lower().strip(): v for k, v in aliases.items()}
    unknown: List[str] = []

    out: List[str] = []
    for raw in labels.fillna("").astype(str):
        text = raw.strip()
        if text == "":
            out.append("")
            continue

        key = text.lower()
        if key in class_lookup:
            out.append(class_lookup[key])
            continue
        if key in alias_lookup:
            out.append(alias_lookup[key])
            continue

        unknown.append(text)
        out.append(text)

    return pd.Series(out, index=labels.index), sorted(set(unknown))


def build_stable_event_id(row: pd.Series, stable_cfg: Dict) -> str:
    time_round_s = float(stable_cfg["time_round_s"])
    lat_lon_decimals = int(stable_cfg["lat_lon_decimals"])
    peak_mag_decimals = int(stable_cfg["peak_mag_decimals"])
    peak_gyro_decimals = int(stable_cfg["peak_gyro_decimals"])
    jerk_decimals = int(stable_cfg["jerk_decimals"])

    t = float(row["time_s"])
    time_bucket = round(t / time_round_s) * time_round_s

    payload = "|".join(
        [
            str(row["trip_id"]),
            f"{time_bucket:.3f}",
            f"{round(float(row['lat']), lat_lon_decimals):.{lat_lon_decimals}f}",
            f"{round(float(row['lon']), lat_lon_decimals):.{lat_lon_decimals}f}",
            f"{round(float(row['peak_vertical']), peak_mag_decimals):.{peak_mag_decimals}f}",
            f"{round(float(row['peak_gyro_mag']), peak_gyro_decimals):.{peak_gyro_decimals}f}",
            f"{round(float(row['vert_jrk']), jerk_decimals):.{jerk_decimals}f}",
        ]
    )
    return hash_text(payload)[:20]


def deterministic_trip_split(trip_ids: Iterable[str], split_cfg: Dict) -> pd.DataFrame:
    train_ratio = float(split_cfg["train_ratio"])
    val_ratio = float(split_cfg["val_ratio"])
    test_ratio = float(split_cfg["test_ratio"])
    seed = str(split_cfg["seed"])

    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Split ratio harus 1.0, saat ini {total}")

    rows = []
    train_bound = train_ratio
    val_bound = train_ratio + val_ratio
    for trip in sorted(set(trip_ids)):
        digest_int = int(hash_text(f"{seed}|{trip}")[:12], 16)
        unit = (digest_int % 1_000_000) / 1_000_000.0
        if unit < train_bound:
            split = "train"
        elif unit < val_bound:
            split = "val"
        else:
            split = "test"
        rows.append({"trip_id": trip, "split": split, "hash_unit": unit})
    return pd.DataFrame(rows)


def build_raw_signal_quality_report(repo_root: Path, config: Dict) -> pd.DataFrame:
    meta_pattern = resolve_repo_path(repo_root, config["input"]["meta_glob"])
    raw_dir = resolve_repo_path(repo_root, config["input"]["raw_csv_dir"])

    rows = []
    for meta_path_str in sorted(glob.glob(str(meta_pattern))):
        meta_path = Path(meta_path_str)
        meta = load_json(meta_path)
        trip_id = meta.get("tripId", "")
        csv_name = f"{meta_path.stem}.csv"
        csv_path = raw_dir / csv_name

        if not csv_path.exists():
            rows.append(
                {
                    "trip_id": trip_id,
                    "raw_csv": csv_name,
                    "status": "missing_csv",
                    "n_rows": 0,
                    "duplicate_timestamp_rows": 0,
                    "timestamp_unique_ratio": 0.0,
                    "estimated_fs_hz": np.nan,
                    "median_dt_ms": np.nan,
                    "p95_dt_ms": np.nan,
                    "irregular_dt_ratio": np.nan,
                }
            )
            continue

        df = pd.read_csv(csv_path, usecols=["timestamp"])
        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna().astype(float)
        n_rows = int(len(ts))
        n_unique = int(ts.nunique())
        dup_rows = n_rows - n_unique

        ts_sorted = np.sort(ts.values)
        diffs = np.diff(ts_sorted)
        diffs = diffs[diffs > 0]

        if diffs.size == 0:
            med_dt = np.nan
            p95_dt = np.nan
            fs = np.nan
            irr_ratio = np.nan
        else:
            med_dt = float(np.median(diffs))
            p95_dt = float(np.percentile(diffs, 95))
            fs = float(1000.0 / med_dt) if med_dt > 0 else np.nan
            irr_ratio = float(np.mean(np.abs(diffs - med_dt) > (0.5 * med_dt))) if med_dt > 0 else np.nan

        rows.append(
            {
                "trip_id": trip_id,
                "raw_csv": csv_name,
                "status": "ok",
                "n_rows": n_rows,
                "duplicate_timestamp_rows": dup_rows,
                "timestamp_unique_ratio": float(n_unique / n_rows) if n_rows > 0 else 0.0,
                "estimated_fs_hz": fs,
                "median_dt_ms": med_dt,
                "p95_dt_ms": p95_dt,
                "irregular_dt_ratio": irr_ratio,
            }
        )

    return pd.DataFrame(rows)


def preprocess_raw_trip_csv(csv_path: Path, target_fs_hz: float) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Kolom 'timestamp' tidak ditemukan di {csv_path}")

    # Keep numeric signals only to keep interpolation deterministic and simple.
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "timestamp" not in num_cols:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "timestamp" not in num_cols:
        raise ValueError(f"Kolom 'timestamp' gagal dikonversi numerik di {csv_path}")

    signal = df[num_cols].copy()
    signal["timestamp"] = pd.to_numeric(signal["timestamp"], errors="coerce")
    signal = signal.dropna(subset=["timestamp"]).sort_values("timestamp")
    input_rows = int(len(signal))
    if input_rows < 2:
        return signal, {
            "input_rows": input_rows,
            "rows_after_dedup": input_rows,
            "rows_after_resample": input_rows,
            "target_fs_hz": target_fs_hz,
            "status": "too_few_rows",
        }

    grouped = signal.groupby("timestamp", as_index=False).mean(numeric_only=True)
    dedup_rows = int(len(grouped))

    step_ms = 1000.0 / float(target_fs_hz)
    t_start = float(grouped["timestamp"].iloc[0])
    t_end = float(grouped["timestamp"].iloc[-1])
    if t_end <= t_start:
        return grouped, {
            "input_rows": input_rows,
            "rows_after_dedup": dedup_rows,
            "rows_after_resample": dedup_rows,
            "target_fs_hz": target_fs_hz,
            "status": "invalid_time_range",
        }

    target_index = np.arange(t_start, t_end + step_ms, step_ms)
    reindexed = grouped.set_index("timestamp").reindex(target_index)
    reindexed.index.name = "timestamp"
    interpolated = reindexed.interpolate(method="index", limit_direction="both")
    out_df = interpolated.reset_index()
    out_df["timestamp"] = np.round(out_df["timestamp"]).astype(np.int64)

    return out_df, {
        "input_rows": input_rows,
        "rows_after_dedup": dedup_rows,
        "rows_after_resample": int(len(out_df)),
        "target_fs_hz": target_fs_hz,
        "status": "ok",
    }


def ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible event-labeling pipeline (copilot variant).")
    parser.add_argument("--repo-root", default=".", help="Path root repository.")
    parser.add_argument("--config", default=None, help="Path pipeline config JSON.")
    parser.add_argument("--taxonomy", default=None, help="Path taxonomy JSON.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve() if args.config else (script_dir / "config" / "pipeline_config.json")
    taxonomy_path = Path(args.taxonomy).resolve() if args.taxonomy else (script_dir / "config" / "taxonomy.json")

    config = load_json(config_path)
    taxonomy = load_json(taxonomy_path)

    out_dir = resolve_repo_path(repo_root, config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = resolve_repo_path(repo_root, config["input"]["candidates_events_csv"])
    gt_path = resolve_repo_path(repo_root, config["input"]["ground_truth_csv"])

    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file tidak ditemukan: {candidates_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file tidak ditemukan: {gt_path}")

    candidates_df = pd.read_csv(candidates_path)
    gt_df = pd.read_csv(gt_path)
    ensure_columns(candidates_df, REQUIRED_CANDIDATE_COLS, "candidates_events")
    ensure_columns(gt_df, REQUIRED_GT_COLS, "ground_truth_labels")

    gt_df["label"] = gt_df["label"].fillna("").astype(str).str.strip()
    gt_df["labeled_at"] = gt_df.get("labeled_at", "")
    gt_df["labeled_at_ts"] = pd.to_datetime(gt_df["labeled_at"], errors="coerce")
    gt_df = gt_df.sort_values("labeled_at_ts").drop_duplicates(subset=["trip_id", "event_id"], keep="last")

    normalized_labels, unknown_labels = normalize_label_series(
        gt_df["label"], taxonomy["classes"], taxonomy["aliases"]
    )
    gt_df["label_canonical"] = normalized_labels

    if config.get("strict_label_validation", True) and unknown_labels:
        raise ValueError(
            "Label tidak terdaftar di taxonomy: "
            + ", ".join(unknown_labels)
            + ". Tambahkan ke taxonomy.json alias/classes terlebih dahulu."
        )

    candidates_df = candidates_df.sort_values(["trip_id", "time_s", "event_id"]).reset_index(drop=True)
    candidates_df["stable_event_id"] = candidates_df.apply(
        lambda r: build_stable_event_id(r, config["stable_key"]), axis=1
    )

    duplicate_stable_id_count = int(candidates_df["stable_event_id"].duplicated(keep=False).sum())
    if duplicate_stable_id_count > 0:
        raise ValueError(
            f"Terdeteksi {duplicate_stable_id_count} baris dengan stable_event_id duplikat. "
            "Perketat konfigurasi stable_key (misalnya time_round_s/decimal) sebelum lanjut labeling."
        )

    gt_match = gt_df.merge(
        candidates_df[["trip_id", "event_id"]],
        on=["trip_id", "event_id"],
        how="left",
        indicator=True,
    )
    gt_unmatched = gt_match[gt_match["_merge"] == "left_only"].copy()

    merged = candidates_df.merge(
        gt_df[["trip_id", "event_id", "label_canonical", "labeled_at"]],
        on=["trip_id", "event_id"],
        how="left",
    )
    labeled_events = merged[merged["label_canonical"].notna() & (merged["label_canonical"] != "")].copy()
    unlabeled_events = merged[merged["label_canonical"].isna() | (merged["label_canonical"] == "")].copy()

    if labeled_events.empty:
        raise ValueError("Tidak ada event berlabel yang valid setelah normalisasi taxonomy.")

    split_df = deterministic_trip_split(labeled_events["trip_id"].tolist(), config["split"])
    labeled_events = labeled_events.merge(split_df[["trip_id", "split"]], on="trip_id", how="left")

    features = [c for c in config["dataset"]["feature_columns"] if c in labeled_events.columns]
    dataset_cols = ["stable_event_id", "trip_id", "event_id", "time_s", "label_canonical", "split"] + features
    dataset_df = labeled_events[dataset_cols].rename(columns={"label_canonical": "label"})

    events_stable_path = out_dir / "events_stable.csv"
    gt_stable_path = out_dir / "ground_truth_stable.csv"
    dataset_path = out_dir / "event_classification_dataset.csv"
    split_path = out_dir / "trip_split_assignment.csv"
    review_queue_path = out_dir / "review_queue.csv"
    raw_quality_path = out_dir / "raw_signal_quality_report.csv"
    preprocess_summary_path = out_dir / "preprocessed_trip_summary.csv"
    qa_report_path = out_dir / "label_quality_report.json"
    manifest_path = out_dir / "run_manifest.json"

    candidates_df.to_csv(events_stable_path, index=False)

    gt_stable_df = labeled_events[
        ["stable_event_id", "trip_id", "event_id", "label_canonical", "labeled_at"]
    ].rename(columns={"label_canonical": "label"})
    gt_stable_df = gt_stable_df.sort_values("labeled_at").drop_duplicates(
        subset=["stable_event_id"], keep="last"
    )
    gt_stable_df.to_csv(gt_stable_path, index=False)

    dataset_df.to_csv(dataset_path, index=False)
    split_df.to_csv(split_path, index=False)

    priority_order = {"high": 0, "medium": 1, "low": 2}
    if "priority" in unlabeled_events.columns:
        unlabeled_events["priority_rank"] = unlabeled_events["priority"].map(priority_order).fillna(9)
    else:
        unlabeled_events["priority_rank"] = 9
    if "score" not in unlabeled_events.columns:
        unlabeled_events["score"] = np.nan
    review_cols = [
        "stable_event_id",
        "trip_id",
        "event_id",
        "time_s",
        "lat",
        "lon",
        "priority",
        "score",
        "peak_vertical_g",
        "peak_gyro_mag",
        "vert_jrk",
    ]
    review_cols = [c for c in review_cols if c in unlabeled_events.columns]
    review_queue = unlabeled_events.sort_values(
        by=["priority_rank", "score", "time_s"], ascending=[True, False, True]
    ).head(int(config["dataset"].get("max_review_queue", 2000)))
    review_queue[review_cols].to_csv(review_queue_path, index=False)

    raw_quality_df = build_raw_signal_quality_report(repo_root, config)
    raw_quality_df.to_csv(raw_quality_path, index=False)

    preprocess_enabled = bool(config.get("preprocessing", {}).get("enabled", False))
    preprocess_rows: List[Dict] = []
    if preprocess_enabled:
        meta_pattern = resolve_repo_path(repo_root, config["input"]["meta_glob"])
        raw_dir = resolve_repo_path(repo_root, config["input"]["raw_csv_dir"])
        target_fs_hz = float(config["preprocessing"]["target_fs_hz"])
        out_subdir = str(config["preprocessing"]["output_subdir"])
        preprocess_dir = resolve_repo_path(repo_root, f"{config['output_dir']}\\{out_subdir}")
        preprocess_dir.mkdir(parents=True, exist_ok=True)

        for meta_path_str in sorted(glob.glob(str(meta_pattern))):
            meta_path = Path(meta_path_str)
            meta = load_json(meta_path)
            trip_id = str(meta.get("tripId", meta_path.stem))
            csv_name = f"{meta_path.stem}.csv"
            csv_path = raw_dir / csv_name

            if not csv_path.exists():
                preprocess_rows.append(
                    {
                        "trip_id": trip_id,
                        "raw_csv": csv_name,
                        "status": "missing_csv",
                        "input_rows": 0,
                        "rows_after_dedup": 0,
                        "rows_after_resample": 0,
                        "target_fs_hz": target_fs_hz,
                        "output_csv": "",
                    }
                )
                continue

            pre_df, stat = preprocess_raw_trip_csv(csv_path, target_fs_hz=target_fs_hz)
            out_name = f"{trip_id}.csv"
            out_csv_path = preprocess_dir / out_name
            pre_df.to_csv(out_csv_path, index=False)

            preprocess_rows.append(
                {
                    "trip_id": trip_id,
                    "raw_csv": csv_name,
                    "status": stat["status"],
                    "input_rows": stat["input_rows"],
                    "rows_after_dedup": stat["rows_after_dedup"],
                    "rows_after_resample": stat["rows_after_resample"],
                    "target_fs_hz": stat["target_fs_hz"],
                    "output_csv": str(out_csv_path),
                }
            )

    preprocess_summary_cols = [
        "trip_id",
        "raw_csv",
        "status",
        "input_rows",
        "rows_after_dedup",
        "rows_after_resample",
        "target_fs_hz",
        "output_csv",
    ]
    preprocess_summary_df = pd.DataFrame(preprocess_rows, columns=preprocess_summary_cols)
    preprocess_summary_df.to_csv(preprocess_summary_path, index=False)

    class_distribution = dataset_df["label"].value_counts().to_dict()
    split_distribution = dataset_df["split"].value_counts().to_dict()
    qa_report = {
        "pipeline": "labeling-copilot",
        "pipeline_version": config["pipeline_version"],
        "taxonomy_version": taxonomy["taxonomy_version"],
        "summary": {
            "n_candidates_total": int(len(candidates_df)),
            "n_labeled_events": int(len(dataset_df)),
            "n_unlabeled_events": int(len(unlabeled_events)),
            "n_gt_rows_input": int(len(gt_df)),
            "n_gt_unmatched_event_id_trip_id": int(len(gt_unmatched)),
            "n_duplicate_stable_event_id": duplicate_stable_id_count,
            "n_preprocessed_trips": int(len(preprocess_summary_df)) if preprocess_enabled else 0,
        },
        "distributions": {
            "class_distribution": class_distribution,
            "split_distribution": split_distribution,
        },
        "taxonomy": {
            "unknown_labels_detected": unknown_labels,
        },
    }
    with qa_report_path.open("w", encoding="utf-8") as f:
        json.dump(qa_report, f, indent=2, ensure_ascii=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "labeling-copilot",
        "pipeline_version": config["pipeline_version"],
        "taxonomy_version": taxonomy["taxonomy_version"],
        "input_files": {
            "candidates_events_csv": {
                "path": str(candidates_path),
                "sha256": hash_file(candidates_path),
                "rows": int(len(candidates_df)),
            },
            "ground_truth_csv": {
                "path": str(gt_path),
                "sha256": hash_file(gt_path),
                "rows": int(len(gt_df)),
            },
        },
        "config": {
            "config_path": str(config_path),
            "taxonomy_path": str(taxonomy_path),
            "config_sha1": hash_text(json.dumps(config, sort_keys=True)),
            "taxonomy_sha1": hash_text(json.dumps(taxonomy, sort_keys=True)),
        },
        "output_files": {
            "events_stable_csv": str(events_stable_path),
            "ground_truth_stable_csv": str(gt_stable_path),
            "dataset_csv": str(dataset_path),
            "split_csv": str(split_path),
            "review_queue_csv": str(review_queue_path),
            "raw_signal_quality_csv": str(raw_quality_path),
            "preprocessed_trip_summary_csv": str(preprocess_summary_path),
            "qa_report_json": str(qa_report_path),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("[OK] Labeling copilot pipeline selesai.")
    print(f"Output directory: {out_dir}")
    print(f"Labeled events: {len(dataset_df)}")
    print(f"Unlabeled review queue: {len(review_queue)}")
    print(f"Unknown labels: {len(unknown_labels)}")


if __name__ == "__main__":
    main()
