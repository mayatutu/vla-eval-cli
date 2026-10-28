"""Evaluation-only copy of the deployed Client exporter, with exact RGB statistics.

Source SHA256: 84c37e813c2286e301ce5a2767999cfe3f1548453cf499c071fd50a87ac59546.
The isolated copy keeps the normal Client unchanged; only image statistics differ.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {**{name: function(values, axis=0).tolist() for name, function in
               [("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)]},
            "count": [len(values)]}


def _clear_source_images(path, cameras, rows):
    """Only remove this recording's frames, after the export is committed."""
    for camera in cameras:
        for row in rows:
            (path / camera / f"{row['frame']:06d}.png").unlink(missing_ok=True)


def _image_statistics(path, camera, rows):
    """Return padded dimensions and full-pixel RGB statistics without float images."""
    histogram = np.zeros((3, 256), dtype=np.uint64)
    first_size = None
    for row in rows:
        with Image.open(path / camera / f"{row['frame']:06d}.png") as image:
            with image.convert("RGB") as rgb:
                if first_size is None:
                    first_size = rgb.size
                if rgb.size != first_size:
                    raise ValueError("Camera image shape changed within the trial")
                histogram += np.asarray(rgb.histogram(), dtype=np.uint64).reshape(3, 256)
    source_width, source_height = first_size
    height = source_height + source_height % 2
    width = source_width + source_width % 2
    count = len(rows)
    # Match the encoder's black padding and count every selected output frame,
    # including repeated references to an actual source frame.
    histogram[:, 0] += count * (height * width - source_height * source_width)
    pixel_count = count * height * width
    levels = np.arange(256, dtype=np.float64)
    mean_pixels = (histogram * levels).sum(axis=1) / pixel_count
    # A centered second moment stays accurate for nearly constant images.
    variance = (histogram * (levels[None, :] - mean_pixels[:, None]) ** 2).sum(axis=1) / pixel_count
    minimum = np.asarray([np.flatnonzero(channel)[0] for channel in histogram]) / 255
    maximum = np.asarray([np.flatnonzero(channel)[-1] for channel in histogram]) / 255
    stats = {name: value.reshape(3, 1, 1).tolist() for name, value in
             [("min", minimum), ("max", maximum), ("mean", mean_pixels / 255),
              ("std", np.sqrt(variance) / 255)]}
    stats["count"] = [count]
    return height, width, stats


def export_trial(path: Path, cameras: list[str], *, fps: int, robot_type: str,
                 task: str, metadata: dict) -> Path:
    """Write atomically; retries reuse only a matching, complete export.

    Dataset timestamps index control steps at nominal FPS. The original irregular
    capture clock is preserved separately, without resampling observations/actions.
    Source PNGs are removed after export, without waiting for upload. The final
    observation with no action retains only its numeric source record.
    """
    path = Path(path)
    if fps <= 0 or not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("Positive FPS and unique cameras are required")
    if any(camera not in {"head_image", "left_image", "right_image"} for camera in cameras):
        raise ValueError("Unknown camera")
    status = json.loads((path / "recording-status.json").read_text())
    if not status["complete"]:
        raise ValueError("Cannot export an incomplete recording")
    raw = (path / "frames.jsonl").read_bytes()
    all_rows = [json.loads(line) for line in raw.splitlines()]
    if status["frames"] != len(all_rows):
        raise ValueError("Recording frame count differs from its status")
    rows = [row for row in all_rows if row["action"] is not None]
    if not rows or rows != all_rows[:len(rows)] or len(all_rows) - len(rows) > 1:
        raise ValueError("Expected action frames followed by at most one final observation")
    if [row["frame"] for row in all_rows] != list(range(len(all_rows))):
        raise ValueError("Recording frame indices are not contiguous")
    for row in rows:
        for key in ("joints", "gripper", "cartesian_position", "action"):
            value = np.asarray(row.get(key), dtype=np.float32)
            if value.ndim != 1 or not value.size or not np.isfinite(value).all():
                raise ValueError(f"Missing or invalid {key}; older recordings may lack full state")
    vectors = {
        "observation.state": [row["joints"] + row["gripper"] for row in rows],
        "observation.cartesian_position": [row["cartesian_position"] for row in rows],
        "action": [row["action"] for row in rows],
    }
    vectors = {key: np.asarray(values, dtype=np.float32) for key, values in vectors.items()}
    capture = np.asarray([row["seconds"] for row in rows], dtype=np.float64)
    if not np.isfinite(capture).all() or np.any(capture < 0) or np.any(np.diff(capture) <= 0):
        raise ValueError("Capture times must be finite, nonnegative and strictly increasing")
    policy_path = path / "policy.json"
    context = {**metadata, "robot_type": robot_type, "task": task, "fps": fps,
               "cameras": cameras, "policy": json.loads(policy_path.read_text()) if policy_path.exists() else None,
               "timestamp_semantics": "frame_index / nominal control FPS; no resampling",
               "capture_timestamp_semantics": "original seconds since recorder start",
               "action_semantics": "selected command; not measured execution"}
    signature = hashlib.sha256(raw + json.dumps(context, sort_keys=True).encode()).hexdigest()
    destination = path / "lerobot"
    if destination.exists():
        marker = json.loads((destination / "meta/colosseum.json").read_text())
        if marker.get("source_sha256") != signature:
            raise ValueError("Existing LeRobot export differs from this recording/context")
        for relative, digest in marker["files"].items():
            if hashlib.sha256((destination / relative).read_bytes()).hexdigest() != digest:
                raise ValueError("Existing LeRobot export is incomplete or modified")
        _clear_source_images(path, cameras, all_rows)
        return destination
    staging = Path(tempfile.mkdtemp(prefix=".lerobot-", dir=path))
    try:
        count = len(rows)
        features = {}
        columns = {}
        stats = {}
        joint_names = [f"joint_{i}" for i in range(len(rows[0]["joints"]))]
        gripper_names = [f"gripper_{i}" for i in range(len(rows[0]["gripper"]))]
        for key, array in vectors.items():
            names = joint_names + gripper_names if key == "observation.state" else [
                f"{key.rsplit('.', 1)[-1]}_{i}" for i in range(array.shape[1])]
            features[key] = {"dtype": "float32", "shape": [array.shape[1]], "names": names}
            columns[key] = pa.array(array.tolist(), type=pa.list_(pa.float32(), array.shape[1]))
            stats[key] = _stats(array)
        scalars = {"timestamp": np.arange(count, dtype=np.float32) / fps,
                   "frame_index": np.arange(count, dtype=np.int64),
                   "episode_index": np.zeros(count, dtype=np.int64),
                   "index": np.arange(count, dtype=np.int64),
                   "task_index": np.zeros(count, dtype=np.int64),
                   "observation.capture_timestamp": capture}
        for key, values in scalars.items():
            features[key] = {"dtype": str(values.dtype), "shape": [1], "names": None}
            columns[key] = pa.array(values)
            stats[key] = _stats(values)
        data = staging / "data/chunk-000/file-000.parquet"
        data.parent.mkdir(parents=True)
        pq.write_table(pa.table(columns), data)
        # LeRobot tasks.parquet uses task text as the pandas index.
        tasks = pa.table({"task_index": pa.array([0], type=pa.int64()),
                          "__index_level_0__": [task]})
        pandas_metadata = {"index_columns": ["__index_level_0__"], "column_indexes": [],
            "columns": [
                {"name": "task_index", "field_name": "task_index", "pandas_type": "int64",
                 "numpy_type": "int64", "metadata": None},
                {"name": None, "field_name": "__index_level_0__", "pandas_type": "unicode",
                 "numpy_type": "object", "metadata": None}],
            "creator": {"library": "pyarrow", "version": pa.__version__}, "pandas_version": "2.2.3"}
        tasks = tasks.replace_schema_metadata({b"pandas": json.dumps(pandas_metadata).encode()})
        (staging / "meta").mkdir()
        pq.write_table(tasks, staging / "meta/tasks.parquet")
        episode = {"episode_index": 0, "tasks": [task], "length": count,
                   "dataset_from_index": 0, "dataset_to_index": count,
                   "data/chunk_index": 0, "data/file_index": 0,
                   "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0}
        for camera in cameras:
            key = f"observation.images.{camera}"
            height, width, stats[key] = _image_statistics(path, camera, rows)
            video = staging / f"videos/{key}/chunk-000/file-000.mp4"
            video.parent.mkdir(parents=True)
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(fps), "-start_number", "0", "-i", str(path / camera / "%06d.png"),
                "-frames:v", str(count), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
                check=True)
            features[key] = {"dtype": "video", "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.height": height, "video.width": width, "video.codec": "h264",
                         "video.pix_fmt": "yuv420p", "video.fps": fps, "video.channels": 3,
                         "video.is_depth_map": False, "has_audio": False}}
            episode.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                            f"videos/{key}/from_timestamp": 0.0,
                            f"videos/{key}/to_timestamp": count / fps})
        for key, values in stats.items():
            episode.update({f"stats/{key}/{name}": value for name, value in values.items()})
        ep_path = staging / "meta/episodes/chunk-000/file-000.parquet"
        ep_path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([episode]), ep_path)
        _json(staging / "meta/stats.json", stats)
        _json(staging / "meta/info.json", {
            "codebase_version": "v3.0", "robot_type": robot_type, "total_episodes": 1,
            "total_frames": count, "total_tasks": 1, "fps": fps,
            "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 500,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features})
        files = {str(file.relative_to(staging)): hashlib.sha256(file.read_bytes()).hexdigest()
                 for file in staging.rglob("*") if file.is_file()}
        _json(staging / "meta/colosseum.json", {**context, "source_sha256": signature, "files": files})
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    _clear_source_images(path, cameras, all_rows)
    return destination
