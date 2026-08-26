#!/usr/bin/env python3
"""Example of using the Rerun SDK to log the Objectron dataset."""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import math
import os
import sys
import time

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

import websockets

import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb

from .download_dataset import (
    ANNOTATIONS_FILENAME,
    AVAILABLE_RECORDINGS,
    GEOMETRY_FILENAME,
    LOCAL_DATASET_DIR,
    ensure_recording_available,
)
from .viewer_control import DEFAULT_VIEWER_PORT, seek_viewer

from .proto.objectron.proto import ARCamera, ARFrame, ARPointCloud, Object, ObjectType, Sequence

# Store data for each logged frame — used by websocket handler
_frame_data: list[tuple[int, float, np.ndarray, Path]] = []

def quat_forward_up(q_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the forward (0, 0, -1) and up (0, 1, 0) vectors rotated by quaternion `q_xyzw` (xyzw order)."""
    x, y, z, w = q_xyzw
    # Rotate (0, 0, -1) by unit quaternion q:
    #   v' = v + 2*w*cross(q_xyz, v) + 2*cross(q_xyz, cross(q_xyz, v))
    # Simplified for v = (0, 0, -1) and v = (0, 1, 0):
    forward = np.array([
        -2.0 * w * y - 2.0 * x * z,
         2.0 * w * x - 2.0 * y * z,
        -1.0 + 2.0 * (x * x + y * y),
    ])
    up = np.array([
         2.0 * x * y - 2.0 * w * z,
         1.0 - 2.0 * (x * x + z * z),
         2.0 * w * x + 2.0 * y * z,
    ])
    return forward, up


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@dataclass
class SampleARFrame:
    """An `ARFrame` sample and the relevant associated metadata."""

    index: int
    timestamp: float
    dirpath: Path
    frame: ARFrame
    image_path: Path


def read_ar_frames(
    dirpath: Path,
    num_frames: int,
    run_forever: bool,
    per_frame_sleep: float,
) -> Iterator[SampleARFrame]:
    """
    Loads up to `num_frames` consecutive ARFrames from the given path on disk.

    `dirpath` should be of the form `dataset/bike/batch-8/16/`.
    """

    path = dirpath / GEOMETRY_FILENAME
    print(f"loading ARFrames from {path}")

    time_offset = 0
    frame_offset = 0
    frame = ARFrame()

    while True:
        frame_idx = 0
        data = Path(path).read_bytes()
        while len(data) > 0 and frame_idx < num_frames:
            next_len = int.from_bytes(data[:4], byteorder="little", signed=False)
            data = data[4:]

            frame = ARFrame().parse(data[:next_len])
            img_path = Path(os.path.join(dirpath, f"video/{frame_idx}.jpg"))
            yield SampleARFrame(
                index=frame_idx + frame_offset,
                timestamp=frame.timestamp + time_offset,
                dirpath=dirpath,
                frame=frame,
                image_path=img_path,
            )

            data = data[next_len:]
            frame_idx += 1

            if run_forever and per_frame_sleep > 0.0:
                time.sleep(per_frame_sleep)

        if run_forever:
            time_offset += frame.timestamp
            frame_offset += frame_idx
        else:
            break


def read_annotations(dirpath: Path) -> Sequence:
    """
    Loads the annotations from the given path on disk.

    `dirpath` should be of the form `dataset/bike/batch-8/16/`.
    """

    path = dirpath / ANNOTATIONS_FILENAME
    print(f"loading annotations from {path}")
    data = Path(path).read_bytes()

    seq = Sequence().parse(data)

    return seq


def log_ar_frames(samples: Iterable[SampleARFrame], seq: Sequence) -> None:
    """Logs a stream of `ARFrame` samples and their annotations with the Rerun SDK."""

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    log_annotated_bboxes(seq.objects)

    for sample in samples:
        rr.set_time("frame", sequence=sample.index)
        rr.set_time("time", duration=sample.timestamp)

        # Store frame transform for websocket lookup
        world_from_cam = np.asarray(sample.frame.camera.transform).reshape((4, 4))
        _frame_data.append((sample.index, sample.timestamp, world_from_cam, sample.image_path))

        rr.log("world/camera", rr.EncodedImage(path=sample.image_path))
        log_camera(sample.frame.camera)
        log_point_cloud(sample.frame.raw_feature_points)


def log_camera(cam: ARCamera) -> None:
    """Logs a camera from an `ARFrame` using the Rerun SDK."""

    X = np.asarray([1.0, 0.0, 0.0])
    Z = np.asarray([0.0, 0.0, 1.0])

    world_from_cam = np.asarray(cam.transform).reshape((4, 4))
    translation = world_from_cam[0:3, 3]
    intrinsics = np.asarray(cam.intrinsics).reshape((3, 3))
    rot = R.from_matrix(world_from_cam[0:3, 0:3])
    (w, h) = (cam.image_resolution_width, cam.image_resolution_height)

    # Because the dataset was collected in portrait:
    swizzle_x_y = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    intrinsics = swizzle_x_y @ intrinsics @ swizzle_x_y
    rot = rot * R.from_rotvec((math.tau / 4.0) * Z)
    (w, h) = (h, w)

    rot = rot * R.from_rotvec((math.tau / 2.0) * X)  # TODO(emilk): figure out why this is needed

    rr.log(
        "world/camera",
        rr.Transform3D(translation=translation, rotation=rr.Quaternion(xyzw=rot.as_quat())),
    )
    rr.log(
        "world/camera",
        rr.Pinhole(
            resolution=[w, h],
            image_from_camera=intrinsics,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )


def log_point_cloud(point_cloud: ARPointCloud) -> None:
    """Logs a point cloud from an `ARFrame` using the Rerun SDK."""

    positions = np.array([[p.x, p.y, p.z] for p in point_cloud.point]).astype(np.float32)
    rr.log("world/points", rr.Points3D(positions, colors=[255, 255, 255, 255]))


def log_annotated_bboxes(bboxes: Iterable[Object]) -> None:
    """Logs all the bounding boxes annotated in an `ARFrame` sequence using the Rerun SDK."""

    for bbox in bboxes:
        if bbox.type != ObjectType.BOUNDING_BOX:
            print(f"err: object type not supported: {bbox.type}")
            continue

        rot = R.from_matrix(np.asarray(bbox.rotation).reshape((3, 3)))
        rr.log(
            f"world/annotations/box-{bbox.id}",
            rr.Boxes3D(
                half_sizes=0.5 * np.array(bbox.scale),
                centers=bbox.translation,
                rotations=rr.Quaternion(xyzw=rot.as_quat()),
                colors=[160, 230, 130, 255],
                labels=bbox.category,
            ),
            static=True,
        )


async def ws_handler(websocket, viewer_port: int) -> None:
    """Handle incoming websocket messages: find closest logged frame and seek the viewer to it."""
    async for message in websocket:
        data = json.loads(message)
        eye_pos = np.array(data["eye_position"])
        eye_rot = np.array(data["eye_rotation"])

        # Convert eye quaternion (xyzw) to forward vector
        eye_forward, eye_up = quat_forward_up(eye_rot)

        def heuristic(cam: np.ndarray, eye_pos: np.ndarray, eye_forward: np.ndarray, eye_up: np.ndarray) -> np.floating:
            cam_pos = cam[0:3, 3]
            cam_forward = -cam[0:3, 2]
            cam_up = cam[0:3, 1]

            pos_dist = np.linalg.norm(cam_pos - eye_pos)
            angle_dist = np.linalg.norm(cam_forward - eye_forward)
            up_dist = np.linalg.norm(cam_up - eye_up)

            return pos_dist + 1.0 * angle_dist + 1.0 * up_dist

        # Find closest logged frame by combined position + orientation distance
        best_index, _best_ts, _world_from_cam, _image_path = min(
            _frame_data, key=lambda x: heuristic(x[2], eye_pos, eye_forward, eye_up)
        )

        # Move the viewer's time cursor to the closest frame (sequence timeline).
        try:
            await seek_viewer(timeline="frame", time=best_index, viewer_port=viewer_port)
        except Exception as err:  # noqa: BLE001 - keep the websocket loop alive if the viewer is unreachable
            logging.warning("Failed to move the viewer time cursor: %s", err)


async def run_ws_server(host: str, port: int, viewer_port: int) -> None:
    """Start a websocket server that listens for camera pose updates."""
    handler = functools.partial(ws_handler, viewer_port=viewer_port)
    async with websockets.serve(handler, host, port):
        print(f"WebSocket server at ws://{host}:{port}")
        await asyncio.Future()  # run forever


def main() -> None:
    # Ensure the logging in download_dataset.py gets written to stderr:
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel("INFO")

    parser = argparse.ArgumentParser(description="Logs Objectron data using the Rerun SDK.")
    parser.add_argument(
        "--frames",
        type=int,
        default=sys.maxsize,
        help="If specified, limits the number of frames logged",
    )
    parser.add_argument("--run-forever", action="store_true", help="Run forever, continually logging data.")
    parser.add_argument(
        "--per-frame-sleep",
        type=float,
        default=0.1,
        help="Sleep this much for each frame read, if --run-forever",
    )
    parser.add_argument(
        "--recording",
        type=str,
        choices=AVAILABLE_RECORDINGS,
        default=AVAILABLE_RECORDINGS[1],
        help="The objectron recording to log to Rerun.",
    )
    parser.add_argument(
        "--force-reprocess-video",
        action="store_true",
        help="Reprocess video frames even if they already exist",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=LOCAL_DATASET_DIR,
        help="Directory to save example videos to.",
    )
    parser.add_argument(
        "--viewer-port",
        type=int,
        default=DEFAULT_VIEWER_PORT,
        help="gRPC port of the spawned viewer's ViewerControlService (used to seek the time cursor).",
    )

    rr.script_add_args(parser)
    args = parser.parse_args()

    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(origin="/world", name="World"),
        rrb.Spatial2DView(origin="/world/camera", name="Camera", contents=["/world/**"]),
    )
    rr.script_setup(
        args,
        "rerun_example_objectron_interact",
        default_blueprint=blueprint,
    )

    dir = ensure_recording_available(args.recording, args.dataset_dir, args.force_reprocess_video)

    samples = read_ar_frames(dir, args.frames, args.run_forever, args.per_frame_sleep)
    seq = read_annotations(dir)
    log_ar_frames(samples, seq)
    asyncio.run(run_ws_server("0.0.0.0", 8080, args.viewer_port))

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
