#!/usr/bin/env python3
"""Writes a tiny synthetic dataset in the HeLiMOS layout, to smoke test the code without the data.

<root>/{Ouster,Velodyne,Aeva,Avia}/{velodyne/*.bin,labels/*.label,calib.txt}
<root>/{train,val,test}.txt

The scene is a ground plane inside a room with a few static and moving boxes. Each sensor
ray-casts it with its own beam pattern from a vehicle that drives forward.
"""
import sys
from pathlib import Path

import numpy as np

# name: (number of beams, elevation range [deg], number of columns, azimuth range [deg])
SENSORS = {
    "Ouster": (128, (-22.5, 22.5), 360, (-180, 180)),
    "Velodyne": (16, (-15.0, 15.0), 360, (-180, 180)),
    "Aeva": (64, (-15.0, 15.0), 160, (-55, 55)),
    "Avia": (48, (-25.0, 25.0), 120, (-35, 35)),
}
N_SCANS = 44
SPLITS = {"train": range(0, 24), "val": range(24, 34), "test": range(34, 44)}
STATIC, MOVING = 9, 251  # raw labels, > 250 is moving
SENSOR_HEIGHT = 1.8
ROOM = 30.0


def ray_box(origin, dirs, box_min, box_max):
    """Distance along each ray to an axis-aligned box, inf if it is not hit."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (box_min - origin) / dirs
        t2 = (box_max - origin) / dirs
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (t_near <= t_far) & (t_far > 0) & (t_near > 0)
    return np.where(hit, t_near, np.inf)


def scan(sensor, frame, rng):
    beams, (el_min, el_max), columns, (az_min, az_max) = SENSORS[sensor]
    elevation = np.deg2rad(np.linspace(el_min, el_max, beams))
    azimuth = np.deg2rad(np.linspace(az_min, az_max, columns, endpoint=False))
    el, az = np.meshgrid(elevation, azimuth, indexing="ij")
    dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1).reshape(-1, 3)

    # The vehicle drives along x, all objects are given in the world frame
    origin = np.array([0.25 * frame - 5.0, 0.0, 0.0])
    static_boxes = [((8, 4, -1.8), (10, 6, 0.5)), ((-12, -9, -1.8), (-9, -6, 1.0)), ((15, -7, -1.8), (16, -3, 2.0))]
    moving_boxes = [  # pedestrians and cars that move with their own velocity
        ((6 + 0.15 * frame, -3, -1.8), (6.8 + 0.15 * frame, -2.2, 0.0)),
        ((12 - 0.5 * frame, 2, -1.8), (16 - 0.5 * frame, 3.8, -0.2)),
        ((2, 8 - 0.3 * frame, -1.8), (3.8, 12 - 0.3 * frame, -0.2)),
    ]

    with np.errstate(divide="ignore"):
        t_ground = np.where(dirs[:, 2] < 0, -SENSOR_HEIGHT / dirs[:, 2], np.inf)
    # Inside the room the wall is hit when leaving the box
    with np.errstate(divide="ignore", invalid="ignore"):
        t_exit = np.nanmin(
            np.maximum((np.array([-ROOM, -ROOM, -1e9]) - origin) / dirs, (np.array([ROOM, ROOM, 1e9]) - origin) / dirs), axis=1
        )
    ranges = np.minimum(t_ground, t_exit)
    labels = np.full(len(dirs), STATIC, dtype=np.int32)
    for box_min, box_max in static_boxes:
        ranges = np.minimum(ranges, ray_box(origin, dirs, np.array(box_min, float), np.array(box_max, float)))
    for box_min, box_max in moving_boxes:
        t = ray_box(origin, dirs, np.array(box_min, float), np.array(box_max, float))
        labels[t < ranges] = MOVING
        ranges = np.minimum(ranges, t)

    valid = np.isfinite(ranges) & (ranges < 80.0)
    points = dirs[valid] * ranges[valid, None] + rng.normal(0, 0.01, (valid.sum(), 3))
    return points.astype(np.float32), labels[valid]


def main(root):
    root = Path(root)
    rng = np.random.default_rng(0)
    for sensor in SENSORS:
        (root / sensor / "velodyne").mkdir(parents=True, exist_ok=True)
        (root / sensor / "labels").mkdir(parents=True, exist_ok=True)
        (root / sensor / "calib.txt").write_text("Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        for frame in range(N_SCANS):
            points, labels = scan(sensor, frame, rng)
            np.hstack([points, np.ones((len(points), 1), np.float32)]).tofile(root / sensor / "velodyne" / f"{frame:06d}.bin")
            labels.tofile(root / sensor / "labels" / f"{frame:06d}.label")
        print(sensor, "points per scan:", len(points), "moving:", int((labels == MOVING).sum()))
    for split, indices in SPLITS.items():
        (root / f"{split}.txt").write_text("\n".join(str(i) for i in indices) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
