"""HeLiMOS sensor registry.

Single source of truth for the sensor identity that X-MOS uses as privileged information:
the HeLiMOS sequence folder name, the integer id attached to every point by the dataloader,
and the short key (V, O, A, L in the paper) of the expert teacher of that sensor.
"""

from typing import Dict, NamedTuple, Tuple


class Sensor(NamedTuple):
    name: str  # HeLiMOS folder name, e.g. "Ouster"
    key: str  # teacher key used in the paper and in the config, e.g. "O"
    index: int  # integer id attached to each point of a scan of this sensor


SENSORS: Tuple[Sensor, ...] = (
    Sensor("Ouster", "O", 0),  # OS2-128, omni-directional, very dense
    Sensor("Velodyne", "V", 1),  # VLP-16, omni-directional, sparse
    Sensor("Aeva", "A", 2),  # Aeries II, solid-state, dense
    Sensor("Avia", "L", 3),  # Livox Avia, solid-state, dense
)

SENSOR_NAME_TO_INDEX: Dict[str, int] = {sensor.name: sensor.index for sensor in SENSORS}
TEACHER_KEY_TO_INDEX: Dict[str, int] = {sensor.key: sensor.index for sensor in SENSORS}

# Sequences that do not belong to a known sensor (e.g. KITTI "08") fall back to this id
DEFAULT_SENSOR_INDEX = 0

# Order in which the teachers are loaded (and therefore summed up in the distillation losses)
TEACHER_KEYS: Tuple[str, ...] = ("O", "A", "V", "L")
