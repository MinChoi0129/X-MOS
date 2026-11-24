"""Building blocks of the X-MOS framework.

- beam_down: channel-down operator (Eq. 1) used by the multi-level adaptive teacher generator
- teachers: loading of the frozen sensor-specific expert teachers
- losses: point-wise logit distillation losses (sensor-aware, plus the attention/average ablations)
"""

from .beam_down import beam_down, make_pseudo_pair
from .losses import (
    KD_MODES,
    attention_based_kd_loss,
    average_based_kd_loss,
    sensor_aware_kd_loss,
)
from .teachers import load_teacher, load_teachers
