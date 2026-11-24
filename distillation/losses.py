"""Point-wise logit distillation losses (Sec. III-C).

All losses take `teacher_outputs`, a dict that maps the teacher key (O, A, V, L) to a tuple of
the teacher's (static, moving) logits [N, 2] and its bottleneck features, and the student's
(static, moving) logits [N, 2].
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from datasets.sensors import TEACHER_KEY_TO_INDEX

KD_MODES = ("sensor_aware", "attention", "average")

TeacherOutputs = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


def soft_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    differentiable_teacher: bool = False,
):
    """Temperature-scaled KL divergence between the student and the teacher distribution.

    Set `differentiable_teacher` if gradients flow through the teacher logits. The gradient of
    the KL divergence w.r.t. a target probability p is log(p) + ..., which is -inf as soon as p
    underflows to zero (logits that are ~100 * temperature apart) and turns every weight into NaN.
    The target is therefore kept in log space in that case, which is the same loss but has a
    finite gradient.
    """
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    if differentiable_teacher:
        log_soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
        kd_loss = F.kl_div(soft_student, log_soft_teacher, reduction="batchmean", log_target=True)
    else:
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
    return kd_loss * (temperature**2)


def sensor_aware_kd_loss(
    teacher_outputs: TeacherOutputs,
    student_logits: torch.Tensor,
    sensor_ids: torch.Tensor,
    temperature: float,
):
    """Teacher activator T_A (Eq. 3): every point is only supervised by the teacher of its sensor.

    Args:
        sensor_ids: [N] sensor index of each point, the privileged information.
    """
    total_kd_loss = torch.tensor(0.0, device=student_logits.device)
    for teacher_key, (teacher_logits, _) in teacher_outputs.items():
        specialist_mask = sensor_ids == TEACHER_KEY_TO_INDEX[teacher_key]
        t, s = teacher_logits[specialist_mask], student_logits[specialist_mask]
        if len(s) > 0:
            total_kd_loss += soft_kd_loss(s, t, temperature)

    total_kd_loss /= len(teacher_outputs)
    return total_kd_loss


def average_based_kd_loss(
    teacher_outputs: TeacherOutputs, student_logits: torch.Tensor, temperature: float
):
    """Ablation (Case 012): distill from the average of the logits of all teachers."""
    averaged_logits = None
    for teacher_logits, _ in teacher_outputs.values():
        if averaged_logits is None:
            averaged_logits = teacher_logits
        else:
            averaged_logits += teacher_logits
    averaged_logits /= len(teacher_outputs)
    return soft_kd_loss(student_logits, averaged_logits, temperature)


def attention_based_kd_loss(
    teacher_outputs: TeacherOutputs,
    student_logits: torch.Tensor,
    student_bottleneck: torch.Tensor,
    q_projection: torch.nn.Module,
    k_projection: torch.nn.Module,
    temperature: float,
):
    """Ablation (Case 011): blend the teacher logits with learned attention weights.

    The global average of the student bottleneck is the query, the global averages of the teacher
    bottlenecks are the keys and the teacher logits are the values.

    Returns:
        The loss and the attention weight of each teacher (for logging).
    """
    q_feature = torch.mean(student_bottleneck, dim=0, keepdim=True)  # (1, C)

    teacher_keys, teacher_values = [], []
    for teacher_logits, teacher_bottleneck in teacher_outputs.values():
        teacher_keys.append(torch.mean(teacher_bottleneck, dim=0, keepdim=True))  # (1, C)
        teacher_values.append(teacher_logits)
    teacher_keys = torch.stack(teacher_keys).squeeze(1)  # (num_teachers, C)

    # Scaled dot-product attention between the student and the teachers
    q_proj = q_projection(q_feature)  # (1, attention_dim)
    k_proj = k_projection(teacher_keys)  # (num_teachers, attention_dim)
    attention_scores = torch.matmul(q_proj, k_proj.T) / (q_proj.shape[-1] ** 0.5)
    attention_weights = F.softmax(attention_scores, dim=-1)  # (1, num_teachers)

    # Weighted sum of the teacher logits: (num_teachers, N, 2) -> (N, 2)
    teacher_values = torch.stack(teacher_values)
    blended_logits = torch.sum(teacher_values * attention_weights.T.unsqueeze(-1), dim=0)

    weights = dict(zip(teacher_outputs.keys(), attention_weights.flatten().detach().cpu()))
    # The attention weights are learned through the blended teacher logits
    kd_loss = soft_kd_loss(student_logits, blended_logits, temperature, differentiable_teacher=True)
    return kd_loss, weights
