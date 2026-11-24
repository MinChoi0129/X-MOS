"""Channel-down operator of the multi-level adaptive teacher generator (Sec. III-B.2, Eq. 1).

P_low = Psi(P_high; K, M, p): the elevation angles of a high-resolution point cloud are clustered
into K beams with a 1D K-Means, M of them are selected at uniform intervals and the points of the
selected beams are kept with probability p.
"""

from typing import List, Optional, Tuple

import torch


def kmeans_1d(values: torch.Tensor, k: int, iters: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """1D K-Means on `values` [M] with uniformly spaced initial centers. Returns labels, centers."""
    vmin, vmax = values.min(), values.max()
    if (vmax - vmin).abs() < 1e-6:
        centers = vmin + torch.linspace(0, 1, k, device=values.device, dtype=values.dtype) * 0
    else:
        centers = torch.linspace(vmin, vmax, k, device=values.device, dtype=values.dtype)
    labels = torch.zeros_like(values, dtype=torch.long)
    for _ in range(max(1, iters)):
        # Assignment
        d = (values[:, None] - centers[None, :]).abs()
        labels = torch.argmin(d, dim=1)
        # Update
        new_centers = centers.clone()
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers[j] = values[mask].mean()
        if torch.allclose(new_centers, centers, atol=1e-6, rtol=0):
            centers = new_centers
            break
        centers = new_centers
    return labels, centers


def beam_down(
    batch: torch.Tensor,
    src_clusters: int,
    target_beams: int,
    kmeans_iters: int = 10,
    within_band_keep: float = 0.5,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Generates a pseudo low-beam point cloud in the style of LiDAR Distillation.

    Args:
        batch: [N, 6] = [batch_idx, x, y, z, t, label]
        src_clusters: number of clusters K approximating the beams of the source
        target_beams: number of beams M that are kept
        kmeans_iters: number of K-Means iterations
        within_band_keep: keep probability p of the points inside the selected beams

    Returns:
        The reduced batch and the indices of the kept points w.r.t. `batch`. If no point survives,
        the batch is returned unchanged together with `None`.
    """
    if batch.numel() == 0:
        return batch, None

    device = batch.device
    with torch.no_grad():
        batch_idx_vec = batch[:, 0].to(torch.long)
        kept_indices = []

        for b in torch.unique(batch_idx_vec).tolist():
            mask_b = batch_idx_vec == b
            if not mask_b.any():
                continue
            xyz = batch[mask_b, 1:4]
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            r_xy = torch.sqrt(torch.clamp(x * x + y * y, min=1e-12))
            theta = torch.atan2(z, r_xy)  # elevation angle, [Mb]

            K = max(1, int(src_clusters))
            labels, centers = kmeans_1d(theta, K, kmeans_iters)

            # Select target_beams clusters at uniform intervals of the sorted centers
            _, order = torch.sort(centers)
            if target_beams >= K:
                keep_clusters = torch.arange(K, device=device)
            else:
                idx_lin = torch.linspace(0, K - 1, steps=target_beams, device=device)
                keep_clusters = idx_lin.round().to(torch.long)
            keep_labels = order[keep_clusters]

            # Keep the points of the selected beams with probability within_band_keep
            mask_kept = (labels.unsqueeze(1) == keep_labels.unsqueeze(0)).any(dim=1)
            if within_band_keep < 1.0:
                rand_keep = torch.rand(mask_kept.shape[0], device=device) < float(within_band_keep)
                mask_kept = mask_kept & rand_keep

            kept_local = torch.where(mask_b)[0][mask_kept]
            if kept_local.numel() > 0:
                kept_indices.append(kept_local)

        if len(kept_indices) == 0:
            return batch, None
        kept_indices = torch.cat(kept_indices, dim=0)
        return batch[kept_indices], kept_indices


def make_pseudo_pair(
    batch: torch.Tensor,
    chain: List[int],
    stage_index: int,
    kmeans_iters: int = 10,
    within_band_keep: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Builds the (teacher, student) inputs of one stage of the distillation chain.

    Teacher and student are roles within one stage, not fixed models: the student of a stage is
    the teacher of the next one. For chain = [128, 64, 32, 16]:

        stage 0: teacher sees the original 128 beams, student sees pseudo 64 beams
        stage 1: teacher sees pseudo 64 beams,        student sees pseudo 32 beams
        stage 2: teacher sees pseudo 32 beams,        student sees pseudo 16 beams

    `batch` is assumed to have the resolution chain[0]. For the later stages it is first reduced
    to the resolution of the teacher, so that every teacher sees the resolution it was trained on.

    Returns:
        before: the batch at resolution chain[stage] (teacher input)
        after: the batch at resolution chain[stage + 1] (student input), a subset of `before`
        kept: indices of the points of `after` w.r.t. `before`
    """
    stage = max(0, min(stage_index, len(chain) - 2))
    desired_before = int(chain[stage])
    desired_after = int(chain[stage + 1])

    # Cascade down from the top of the chain until the resolution of the teacher is reached
    before = batch
    if desired_before < int(chain[0]):
        for src, tgt in zip(chain, chain[1:]):
            src, tgt = int(src), int(tgt)
            before, _ = beam_down(before, src, tgt, kmeans_iters, within_band_keep)
            if tgt == desired_before:
                break

    after, kept = beam_down(before, desired_before, desired_after, kmeans_iters, within_band_keep)
    return before, after, kept
