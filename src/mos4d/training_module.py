# MIT License
#
# Copyright (c) 2023 Benedikt Mersch, Tiziano Guadagnino, Ignacio Vizzo, Cyrill Stachniss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
import numpy as np
import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F

from mos4d.config import MOS4DConfig
from mos4d.metrics import get_confusion_matrix, get_iou, get_precision, get_recall
from mos4d.mos4d import MOS4DNet
from mos4d.utils.augmentation import (
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud,
    rotate_point_cloud,
)


class TrainingModule(LightningModule):
    def __init__(self, config: MOS4DConfig):
        super().__init__()
        self.save_hyperparameters(dict(config))
        self.batch_size = config.training.batch_size
        self.lr = config.training.lr
        self.lr_epoch = config.training.lr_epoch
        self.lr_decay = config.training.lr_decay
        self.weight_decay = config.training.weight_decay
        self.voxel_size = config.mos.voxel_size_mos
        self.mos = MOS4DNet(self.voxel_size, is_student=True)

        self._config_id = dict(config)["training"].id
        self._is_distill_mode = config.training.is_distill_mode
        self._ce_loss = torch.nn.CrossEntropyLoss()
        self._lambda_ce = config.training.lambda_ce

        if config.training.keep_training:
            state_dict = {
                k.replace("model.", ""): v for k, v in torch.load(config.training.keep_training_weights)["state_dict"].items()
            }
            state_dict = {k.replace("mos.", ""): v for k, v in state_dict.items()}
            state_dict = {k: v for k, v in state_dict.items() if "MOSLoss" not in k}
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(("q_projection", "k_projection"))}
            self.mos.load_state_dict(state_dict)
            self.mos.cuda()
            print("이어서 학습을 진행합니다. :", config.training.keep_training_weights)

        if self._is_distill_mode:
            # Distillation chain config
            self._teachers = {}
            self._distill_chain = list(config.training.distill_chain)
            self._distill_stage_index = int(config.training.distill_stage_index)
            self._beam_within_band_keep = float(config.training.beam_within_band_keep)
            self._beam_kmeans_iters = int(config.training.beam_kmeans_iters)
            self._temperature = config.training.temperature

            print("\n\n")
            print("*" * 100)

            # 센서별 teacher를 고유 이름으로 로드
            if config.training.O_teacher_weights:
                self.load_teacher("O", config.training.O_teacher_weights, self.voxel_size)
            if config.training.A_teacher_weights:
                self.load_teacher("A", config.training.A_teacher_weights, self.voxel_size)
            if config.training.V_teacher_weights:
                self.load_teacher("V", config.training.V_teacher_weights, self.voxel_size)
            if config.training.L_teacher_weights:
                self.load_teacher("L", config.training.L_teacher_weights, self.voxel_size)

            self._distill_loss = torch.nn.KLDivLoss(reduction="batchmean")
            self._lambda_kd = config.training.lambda_kd

            """ Attention-based KD를 위한 레이어 추가 """
            # # bottleneck 특징의 차원을 64로 가정
            # bottleneck_dim = 64
            # self.attention_dim = 64  # 어텐션 차원 설정

            # # 학생 bottleneck을 Query로 투영하는 레이어
            # self.q_projection = torch.nn.Linear(bottleneck_dim, self.attention_dim)
            # # 교사 bottleneck을 Key로 투영하는 레이어
            # self.k_projection = torch.nn.Linear(bottleneck_dim, self.attention_dim)

        self.train_reset()
        self.val_reset()

    def load_teacher(self, teacher_name: str, weights: Path, voxel_size: float):
        # Pipeline
        state_dict = {k.replace("model.", ""): v for k, v in torch.load(weights)["state_dict"].items()}
        state_dict = {k.replace("mos.", ""): v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if "MOSLoss" not in k}

        teacher_model = MOS4DNet(voxel_size, is_student=False)
        teacher_model.load_state_dict(state_dict)
        teacher_model.cuda().eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        # 이름으로 보관(등록하지 않아 state_dict 저장에서 제외됨)
        self._teachers[teacher_name] = teacher_model

        print(f"**Teacher {teacher_name} loaded: {weights}**")

    def train_reset(self):
        self.train_confusion_matrix = torch.zeros(2, 2)

    def val_reset(self):
        self.val_confusion_matrix = torch.zeros(2, 2)

    def attention_based_kd_loss(self, teacher_outputs, student_logits_for_kd, student_bottleneck):
        # 1. Query (Student) 및 Keys (Teachers) 준비
        # bottleneck feature들의 전역 평균을 대표 벡터로 사용
        q_feature = torch.mean(student_bottleneck, dim=0, keepdim=True)  # (1, 64)

        teacher_keys = []
        teacher_values = []
        for sensor_name, (t_logits, t_bottleneck_features) in teacher_outputs.items():
            k_feature = torch.mean(t_bottleneck_features, dim=0, keepdim=True)  # (1, 64)
            teacher_keys.append(k_feature)
            teacher_values.append(t_logits)

        # (num_teachers, 1, 64) -> (num_teachers, 64)
        teacher_keys_tensor = torch.stack(teacher_keys).squeeze(1)

        # 2. Q, K를 학습 가능한 레이어에 투영
        q_proj = self.q_projection(q_feature)  # (1, attention_dim)
        k_proj = self.k_projection(teacher_keys_tensor)  # (num_teachers, attention_dim)

        # 3. 어텐션 스코어 및 가중치 계산
        # 스케일링을 포함한 dot-product attention
        attention_scores = torch.matmul(q_proj, k_proj.T) / (self.attention_dim**0.5)  # (1, num_teachers)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (1, num_teachers)

        # 1. 로깅할 딕셔너리 생성
        weights_dict = dict(zip(self._teachers.keys(), attention_weights.flatten().detach().cpu()))
        # 2. TensorBoard/Wandb 등에서 그룹화를 위해 key 이름에 접두사 추가
        logs_to_track = {f"attention_weight/{name}": weight for name, weight in weights_dict.items()}

        # 3. self.log_dict()를 사용하여 한 번에 로깅
        self.log_dict(logs_to_track, on_step=True)

        # 4. 가중치를 이용해 Teacher Logits (Values) 융합
        # (num_teachers, N, 2)
        teacher_values_tensor = torch.stack(teacher_values)
        # 가중치 브로드캐스팅을 위해 차원 변경: (1, num_teachers) -> (num_teachers, 1, 1)
        attention_weights_reshaped = attention_weights.T.unsqueeze(-1)

        # 가중합 계산
        blended_logits = torch.sum(teacher_values_tensor * attention_weights_reshaped, dim=0)  # (N, 2)

        # 5. KD Loss 계산
        soft_teacher = F.softmax(blended_logits / self._temperature, dim=-1)
        soft_student = F.log_softmax(student_logits_for_kd / self._temperature, dim=-1)
        total_kd_loss = self._distill_loss(soft_student, soft_teacher) * (self._temperature**2)

        return total_kd_loss

    def sensor_aware_kd_loss(self, teacher_outputs, student_logits_for_kd, sequence):
        # 교사 이름을 센서 인덱스로 매핑
        teacher_to_sensor_idx = {"O": 0, "V": 1, "A": 2, "L": 3}

        # 각 교사별로 센서별 KD 손실 계산
        for teacher_name, (teacher_logits_for_kd, _) in teacher_outputs.items():
            sensor_idx = teacher_to_sensor_idx[teacher_name]

            # 센서별 마스킹
            specialist_mask = sequence == sensor_idx
            t, s = teacher_logits_for_kd[specialist_mask], student_logits_for_kd[specialist_mask]

            # 전문 센서에 대한 KD 손실
            if len(s) == 0:
                specialist_kd_loss = torch.tensor(0.0, device=student_logits_for_kd.device)
            else:
                soft_student = F.log_softmax(s / self._temperature, dim=-1)
                soft_teacher = F.softmax(t / self._temperature, dim=-1)
                specialist_kd_loss = self._distill_loss(soft_student, soft_teacher) * (self._temperature**2)

            total_kd_loss += specialist_kd_loss

        total_kd_loss /= len(teacher_outputs)
        return total_kd_loss

    def average_based_kd_loss(self, teacher_outputs, student_logits_for_kd):
        averaged_logits = None
        for sensor_name, (teacher_logits_for_kd, _) in teacher_outputs.items():
            if averaged_logits is None:
                averaged_logits = teacher_logits_for_kd
            else:
                averaged_logits += teacher_logits_for_kd
        averaged_logits /= len(teacher_outputs)

        soft_teacher = F.softmax(averaged_logits / self._temperature, dim=-1)
        soft_student = F.log_softmax(student_logits_for_kd / self._temperature, dim=-1)
        total_kd_loss = self._distill_loss(soft_student, soft_teacher) * (self._temperature**2)
        return total_kd_loss

    def training_step(self, batch: torch.Tensor, batch_idx, dataloader_index=0):
        batch, sequence = batch

        ##############################################################################################
        # Skip step if too few moving points
        num_moving_points = len(batch[batch[:, -1] == 1.0])
        num_points = len(batch)
        if num_points == 0 or num_moving_points / num_points < 0.001:
            return None

        before_beam_down_batch, after_beam_down_batch, kept_indices = self.augmentation(
            batch,
            is_pseudo_mode=False,
            is_train_data=True,
        )

        if kept_indices is not None:
            sequence = sequence[kept_indices]

        # Only train if enough points are left (after beam_down)
        if after_beam_down_batch is None or len(after_beam_down_batch) < 100:
            return None
        ##############################################################################################

        # Teacher: before beam-down, Student: after beam-down
        teacher_coordinates, student_coordinates, gt_labels = (
            before_beam_down_batch[:, :5],
            after_beam_down_batch[:, :5],
            after_beam_down_batch[:, -1],
        )

        student_rtn = self.mos.forward(student_coordinates)
        student_logits, student_bottleneck = student_rtn["logits"], student_rtn["bottleneck"]  # (N, 3), (N, 64)
        gt_indices = (gt_labels + 1).to(torch.long)

        """ Cross Entropy Loss """
        ce_loss = self._ce_loss(student_logits, gt_indices)
        self.log("train_ce_loss", ce_loss.item(), on_step=True)

        """ Distillation Loss """
        if self._is_distill_mode:
            # In distillation, we have to use logits[:, 1:] because logits[:, 0] is filled with -inf
            student_logits_for_kd = student_logits[:, 1:]  # (N, 2) -> static, moving
            teacher_outputs = {}
            with torch.no_grad():
                for sensor_name, teacher_model in self._teachers.items():
                    teacher_rtn = teacher_model.forward(teacher_coordinates)
                    t_logits = teacher_rtn["logits"][:, 1:]  # (N, 2)
                    t_bottleneck = teacher_rtn["bottleneck"]  # (N, 64)
                    teacher_outputs[sensor_name] = (t_logits, t_bottleneck)

            """KD-Loss by version"""
            # kd_loss = self.attention_based_kd_loss(teacher_outputs, student_logits_for_kd, student_bottleneck)
            # kd_loss = self.average_based_kd_loss(teacher_outputs, student_logits_for_kd)
            kd_loss = self.sensor_aware_kd_loss(teacher_outputs, student_logits_for_kd, sequence)

        """ Total Loss with Weights"""
        loss = self._lambda_ce * ce_loss
        if self._is_distill_mode:
            loss += self._lambda_kd * kd_loss
            self.log("train_kd_loss", (self._lambda_kd * kd_loss).item(), on_step=True)

        self.log("train_loss", loss.item(), on_step=True)

        student_logits = student_logits.detach().cpu()
        gt_labels = gt_labels.detach().cpu()
        pred_logits = self.mos.to_single_logit(student_logits)
        pred_labels = self.mos.to_label(pred_logits)

        self.train_confusion_matrix += get_confusion_matrix(pred_labels, gt_labels)

        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        iou = get_iou(self.train_confusion_matrix)
        recall = get_recall(self.train_confusion_matrix)
        precision = get_precision(self.train_confusion_matrix)
        self.log("train_moving_iou", iou[1].item())
        self.log("train_moving_recall", recall[1].item())
        self.log("train_moving_precision", precision[1].item())
        self.train_reset()
        torch.cuda.empty_cache()

    def validation_step(self, batch: torch.Tensor, batch_idx):
        batch, _ = batch
        if len(batch) < 100:
            return None

        batch, _, _ = self.augmentation(batch, is_pseudo_mode=False, is_train_data=False)
        coordinates = batch[:, :5].reshape(-1, 5)
        gt_labels = batch[:, -1].reshape(-1)

        student_rtn = self.mos.forward(coordinates)
        logits = student_rtn["logits"]

        gt_indices = (gt_labels + 1).to(torch.long)
        loss = self._ce_loss(logits, gt_indices)

        self.log("val_loss", loss.item(), batch_size=len(batch), prog_bar=True, on_epoch=True)

        logits = logits.detach().cpu()
        gt_labels = gt_labels.detach().cpu()

        pred_logits = self.mos.to_single_logit(logits)
        pred_labels = self.mos.to_label(pred_logits)

        # Logging metrics
        self.val_confusion_matrix += get_confusion_matrix(pred_labels, gt_labels)
        torch.cuda.empty_cache()
        return loss

    def on_validation_epoch_end(self):
        iou = get_iou(self.val_confusion_matrix)
        recall = get_recall(self.val_confusion_matrix)
        precision = get_precision(self.val_confusion_matrix)
        print(f"[Running Validation : {self._config_id}] | IoU: {iou[1].item():.4f}")
        self.log("val_moving_iou", iou[1].item())
        self.log("val_moving_recall", recall[1].item())
        self.log("val_moving_precision", precision[1].item())
        self.val_reset()
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_epoch, gamma=self.lr_decay)
        return [optimizer], [scheduler]

    def augmentation(self, batch, is_pseudo_mode, is_train_data):
        if is_train_data:
            # crop 단계에서 인덱스 추적
            crop_mask = self.crop_with_mask(batch)
            batch = batch[crop_mask]

            # 기하학적 변환 (인덱스 변화 없음)
            batch[:, 1:4] = rotate_point_cloud(batch[:, 1:4])
            batch[:, 1:4] = rotate_perturbation_point_cloud(batch[:, 1:4])
            batch[:, 1:4] = random_flip_point_cloud(batch[:, 1:4])
            batch[:, 1:4] = random_scale_point_cloud(batch[:, 1:4])

            # subsample 단계에서 인덱스 추적
            batch, subsample_mask = self.subsample(batch)

            # 전체 인덱스 계산 (crop + subsample)
            final_indices = torch.where(crop_mask)[0][subsample_mask]

        if not is_pseudo_mode:
            return batch, batch, final_indices if is_train_data else None

        # Build before/after according to chain stage
        chain = self._distill_chain
        stage = max(0, min(self._distill_stage_index, len(chain) - 2))
        # Desired mapping: chain[stage] -> chain[stage+1]
        desired_before = int(chain[stage])
        desired_after = int(chain[stage + 1])

        # Start from the original batch (assumed highest resolution)
        current_batch = batch

        # If current desired_before is less than original top of chain, cascade down until we reach it
        top_beams = int(chain[0])
        if desired_before < top_beams:
            # Walk the chain from top until desired_before, without keeping indices
            for src, tgt in zip(chain, chain[1:]):
                src = int(src)
                tgt = int(tgt)
                current_batch = self.beam_down(
                    current_batch,
                    src_clusters=src,
                    target_beams=tgt,
                    kmeans_iters=self._beam_kmeans_iters,
                    within_band_keep=self._beam_within_band_keep,
                    return_indices=False,
                )
                if tgt == desired_before:
                    break

        before_batch = current_batch

        # Now produce after from before, keeping indices w.r.t. before
        after_batch, kept = self.beam_down(
            before_batch,
            src_clusters=desired_before,
            target_beams=desired_after,
            kmeans_iters=self._beam_kmeans_iters,
            within_band_keep=self._beam_within_band_keep,
            return_indices=True,
        )

        return before_batch, after_batch, kept

    def crop(self, batch):
        sample_point = batch[np.random.choice(range(len(batch))), 1:4]
        crop_x = np.random.normal(15, 2)
        crop_y = np.random.normal(15, 2)

        dist = torch.abs(batch[:, 1:4] - sample_point).reshape(-1, 3)
        mask = dist[:, 0] < crop_x
        mask = torch.logical_and(mask, dist[:, 1] < crop_y)
        return batch[mask]

    def crop_with_mask(self, batch):
        sample_point = batch[np.random.choice(range(len(batch))), 1:4]
        crop_x = np.random.normal(15, 2)
        crop_y = np.random.normal(15, 2)

        dist = torch.abs(batch[:, 1:4] - sample_point).reshape(-1, 3)
        mask = dist[:, 0] < crop_x
        mask = torch.logical_and(mask, dist[:, 1] < crop_y)
        return mask

    def subsample(self, batch, max_dropout_ratio=0.5):
        dropout = (1 - max_dropout_ratio) * torch.rand(1) + max_dropout_ratio
        keep = torch.rand(len(batch)) < dropout
        return batch[keep], keep

    def beam_down(
        self,
        batch: torch.Tensor,
        src_clusters: int,
        target_beams: int,
        kmeans_iters: int = 10,
        within_band_keep: float = 0.5,
        return_indices: bool = False,
    ) -> torch.Tensor:
        """
        LiDAR Distillation 스타일의 pseudo low-beam 생성.
        - 각 샘플(batch_idx별)에서 고도각(theta)을 1D K-Means로 클러스터링하여 소스 빔을 근사
        - target_beams 개의 클러스터만 선택(centroid 정렬 후 균등 간격 선택)
        - 선택된 클러스터 내부에서 확률 다운샘플로 밀도 정렬

        Args:
            batch: [N, 6] = [batch_idx, x, y, z, t, label]
            src_clusters: 소스 빔 근사용 클러스터 개수(K)
            target_beams: 목표 빔 수(예: 16)
            kmeans_iters: K-Means 반복 횟수
            within_band_keep: 선택된 클러스터 내부 보존 확률(0~1)
        """
        if batch.numel() == 0:
            return batch

        device, dtype = batch.device, batch.dtype
        # 결과 동일성 보장을 위해 수학은 그대로 두되, 그래디언트 추적을 끄어 오버헤드만 제거
        with torch.no_grad():
            batch_idx_vec = batch[:, 0].to(torch.long)
            unique_batches = torch.unique(batch_idx_vec).tolist()

            kept_indices = []

            def kmeans_1d(values: torch.Tensor, k: int, iters: int):
                # values: [M]
                # 초기 중심: 구간 균등 배치
                vmin, vmax = values.min(), values.max()
                if (vmax - vmin).abs() < 1e-6:
                    centers = vmin + torch.linspace(0, 1, k, device=values.device, dtype=values.dtype) * 0
                else:
                    centers = torch.linspace(vmin, vmax, k, device=values.device, dtype=values.dtype)
                labels = torch.zeros_like(values, dtype=torch.long)
                for _ in range(max(1, iters)):
                    # 할당
                    d = (values[:, None] - centers[None, :]).abs()
                    labels = torch.argmin(d, dim=1)
                    # 업데이트
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

            for b in unique_batches:
                mask_b = batch_idx_vec == b
                if not mask_b.any():
                    continue
                xyz = batch[mask_b, 1:4]
                x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                r_xy = torch.sqrt(torch.clamp(x * x + y * y, min=1e-12))
                theta = torch.atan2(z, r_xy)  # [Mb]

                K = max(1, int(src_clusters))
                labels, centers = kmeans_1d(theta, K, kmeans_iters)

                # 중심 정렬 후 target_beams개 균등 간격 선택
                sorted_centers, order = torch.sort(centers)
                if target_beams >= K:
                    keep_clusters = torch.arange(K, device=device)
                else:
                    idx_lin = torch.linspace(0, K - 1, steps=target_beams, device=device)
                    keep_clusters = idx_lin.round().to(torch.long)
                # 원래 라벨 인덱스로 변환
                keep_labels = order[keep_clusters]

                # 마스크 생성 + 밴드 내 확률 보존
                # isin 대신 브로드캐스트 비교로 동일 결과, 더 빠름
                mask_kept = (labels.unsqueeze(1) == keep_labels.unsqueeze(0)).any(dim=1)
                if within_band_keep < 1.0:
                    rand_keep = torch.rand(mask_kept.shape[0], device=device) < float(within_band_keep)
                    mask_kept = mask_kept & rand_keep

                kept_local = torch.where(mask_b)[0][mask_kept]
                if kept_local.numel() > 0:
                    kept_indices.append(kept_local)

            if len(kept_indices) == 0:
                return (batch, None) if return_indices else batch
            kept_indices = torch.cat(kept_indices, dim=0)
            return (batch[kept_indices], kept_indices) if return_indices else batch[kept_indices]

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        체크포인트를 저장하기 직전에 호출됩니다.
        여기서 teacher 모델의 가중치를 state_dict에서 제거합니다.
        """
        # checkpoint 딕셔너리에서 모델의 state_dict를 가져옵니다.
        model_state_dict = checkpoint["state_dict"]

        # '_teacher.'로 시작하지 않는 키만 남겨서 새로운 state_dict를 생성합니다.
        new_state_dict = {key: val for key, val in model_state_dict.items() if not key.startswith("_teacher.")}

        # 기존 state_dict를 teacher 가중치가 제거된 새로운 state_dict로 교체합니다.
        checkpoint["state_dict"] = new_state_dict
