# Adapted from 4DMOS (https://github.com/PRBonn/4DMOS), MIT License, see LICENSE
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningModule

from core.config import MOS4DConfig
from distillation import (
    KD_MODES,
    attention_based_kd_loss,
    average_based_kd_loss,
    load_teachers,
    make_pseudo_pair,
    sensor_aware_kd_loss,
)
from core.metrics import get_confusion_matrix, get_iou, get_precision, get_recall
from networks.mos4d import MOS4DNet
from datasets.augmentation import (
    random_crop_mask,
    random_flip_point_cloud,
    random_scale_point_cloud,
    random_subsample_mask,
    rotate_perturbation_point_cloud,
    rotate_point_cloud,
)
from core.checkpoint import load_mos_state_dict

# Number of channels of the bottleneck of the MinkUNet, used by the attention-based distillation
BOTTLENECK_DIM = 64
ATTENTION_DIM = 64


class TrainingModule(LightningModule):
    """Trains a MOS4DNet in one of the stages of the X-MOS framework, selected by the config:

    - supervised training (easy-to-learn teachers, baselines, fine-tuning of the V teacher)
    - `pseudo_beam_down`: one stage of the multi-level adaptive teacher generation, where the
      network learns on pseudo low-beam data and mimics a teacher that sees the denser data.
      Teacher and student are roles within a stage, the student of a stage is the teacher of the
      next one (see `distillation.make_pseudo_pair`)
    - `is_distill_mode`: knowledge distillation from the configured sensor-specific teachers
    - `keep_training`: weight transfer from a previous stage
    """

    def __init__(self, config: MOS4DConfig):
        super().__init__()
        self.save_hyperparameters(dict(config))
        training = config.training
        self.batch_size = training.batch_size
        self.lr = training.lr
        self.lr_epoch = training.lr_epoch
        self.lr_decay = training.lr_decay
        self.weight_decay = training.weight_decay
        self.voxel_size = config.mos.voxel_size_mos
        self.mos = MOS4DNet(self.voxel_size)

        self._config_id = training.id
        self._ce_loss = torch.nn.CrossEntropyLoss()
        self._lambda_ce = training.lambda_ce

        # Weight transfer
        if training.keep_training:
            self.mos.load_state_dict(load_mos_state_dict(training.keep_training_weights))
            self.mos.cuda()
            print("Continuing the training from:", training.keep_training_weights)

        # Pseudo low-beam data generation
        self._pseudo_beam_down = training.pseudo_beam_down
        self._distill_chain = list(training.distill_chain)
        self._distill_stage_index = int(training.distill_stage_index)
        self._beam_within_band_keep = float(training.beam_within_band_keep)
        self._beam_kmeans_iters = int(training.beam_kmeans_iters)

        # Knowledge distillation
        self._is_distill_mode = training.is_distill_mode
        if self._is_distill_mode:
            assert training.kd_mode in KD_MODES, f"kd_mode must be one of {KD_MODES}"
            self._kd_mode = training.kd_mode
            self._lambda_kd = training.lambda_kd
            self._temperature = training.temperature

            print("\n\n" + "*" * 100)
            # Kept in a plain dict, the teachers must not be part of the student checkpoint
            self._teachers = load_teachers(training, self.voxel_size)
            assert len(self._teachers) > 0, "Distillation needs at least one teacher"

            if self._kd_mode == "attention":
                # Projections of the student (query) and teacher (key) bottleneck features
                self.q_projection = torch.nn.Linear(BOTTLENECK_DIM, ATTENTION_DIM)
                self.k_projection = torch.nn.Linear(BOTTLENECK_DIM, ATTENTION_DIM)

        self.train_reset()
        self.val_reset()

    def train_reset(self):
        self.train_confusion_matrix = torch.zeros(2, 2)

    def val_reset(self):
        self.val_confusion_matrix = torch.zeros(2, 2)

    # Training -------------------------------------------------------------------------------------
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx, dataloader_index=0):
        batch, sensor_ids = batch

        # Skip step if too few moving points
        num_moving_points = len(batch[batch[:, -1] == 1.0])
        num_points = len(batch)
        if num_points == 0 or num_moving_points / num_points < 0.001:
            return None

        teacher_batch, student_batch, sensor_ids, teacher_to_student = self.prepare_training_batch(batch, sensor_ids)

        # Only train if enough points are left
        if student_batch is None or len(student_batch) < 100:
            return None

        student_coordinates, gt_labels = student_batch[:, :5], student_batch[:, -1]
        student_out = self.mos.forward(student_coordinates)
        student_logits = student_out["logits"]  # (N, 3)

        # Supervised MOS loss
        gt_indices = (gt_labels + 1).to(torch.long)
        ce_loss = self._ce_loss(student_logits, gt_indices)
        self.log("train_ce_loss", ce_loss.item(), on_step=True)
        loss = self._lambda_ce * ce_loss

        # Distillation loss
        if self._is_distill_mode:
            kd_loss = self.distillation_loss(
                teacher_batch[:, :5], teacher_to_student, student_out, sensor_ids
            )
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

    def prepare_training_batch(self, batch: torch.Tensor, sensor_ids: torch.Tensor):
        """Augments the batch [N, 6] = [batch_idx, x, y, z, t, label].

        Returns:
            teacher_batch: input of the teachers
            student_batch: input of the student
            sensor_ids: sensor index of each point of the student batch
            teacher_to_student: indices of the student points w.r.t. the teacher batch, or None if
                both see the same points
        """
        # Crop
        crop_mask = random_crop_mask(batch)
        batch = batch[crop_mask]

        # Geometric transformations
        batch[:, 1:4] = rotate_point_cloud(batch[:, 1:4])
        batch[:, 1:4] = rotate_perturbation_point_cloud(batch[:, 1:4])
        batch[:, 1:4] = random_flip_point_cloud(batch[:, 1:4])
        batch[:, 1:4] = random_scale_point_cloud(batch[:, 1:4])

        # Subsample
        subsample_mask = random_subsample_mask(len(batch))
        batch = batch[subsample_mask]
        sensor_ids = sensor_ids[torch.where(crop_mask)[0][subsample_mask]]

        if not self._pseudo_beam_down:
            return batch, batch, sensor_ids, None

        # The teacher sees the higher and the student the lower resolution of the chain stage
        teacher_batch, student_batch, teacher_to_student = self.make_pseudo_pair(batch)
        if teacher_to_student is not None:
            sensor_ids = sensor_ids[teacher_to_student]
        return teacher_batch, student_batch, sensor_ids, teacher_to_student

    def make_pseudo_pair(self, batch: torch.Tensor):
        return make_pseudo_pair(
            batch,
            chain=self._distill_chain,
            stage_index=self._distill_stage_index,
            kmeans_iters=self._beam_kmeans_iters,
            within_band_keep=self._beam_within_band_keep,
        )

    def distillation_loss(
        self,
        teacher_coordinates: torch.Tensor,
        teacher_to_student: Optional[torch.Tensor],
        student_out: dict,
        sensor_ids: torch.Tensor,
    ):
        # Only the (static, moving) logits are distilled, the first logit is always -inf
        student_logits = student_out["logits"][:, 1:]  # (N, 2)

        teacher_outputs = {}
        with torch.no_grad():
            for teacher_key, teacher in self._teachers.items():
                teacher_out = teacher.forward(teacher_coordinates)
                teacher_logits = teacher_out["logits"][:, 1:]
                if teacher_to_student is not None:
                    teacher_logits = teacher_logits[teacher_to_student]
                teacher_outputs[teacher_key] = (teacher_logits, teacher_out["bottleneck"])

        if self._kd_mode == "sensor_aware":
            return sensor_aware_kd_loss(teacher_outputs, student_logits, sensor_ids, self._temperature)
        if self._kd_mode == "average":
            return average_based_kd_loss(teacher_outputs, student_logits, self._temperature)

        kd_loss, attention_weights = attention_based_kd_loss(
            teacher_outputs,
            student_logits,
            student_out["bottleneck"],
            self.q_projection,
            self.k_projection,
            self._temperature,
        )
        self.log_dict({f"attention_weight/{key}": w for key, w in attention_weights.items()}, on_step=True)
        return kd_loss

    def on_train_epoch_end(self):
        iou = get_iou(self.train_confusion_matrix)
        recall = get_recall(self.train_confusion_matrix)
        precision = get_precision(self.train_confusion_matrix)
        self.log("train_moving_iou", iou[1].item())
        self.log("train_moving_recall", recall[1].item())
        self.log("train_moving_precision", precision[1].item())
        self.train_reset()
        torch.cuda.empty_cache()

    # Validation -----------------------------------------------------------------------------------
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        batch, _ = batch
        if len(batch) < 100:
            return None

        # Validate on the same pseudo resolution the network is trained on
        if self._pseudo_beam_down:
            _, batch, _ = self.make_pseudo_pair(batch)

        coordinates = batch[:, :5].reshape(-1, 5)
        gt_labels = batch[:, -1].reshape(-1)

        logits = self.mos.forward(coordinates)["logits"]

        gt_indices = (gt_labels + 1).to(torch.long)
        loss = self._ce_loss(logits, gt_indices)

        self.log("val_loss", loss.item(), batch_size=len(batch), prog_bar=True, on_epoch=True)

        logits = logits.detach().cpu()
        gt_labels = gt_labels.detach().cpu()

        pred_logits = self.mos.to_single_logit(logits)
        pred_labels = self.mos.to_label(pred_logits)

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
