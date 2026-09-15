import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from dynaslots.model.common.normalizer import LinearNormalizer
from dynaslots.common.pytorch_util import dict_apply
from dynaslots.common.model_util import print_params
from dynaslots.model.vision.pointnet_extractor import VisEncoder, DiTFDMDecoder
from dynaslots.model.vision.slot_binder import (
    SlotBinder,
    SlotInteractionHead,
)


class DynaSlots(nn.Module):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            # ===== param =====
            ema_momentum_start=0.996,
            ema_target_epoch=300,
            lambda_long_term=1.0,
            lambda_reverse=1.0,
            fdm_d_model=256,
            vicreg_inv_weight=25.0,
            vicreg_var_weight=25.0,
            vicreg_cov_weight=1.0,
            vicreg_eps=1e-4,
            vicreg_warmup_start=0,        
            vicreg_warmup_epochs=0,   
            use_dynaslots=False,
            num_slots=6,
            slot_iterations=3,
            frame_stride=4,
            slot_graph_depth=3,
            lambda_ifdm=1.0,
            lambda_interaction=0.1,
            lambda_slot_entropy=0.1,
            lambda_slot_balance=0.2,
            slot_balance_warmup_epochs=30,
            contact_radius_ratio=0.2,
            motion_threshold=0.02,
            relation_gating=False,
            relation_topk=3,
            relation_gate_floor=0.05,
            relation_gate_temperature=1.0,
            relation_gate_warmup_epochs=30,
            lambda_relation_consistency=0.0,
            **kwargs):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: 
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # ===== Student / Teacher encoders (Teacher = EMA of Student) =====
        self.vis_encoder = VisEncoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        self.ema_vis_encoder = VisEncoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        # A bootstrap target must start from the online encoder rather than an
        # unrelated random initialization.
        self.ema_vis_encoder.load_state_dict(self.vis_encoder.state_dict())
        for p in self.ema_vis_encoder.parameters():
            p.requires_grad = False

        self.obs_feature_dim = self.vis_encoder.output_shape()  # D
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DynaSlots] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DynaSlots] pointnet_type: {self.pointnet_type}", "yellow")

        D = self.obs_feature_dim
        self.use_dynaslots = bool(use_dynaslots)
        self.num_slots = int(num_slots)
        self.frame_stride = int(frame_stride)
        self.lambda_ifdm = float(lambda_ifdm)
        self.lambda_interaction = float(lambda_interaction)
        self.lambda_slot_entropy = float(lambda_slot_entropy)
        self.lambda_slot_balance = float(lambda_slot_balance)
        self.slot_balance_warmup_epochs = int(slot_balance_warmup_epochs)
        self.contact_radius_ratio = float(contact_radius_ratio)
        self.motion_threshold = float(motion_threshold)
        self.relation_gating = bool(relation_gating)
        self.relation_gate_warmup_epochs = int(relation_gate_warmup_epochs)
        self.lambda_relation_consistency = float(lambda_relation_consistency)

        if self.use_dynaslots and self.pointnet_type != "pointtransformer":
            raise ValueError(
                "DynaSlots needs pointnet_type='pointtransformer' because "
                "pooled PointNet features have no point tokens"
            )

        # bookkeeping
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.kwargs = kwargs

        # ===== IDM: Δf -> 16 dim =====
        self.latent_action_dim = 16
        self.idm_mlp = nn.Sequential(
            nn.Linear(D, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.latent_action_dim)
        )

        # ===== FDM =====
        if self.use_dynaslots:
            self.slot_binder = SlotBinder(
                slot_dim=D,
                num_slots=self.num_slots,
                num_iterations=slot_iterations,
            )
            self.ema_slot_binder = None
            if self.relation_gating:
                self.ema_slot_binder = copy.deepcopy(self.slot_binder)
                for parameter in self.ema_slot_binder.parameters():
                    parameter.requires_grad = False
            self.slot_idm_mlp = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, self.latent_action_dim),
            )
            self.interaction_head = SlotInteractionHead(
                slot_dim=D,
                action_dim=self.latent_action_dim,
            )
            self.fdm_vis_decoder = DiTFDMDecoder(
                input_dim=D,
                hidden=fdm_d_model,
                depth=4,
                num_heads=4,
                num_tokens=self.num_slots,
                latent_action_dim=self.latent_action_dim,
                graph_depth=slot_graph_depth,
                relation_gating=self.relation_gating,
                relation_topk=relation_topk,
                relation_gate_floor=relation_gate_floor,
                relation_gate_temperature=relation_gate_temperature,
                param_type="eps",
            )
        else:
            self.fdm_vis_decoder = DiTFDMDecoder(
                input_dim=D,
                hidden=fdm_d_model,
                depth=4,
                num_heads=4,
            )
        self.noise_scheduler = noise_scheduler

        # training details
        self.lambda_long_term = lambda_long_term
        self.lambda_reverse = lambda_reverse

        # EMA 
        self.ema_momentum_start = float(ema_momentum_start)
        self.ema_target_epoch = int(ema_target_epoch)

        # VICReg 
        self.vicreg_inv_weight = float(vicreg_inv_weight)
        self.vicreg_var_weight = float(vicreg_var_weight)
        self.vicreg_cov_weight = float(vicreg_cov_weight)
        self.vicreg_eps = vicreg_eps

        self.vicreg_warmup_start = int(vicreg_warmup_start)
        self.vicreg_warmup_epochs = int(vicreg_warmup_epochs)

        print_params(self)

    def forward(self, batch, epoch):
        """DDP-compatible training entry point.

        Calling ``compute_loss`` directly bypasses DistributedDataParallel's
        forward bookkeeping.  Keep the public loss method for the existing
        single-GPU code and expose the same operation through ``forward`` for
        multi-GPU runs.
        """
        return self.compute_loss(batch, epoch)

    # ========= VICReg=========
    def _vicreg_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                    inv_w: float, var_w: float, cov_w: float):

        inv_loss = F.mse_loss(z1, z2)

        z1c = z1 - z1.mean(dim=0, keepdim=True)
        z2c = z2 - z2.mean(dim=0, keepdim=True)

        eps = getattr(self, "vicreg_eps", 1e-4)
        std1 = torch.sqrt(z1c.var(dim=0, unbiased=False) + eps)
        std2 = torch.sqrt(z2c.var(dim=0, unbiased=False) + eps)
        var_loss = 0.5 * (F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean())

        def _covariance_term(zc: torch.Tensor) -> torch.Tensor:
            N, D = zc.shape
            if N <= 1:
                return zc.new_zeros(())
            cov = (zc.T @ zc) / (N - 1.0)         # [D, D]
            off = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
            return off / D

        cov_loss = _covariance_term(z1c) + _covariance_term(z2c)

        vicreg = inv_w * inv_loss + var_w * var_loss + cov_w * cov_loss
        return vicreg, inv_loss, var_loss, cov_loss

    def _warmup_scale(self, epoch: int) -> float:
        start = self.vicreg_warmup_start
        warm = self.vicreg_warmup_epochs
        if warm <= 0:
            return 1.0
        if epoch < start:
            return 0.0
        if epoch >= start + warm:
            return 1.0

        e = epoch - start
        return 0.5 * (1.0 - math.cos(math.pi * float(e) / float(warm)))

    # ========= EMA =========
    def _ema_momentum(self, epoch: int) -> float:
        e = max(0, min(epoch, self.ema_target_epoch))
        frac = e / float(self.ema_target_epoch) if self.ema_target_epoch > 0 else 1.0
        return self.ema_momentum_start + (1.0 - self.ema_momentum_start) * frac

    @torch.no_grad()
    def update_teacher(self, epoch: int):
        m = self._ema_momentum(epoch)

        def _ema_update_module(tgt: nn.Module, src: nn.Module):
            for p_t, p_s in zip(tgt.parameters(), src.parameters()):
                p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))
            for b_t, b_s in zip(tgt.buffers(), src.buffers()):
                if b_t.dtype.is_floating_point:
                    b_t.data.mul_(m).add_(b_s.data, alpha=(1.0 - m))
                else:
                    b_t.data.copy_(b_s.data)

        _ema_update_module(self.ema_vis_encoder, self.vis_encoder)
        if self.use_dynaslots and self.ema_slot_binder is not None:
            _ema_update_module(self.ema_slot_binder, self.slot_binder)

    # ========= Utils =========
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor):
        weights = weights.to(values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    @staticmethod
    def _slot_balance_loss(assignments: torch.Tensor):
        """KL(batch slot-load || uniform), with assignments [B,T,K,N]."""
        if assignments.ndim != 4:
            raise ValueError("assignments must have shape [B, T, K, N]")
        num_slots = assignments.shape[2]
        occupancy = assignments.mean(dim=-1).mean(dim=(0, 1))
        occupancy = occupancy.clamp_min(1e-8)
        loss = math.log(float(num_slots)) + (
            occupancy * occupancy.log()
        ).sum()
        return loss, occupancy

    def _slot_vicreg_loss(self, student, target, activations,
                          inv_w, var_w, cov_w):
        """VICReg over corresponding slots without mixing slot identities.

        Variance/covariance statistics are computed across batch and time for
        each slot index. An additional cross-slot covariance penalty prevents
        all slots from becoming homogeneous.
        """
        if student.shape != target.shape or student.ndim != 4:
            raise ValueError("slot VICReg expects matching [B, T, K, D]")
        B, T, K, D = student.shape
        sample_count = B * T
        student_flat = student.reshape(sample_count, K, D)
        target_flat = target.reshape(sample_count, K, D)
        gate = activations.detach().reshape(sample_count, K)

        inv_per_slot = (student_flat - target_flat).pow(2).mean(dim=-1)
        inv_loss = self._weighted_mean(inv_per_slot, gate)

        def variance_and_covariance(features):
            centered = features - features.mean(dim=0, keepdim=True)
            std = torch.sqrt(
                centered.var(dim=0, unbiased=False) + self.vicreg_eps
            )
            var_loss = F.relu(1.0 - std).mean()
            if sample_count <= 1:
                return var_loss, centered.new_zeros(())

            covariance_loss = centered.new_zeros(())
            for slot_index in range(K):
                slot_features = centered[:, slot_index]
                covariance = (
                    slot_features.T @ slot_features / (sample_count - 1.0)
                )
                off_diagonal = (
                    covariance.pow(2).sum()
                    - covariance.diagonal().pow(2).sum()
                ) / D
                covariance_loss = covariance_loss + off_diagonal
            covariance_loss = covariance_loss / K

            cross_slot_loss = centered.new_zeros(())
            pair_count = 0
            for slot_i in range(K):
                for slot_j in range(slot_i + 1, K):
                    cross_covariance = (
                        centered[:, slot_i].T @ centered[:, slot_j]
                        / (sample_count - 1.0)
                    )
                    cross_slot_loss = cross_slot_loss + cross_covariance.pow(2).mean()
                    pair_count += 1
            if pair_count:
                covariance_loss = covariance_loss + cross_slot_loss / pair_count
            return var_loss, covariance_loss

        student_var, student_cov = variance_and_covariance(student_flat)
        target_var, target_cov = variance_and_covariance(target_flat)
        var_loss = 0.5 * (student_var + target_var)
        cov_loss = student_cov + target_cov
        total = inv_w * inv_loss + var_w * var_loss + cov_w * cov_loss
        return total, inv_loss, var_loss, cov_loss

    def _bind_three_frames(self, nobs, frame_indices):
        online_outputs = []
        target_outputs = []
        previous_online = None
        previous_target = None

        for frame_index in frame_indices:
            frame_obs = {
                key: value[:, frame_index]
                for key, value in nobs.items()
            }
            coordinates = frame_obs["point_cloud"][..., :3]
            online_tokens = self.vis_encoder.forward_tokens(frame_obs)
            online = self.slot_binder(
                online_tokens, coordinates, previous_slots=previous_online
            )
            previous_online = online["slots"]
            online_outputs.append(online)

            with torch.no_grad():
                target_tokens = self.ema_vis_encoder.forward_tokens(frame_obs)
                target_binder = (
                    self.ema_slot_binder
                    if self.ema_slot_binder is not None
                    else self.slot_binder
                )
                target = target_binder(
                    target_tokens, coordinates, previous_slots=previous_target
                )
                previous_target = target["slots"]
                target_outputs.append(target)

        def stack(outputs, key):
            return torch.stack([output[key] for output in outputs], dim=1)

        online = {
            key: stack(online_outputs, key) for key in online_outputs[0]
        }
        target = {
            key: stack(target_outputs, key) for key in target_outputs[0]
        }
        coordinates = torch.stack(
            [nobs["point_cloud"][:, index, :, :3] for index in frame_indices],
            dim=1,
        )
        return online, target, coordinates

    def _add_slot_noise(self, clean_slots, noise, timesteps):
        flat_shape = (-1, clean_slots.shape[-1])
        noisy = self.noise_scheduler.add_noise(
            clean_slots.reshape(flat_shape),
            noise.reshape(flat_shape),
            timesteps.reshape(-1),
        )
        return noisy.reshape_as(clean_slots)

    def _interaction_pseudo_labels(self, centroids_t, centroids_tp1,
                                   points_t, points_tp1):
        scene_points = torch.cat([points_t, points_tp1], dim=1)
        scene_diagonal = (
            scene_points.amax(dim=1) - scene_points.amin(dim=1)
        ).norm(dim=-1)
        contact_radius = (
            self.contact_radius_ratio * scene_diagonal
        ).view(-1, 1, 1)

        close = (
            torch.cdist(centroids_t, centroids_t) < contact_radius
        ) | (
            torch.cdist(centroids_tp1, centroids_tp1) < contact_radius
        )
        displacement = (centroids_tp1 - centroids_t).norm(dim=-1)
        moving = displacement > self.motion_threshold
        either_moving = moving.unsqueeze(2) | moving.unsqueeze(1)
        labels = close & either_moving

        diagonal = torch.eye(
            labels.shape[-1], dtype=torch.bool, device=labels.device
        ).unsqueeze(0)
        return labels & ~diagonal

    def _compute_dynaslots_loss(self, nobs, epoch, inv_w, var_w, cov_w):
        B, T = nobs["point_cloud"].shape[:2]
        required_frames = 2 * self.frame_stride + 1
        if T < required_frames:
            raise ValueError(
                f"DynaSlots requires horizon >= {required_frames}, got {T}"
            )
        max_start = T - required_frames
        start = int(torch.randint(
            0, max_start + 1, (1,), device=nobs["point_cloud"].device
        ).item())
        frame_indices = [
            start,
            start + self.frame_stride,
            start + 2 * self.frame_stride,
        ]

        online, target, point_sequence = self._bind_three_frames(
            nobs, frame_indices
        )
        student_slots = online["slots"]
        target_slots = target["slots"].detach()
        B, _, K, D = student_slots.shape

        current_slots = student_slots[:, :-1]
        next_student_slots = student_slots[:, 1:]
        next_target_slots = target_slots[:, 1:]
        centroids_t = online["centroids"][:, :-1]
        centroids_tp1 = online["centroids"][:, 1:]
        latent_actions = self.slot_idm_mlp(
            next_student_slots - current_slots
        )

        pair_batch = B * 2
        current_flat = current_slots.reshape(pair_batch, K, D)
        target_flat = next_target_slots.reshape(pair_batch, K, D)
        centroid_flat = centroids_t.reshape(pair_batch, K, 3)
        action_flat = latent_actions.reshape(
            pair_batch, K, self.latent_action_dim
        )
        interaction_logits = self.interaction_head(
            current_flat, centroid_flat, action_flat
        )
        gate_strength = 1.0
        if self.relation_gate_warmup_epochs > 0:
            gate_strength = min(
                1.0, float(epoch + 1) / self.relation_gate_warmup_epochs
            )
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (pair_batch, K),
            device=student_slots.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(target_flat)
        noisy_target = self._add_slot_noise(target_flat, noise, timesteps)
        predicted_noise = self.fdm_vis_decoder(
            noisy_target,
            timesteps,
            current_flat,
            action_flat,
            centroids=centroid_flat,
            relation_logits=(
                interaction_logits if self.relation_gating else None
            ),
            relation_gate_strength=gate_strength,
        )
        dynamics_gate = target["activations"][:, 1:].reshape(
            pair_batch, K
        ).detach()
        ifdm_per_slot = (predicted_noise - noise).pow(2).mean(dim=-1)
        ifdm_loss = self._weighted_mean(ifdm_per_slot, dynamics_gate)

        reverse_actions = self.slot_idm_mlp(
            current_slots - next_student_slots
        ).reshape(pair_batch, K, self.latent_action_dim)
        reverse_current = next_student_slots.reshape(pair_batch, K, D)
        reverse_centroids = centroids_tp1.reshape(pair_batch, K, 3)
        reverse_interaction_logits = self.interaction_head(
            reverse_current, reverse_centroids, reverse_actions
        )
        reverse_target = target_slots[:, :-1].reshape(pair_batch, K, D)
        reverse_timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (pair_batch, K),
            device=student_slots.device,
            dtype=torch.long,
        )
        reverse_noise = torch.randn_like(reverse_target)
        noisy_reverse_target = self._add_slot_noise(
            reverse_target, reverse_noise, reverse_timesteps
        )
        predicted_reverse_noise = self.fdm_vis_decoder(
            noisy_reverse_target,
            reverse_timesteps,
            reverse_current,
            reverse_actions,
            centroids=reverse_centroids,
            relation_logits=(
                reverse_interaction_logits if self.relation_gating else None
            ),
            relation_gate_strength=gate_strength,
        )
        reverse_gate = target["activations"][:, :-1].reshape(
            pair_batch, K
        ).detach()
        reverse_per_slot = (
            predicted_reverse_noise - reverse_noise
        ).pow(2).mean(dim=-1)
        reverse_loss = self._weighted_mean(reverse_per_slot, reverse_gate)

        slot_vicreg, inv_loss, var_loss, cov_loss = self._slot_vicreg_loss(
            student_slots,
            target_slots,
            target["activations"],
            inv_w=inv_w,
            var_w=var_w,
            cov_w=cov_w,
        )

        # Generate graph targets from the EMA branch. This prevents the online
        # binder and relation head from jointly moving their own supervision.
        target_centroids_t = target["centroids"][:, :-1].reshape(
            pair_batch, K, 3
        )
        target_centroids_tp1 = target["centroids"][:, 1:].reshape(
            pair_batch, K, 3
        )
        pseudo_labels = self._interaction_pseudo_labels(
            target_centroids_t,
            target_centroids_tp1,
            point_sequence[:, :-1].reshape(pair_batch, -1, 3),
            point_sequence[:, 1:].reshape(pair_batch, -1, 3),
        )
        off_diagonal = ~torch.eye(
            K, dtype=torch.bool, device=student_slots.device
        ).unsqueeze(0).expand(pair_batch, -1, -1)
        forward_interaction_loss = F.binary_cross_entropy_with_logits(
            interaction_logits[off_diagonal],
            pseudo_labels[off_diagonal].to(interaction_logits.dtype),
        )
        if self.relation_gating:
            reverse_interaction_loss = F.binary_cross_entropy_with_logits(
                reverse_interaction_logits[off_diagonal],
                pseudo_labels[off_diagonal].to(
                    reverse_interaction_logits.dtype
                ),
            )
            interaction_loss = 0.5 * (
                forward_interaction_loss + reverse_interaction_loss
            )
            forward_symmetric = 0.5 * (
                interaction_logits + interaction_logits.transpose(1, 2)
            )
            reverse_symmetric = 0.5 * (
                reverse_interaction_logits
                + reverse_interaction_logits.transpose(1, 2)
            )
            relation_consistency_loss = F.mse_loss(
                forward_symmetric.sigmoid()[off_diagonal],
                reverse_symmetric.sigmoid()[off_diagonal],
            )
        else:
            interaction_loss = forward_interaction_loss
            relation_consistency_loss = interaction_loss.new_zeros(())

        assignments = online["assignments"].clamp_min(1e-8)
        # Positive entropy is minimized. The sign printed in the paper would
        # instead reward uniform assignments.
        # ``assignments`` has shape [B, T, K, N] and is normalized over the
        # slot axis (K), hence ``dim=2`` is the slot-entropy axis.  Keep the
        # mean over frames/points so this regularizer is independent of the
        # number of sampled points.
        slot_entropy = -(
            assignments * assignments.log()
        ).sum(dim=2).mean()

        # Point-wise entropy and slot-load entropy play different roles.  The
        # former (above) keeps assignments sharp, but by itself it is minimized
        # by assigning every point to one slot.  Balance the *aggregate* load
        # across the batch/time window to prevent that degenerate solution.
        # We use a batch-level distribution rather than per-scene uniformity so
        # K remains an upper bound: scenes with fewer objects may still leave
        # some slots lightly used, while the dataset cannot permanently kill
        # the same slot indices.  Warm-up avoids forcing random early features
        # to be balanced before the visual teacher has stabilized.
        slot_balance, occupancy = self._slot_balance_loss(assignments)
        if self.slot_balance_warmup_epochs > 0:
            balance_scale = min(
                1.0,
                float(epoch + 1) / float(self.slot_balance_warmup_epochs),
            )
        else:
            balance_scale = 1.0

        loss = (
            self.lambda_ifdm * ifdm_loss
            + self.lambda_reverse * reverse_loss
            + slot_vicreg
            + self.lambda_interaction * interaction_loss
            + self.lambda_relation_consistency * relation_consistency_loss
            + self.lambda_slot_entropy * slot_entropy
            + self.lambda_slot_balance * balance_scale * slot_balance
        )
        self.update_teacher(epoch)
        return loss, {
            "loss_total": float(loss.item()),
            "ifdm_loss": float(ifdm_loss.item()),
            "reverse_ifdm_loss": float(reverse_loss.item()),
            "slot_inv_loss": float(inv_loss.item()),
            "slot_var_loss": float(var_loss.item()),
            "slot_cov_loss": float(cov_loss.item()),
            "interaction_loss": float(interaction_loss.item()),
            "relation_consistency_loss": float(
                relation_consistency_loss.item()
            ),
            "interaction_positive_rate": float(
                pseudo_labels[off_diagonal].float().mean().item()
            ),
            "relation_gate_strength": float(gate_strength),
            "slot_entropy": float(slot_entropy.item()),
            "slot_balance": float(slot_balance.item()),
            "slot_balance_scale": float(balance_scale),
            "slot_occupancy_entropy": float(
                (-(occupancy * occupancy.log()).sum()).item()
            ),
            "slot_effective_count": float(
                torch.exp(-(occupancy * occupancy.log()).sum()).item()
            ),
            "slot_activation_mean": float(
                online["activations"].mean().item()
            ),
            "frame_start": float(start),
        }

    # ========= Loss =========
    def compute_loss(self, batch, epoch):
        warm_scale = self._warmup_scale(epoch)
        var_w_now = self.vicreg_var_weight * warm_scale
        cov_w_now = self.vicreg_cov_weight * warm_scale
        inv_w_now = self.vicreg_inv_weight   

        nobs = self.normalizer.normalize(batch['obs'])
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        B, T, _, _ = nobs['point_cloud'].shape
        device = nobs['point_cloud'].device

        if self.use_dynaslots:
            return self._compute_dynaslots_loss(
                nobs,
                epoch,
                inv_w=inv_w_now,
                var_w=var_w_now,
                cov_w=cov_w_now,
            )

        interval = 2
        sampled_times = list(range(0, T, interval))
        M = len(sampled_times)

        nobs_sel = {k: v[:, sampled_times, ...] for k, v in nobs.items()}
        nobs_sel_flat = dict_apply(nobs_sel, lambda x: x.reshape(-1, *x.shape[2:]))

        feats_stu_flat = self.vis_encoder(nobs_sel_flat)  # [B*M, D]
        D = feats_stu_flat.shape[-1]
        feats_stu = feats_stu_flat.view(B, M, D)

        with torch.no_grad():
            feats_tea_flat = self.ema_vis_encoder(nobs_sel_flat)  # [B*M, D]
            feats_tea = feats_tea_flat.view(B, M, D)

        # Student
        f_t_stu   = feats_stu[:, :-1, :]               # [B, M-1, D]
        f_tp1_stu = feats_stu[:,  1:, :]
        # Teacher
        f_t_tea   = feats_tea[:, :-1, :]
        f_tp1_tea = feats_tea[:,  1:, :]

        delta_stu = f_tp1_stu - f_t_stu                 # [B, M-1, D]
        mem = self.idm_mlp(delta_stu)                   # [B, M-1, 16]

        BM = B * (M - 1)
        f_t_cond     = f_t_stu.reshape(BM, D)
        mem_flat     = mem.reshape(BM, self.latent_action_dim)  
        target_flat  = f_tp1_tea.reshape(BM, D)

        t_flat = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (BM,), device=device, dtype=torch.long
        )
        noise = torch.randn_like(target_flat)
        noisy_target = self.noise_scheduler.add_noise(target_flat, noise, t_flat)
        pred_flat = self.fdm_vis_decoder(noisy_target, t_flat, f_t_cond, mem_flat)  # [BM, D]

        long_term_vicreg_loss, inv_loss, var_loss, cov_loss = self._vicreg_loss(
            pred_flat, target_flat,
            inv_w=inv_w_now, var_w=var_w_now, cov_w=cov_w_now
        )
        long_term_loss = long_term_vicreg_loss

        # ===== inverse =====
        delta_rev = f_t_stu - f_tp1_stu                  # [B, M-1, D]
        mem_rev = self.idm_mlp(delta_rev)                # [B, M-1, 16]

        f_tp1_cond = f_tp1_stu.reshape(BM, D)
        mem_rev_flat = mem_rev.reshape(BM, self.latent_action_dim)
        target_rev_tea_flat = f_t_tea.reshape(BM, D).detach()
        t_rev_tea = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (BM,), device=device, dtype=torch.long
        )
        noise_rev_tea = torch.randn_like(target_rev_tea_flat)
        noisy_target_rev_tea = self.noise_scheduler.add_noise(target_rev_tea_flat, noise_rev_tea, t_rev_tea)
        pred_rev_tea_flat = self.fdm_vis_decoder(noisy_target_rev_tea, t_rev_tea, f_tp1_cond, mem_rev_flat)
        
        reverse_vicreg, reverse_inv_loss, reverse_var_loss, reverse_cov_loss = self._vicreg_loss(
            pred_rev_tea_flat, target_rev_tea_flat,
            inv_w=inv_w_now, var_w=var_w_now, cov_w=cov_w_now
        )

        loss = self.lambda_long_term * long_term_loss + self.lambda_reverse * reverse_vicreg
        
        self.update_teacher(epoch)

        return loss, {
            'loss_total': float(loss.item()),
            'inv_loss': float(inv_loss.item()),
            'var_loss': float(var_loss.item()),
            'cov_loss': float(cov_loss.item()),
            'reverse_mse': float(reverse_inv_loss.item()),
            'reverse_var_loss': float(reverse_var_loss.item()),
            'reverse_cov_loss': float(reverse_cov_loss.item()),
            'vicreg_inv_w_now': float(inv_w_now),
            'vicreg_var_w_now': float(var_w_now),
            'vicreg_cov_w_now': float(cov_w_now),
            'vicreg_warmup_scale': float(warm_scale),
        }
