from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops

from dynaslots.model.common.normalizer import LinearNormalizer
from dynaslots.policy.base_policy import BasePolicy
from dynaslots.model.diffusion.conditional_unet1d import ConditionalUnet1D
from dynaslots.model.diffusion.mask_generator import LowdimMaskGenerator
from dynaslots.common.pytorch_util import dict_apply
from dynaslots.common.model_util import print_params
from dynaslots.model.vision.pointnet_extractor import VisEncoder, StaEncoder
from dynaslots.model.vision.slot_binder import SlotBinder, SlotInteractionHead

class DynaSlotsPolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            task_whole_name=None,
            generator_config=None,
            checkpoint=None,
            use_dynaslots=False,
            num_slots=6,
            slot_iterations=3,
            use_slot_tokens=False,
            persistent_slots=False,
            persistent_memory_train_prob=0.5,
            persistent_memory_noise_std=0.05,
            persistent_memory_blend=0.8,
            persistent_memory_similarity_threshold=0.25,
            persistent_memory_temperature=0.1,
            persistent_memory_max_age=32,
            use_relation_tokens=False,
            relation_topk=3,
            relation_gate_floor=0.05,
            condition_token_dropout=0.0,
            condition_token_noise_std=0.0,
            use_centroid_tokens=False,
            use_motion_tokens=False,
            use_uncertainty_tokens=False,
            use_kinematic_tokens=False,
            kinematic_reliability_gate=False,
            token_qk_norm=False,
            bounded_token_film=False,
            bounded_token_film_bias=False,
            finetune_visual=False,
            visual_lr_scale=0.1,
            finetune_slot_binder=False,
            slot_balance_loss_weight=0.02,
            num_action_samples=1,
            executed_action_loss_weight=1.0,
            # parameters passed to step
            **kwargs):
        
        super().__init__()

        self.condition_type = condition_type
        self.task_name = task_whole_name
        self.use_dynaslots = bool(use_dynaslots)
        self.num_slots = int(num_slots)
        self.use_slot_tokens = bool(use_slot_tokens)
        self.persistent_slots = bool(persistent_slots)
        self.persistent_memory_train_prob = float(persistent_memory_train_prob)
        self.persistent_memory_noise_std = float(persistent_memory_noise_std)
        self.persistent_memory_blend = float(persistent_memory_blend)
        self.persistent_memory_similarity_threshold = float(
            persistent_memory_similarity_threshold
        )
        self.persistent_memory_temperature = float(
            persistent_memory_temperature
        )
        self.persistent_memory_max_age = int(persistent_memory_max_age)
        self.use_relation_tokens = bool(use_relation_tokens)
        self.relation_topk = int(relation_topk)
        self.relation_gate_floor = float(relation_gate_floor)
        self.condition_token_dropout = float(condition_token_dropout)
        self.condition_token_noise_std = float(condition_token_noise_std)
        self.use_centroid_tokens = bool(use_centroid_tokens)
        self.use_motion_tokens = bool(use_motion_tokens)
        self.use_uncertainty_tokens = bool(use_uncertainty_tokens)
        self.use_kinematic_tokens = bool(use_kinematic_tokens)
        self.kinematic_reliability_gate = bool(kinematic_reliability_gate)
        self.num_action_samples = int(num_action_samples)
        self.executed_action_loss_weight = float(
            executed_action_loss_weight
        )
        self.finetune_visual = bool(finetune_visual)
        self.visual_lr_scale = float(visual_lr_scale)
        self.finetune_slot_binder = bool(finetune_slot_binder)
        self.slot_balance_loss_weight = float(slot_balance_loss_weight)
        if self.visual_lr_scale <= 0.0:
            raise ValueError("visual_lr_scale must be positive")
        if self.num_action_samples < 1:
            raise ValueError("num_action_samples must be positive")
        if self.executed_action_loss_weight <= 0.0:
            raise ValueError("executed_action_loss_weight must be positive")
        if not 0.0 <= self.condition_token_dropout < 1.0:
            raise ValueError("condition_token_dropout must be in [0, 1)")
        if self.condition_token_noise_std < 0.0:
            raise ValueError("condition_token_noise_std must be non-negative")
        self._slot_memory = None
        self._slot_memory_age = 0
        self._last_slot_balance_loss = None
        cprint(f"task_whole_name: {task_whole_name}","yellow")

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        vis_encoder = VisEncoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                )
        
        sta_encoder = StaEncoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                )

        if self.use_dynaslots:
            if pointnet_type != "pointtransformer":
                raise ValueError(
                    "DynaSlots policy requires pointnet_type='pointtransformer'"
                )
            slot_binder = SlotBinder(
                slot_dim=vis_encoder.output_shape(),
                num_slots=self.num_slots,
                num_iterations=slot_iterations,
            )
        else:
            slot_binder = None

        # create diffusion model
        vis_feature_dim = vis_encoder.output_shape()
        if self.use_dynaslots and not self.use_slot_tokens:
            # Preserve object factorization for the policy instead of pooling
            # slots back into the scene-level vector DynaSlots is meant to fix.
            vis_feature_dim *= self.num_slots
        sta_feature_dim = sta_encoder.output_shape()
        obs_feature_dim = vis_feature_dim + sta_feature_dim
        self.obs_feature_dim = obs_feature_dim
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if self.use_slot_tokens:
                if not self.use_dynaslots:
                    raise ValueError("slot-token conditioning requires DynaSlots")
                if not self.condition_type.startswith(
                        "token_cross_attention"):
                    raise ValueError(
                        "slot-token conditioning requires a "
                        "token_cross_attention_* condition type"
                    )
                global_cond_dim = vis_encoder.output_shape()
            elif "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
            token_qk_norm=token_qk_norm,
            bounded_token_film=bounded_token_film,
            bounded_token_film_bias=bounded_token_film_bias,
        )

        self.vis_encoder = vis_encoder
        self.sta_encoder = sta_encoder
        self.slot_binder = slot_binder
        if self.use_slot_tokens:
            token_dim = vis_encoder.output_shape()
            self.state_token_proj = nn.Linear(sta_feature_dim, token_dim)
            self.slot_index_embedding = nn.Parameter(
                torch.zeros(1, 1, self.num_slots, token_dim)
            )
            self.observation_time_embedding = nn.Parameter(
                torch.zeros(1, n_obs_steps, 1, token_dim)
            )
            self.slot_type_embedding = nn.Parameter(
                torch.zeros(1, 1, 1, token_dim)
            )
            self.state_type_embedding = nn.Parameter(
                torch.zeros(1, 1, 1, token_dim)
            )
            self.condition_token_norm = nn.LayerNorm(token_dim)
            if self.use_centroid_tokens:
                self.centroid_token_proj = nn.Sequential(
                    nn.Linear(3, token_dim),
                    nn.SiLU(),
                    nn.Linear(token_dim, token_dim),
                )
                nn.init.zeros_(self.centroid_token_proj[-1].weight)
                nn.init.zeros_(self.centroid_token_proj[-1].bias)
            if self.use_motion_tokens:
                self.motion_token_proj = nn.Sequential(
                    nn.LayerNorm(token_dim),
                    nn.Linear(token_dim, token_dim),
                )
                nn.init.zeros_(self.motion_token_proj[-1].weight)
                nn.init.zeros_(self.motion_token_proj[-1].bias)
            if self.use_uncertainty_tokens:
                self.uncertainty_token_proj = nn.Sequential(
                    nn.Linear(2, token_dim),
                    nn.SiLU(),
                    nn.Linear(token_dim, token_dim),
                )
                nn.init.zeros_(self.uncertainty_token_proj[-1].weight)
                nn.init.zeros_(self.uncertainty_token_proj[-1].bias)
            if self.use_kinematic_tokens:
                # Encode translational velocity and speed in scene coordinates.
                # The final projection is zero initialized so the baseline is
                # recovered exactly at initialization.
                self.kinematic_token_proj = nn.Sequential(
                    nn.LayerNorm(4),
                    nn.Linear(4, token_dim),
                    nn.SiLU(),
                    nn.Linear(token_dim, token_dim),
                )
                nn.init.zeros_(self.kinematic_token_proj[-1].weight)
                nn.init.zeros_(self.kinematic_token_proj[-1].bias)
            nn.init.trunc_normal_(self.slot_index_embedding, std=0.02)
            nn.init.trunc_normal_(self.observation_time_embedding, std=0.02)
            nn.init.trunc_normal_(self.slot_type_embedding, std=0.02)
            nn.init.trunc_normal_(self.state_type_embedding, std=0.02)
            if self.use_relation_tokens:
                self.slot_idm_mlp = nn.Sequential(
                    nn.LayerNorm(token_dim),
                    nn.Linear(token_dim, 256),
                    nn.GELU(),
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Linear(128, 16),
                )
                self.interaction_head = SlotInteractionHead(
                    slot_dim=token_dim, action_dim=16
                )
                self.relation_token_scale = nn.Parameter(torch.tensor(0.1))
        elif self.use_relation_tokens:
            raise ValueError("relation tokens require slot-token conditioning")
        
        # load pretrained weights
        if self.pointnet_type == "clip" or self.pointnet_type == "dinov2" or self.pointnet_type == "pointnet++_pretrained" or self.pointnet_type == "pointempty":
            cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type is {self.pointnet_type}, don't load pretrained weights", "red")
        else:
            cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type is {self.pointnet_type}, load pretrained weights", "red")
            checkpoint = torch.load(checkpoint, map_location='cpu')
            vis_encoder_weights = {}
            for key, value in checkpoint['state_dicts']['model'].items():
                if 'vis_encoder' in key: 
                    if 'ema_vis_encoder' not in key and 'vis_encoder_teacher' not in key:
                        new_key = key.replace('vis_encoder.', '')
                        vis_encoder_weights[new_key] = value
            self.vis_encoder.load_state_dict(vis_encoder_weights)
            for param in self.vis_encoder.parameters():
                param.requires_grad = self.finetune_visual
            if self.use_dynaslots:
                slot_binder_weights = {}
                for key, value in checkpoint['state_dicts']['model'].items():
                    if key.startswith('slot_binder.'):
                        slot_binder_weights[
                            key.replace('slot_binder.', '', 1)
                        ] = value
                if not slot_binder_weights:
                    raise RuntimeError(
                        "DynaSlots checkpoint does not contain slot_binder weights"
                    )
                self.slot_binder.load_state_dict(slot_binder_weights)
                for param in self.slot_binder.parameters():
                    param.requires_grad = (
                        self.finetune_visual and self.finetune_slot_binder
                    )
                if self.use_relation_tokens:
                    for module_name, module in (
                        ('slot_idm_mlp', self.slot_idm_mlp),
                        ('interaction_head', self.interaction_head),
                    ):
                        module_weights = {
                            key.replace(f'{module_name}.', '', 1): value
                            for key, value in checkpoint[
                                'state_dicts'
                            ]['model'].items()
                            if key.startswith(f'{module_name}.')
                        }
                        if not module_weights:
                            raise RuntimeError(
                                f"RGP checkpoint does not contain {module_name}"
                            )
                        module.load_state_dict(module_weights)
                        for param in module.parameters():
                            param.requires_grad = False
            cprint(f"[DiffusionUnetHybridPointcloudPolicy] load pretrained weights successfully", "red")
            
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        print_params(self)

    def forward(self, batch):
        """DDP-compatible alias for the policy training objective."""
        return self.compute_loss(batch)

    def reset(self):
        self._slot_memory = None
        self._slot_memory_age = 0

    def _bind_visual_sequence(self, observations, batch_size, steps,
                              use_memory=False, simulate_memory=False,
                              return_outputs=False):
        tokens_flat = self.vis_encoder.forward_tokens(observations)
        points_flat = observations['point_cloud'][..., :3]
        tokens = tokens_flat.reshape(
            batch_size, steps, *tokens_flat.shape[1:]
        )
        points = points_flat.reshape(
            batch_size, steps, *points_flat.shape[1:]
        )

        memory_valid = (
            use_memory
            and self._slot_memory is not None
            and self._slot_memory.shape[0] == batch_size
            and self._slot_memory_age < self.persistent_memory_max_age
        )
        if memory_valid or simulate_memory:
            cold_start = self.slot_binder(
                tokens[:, 0], points[:, 0]
            )['slots']
            if memory_valid:
                candidate = self._slot_memory.to(
                    device=tokens.device, dtype=tokens.dtype
                )
                similarity = F.cosine_similarity(
                    candidate, cold_start, dim=-1
                ).mean(dim=-1, keepdim=True).unsqueeze(-1)
                temperature = max(self.persistent_memory_temperature, 1e-6)
                confidence = torch.sigmoid(
                    (
                        similarity
                        - self.persistent_memory_similarity_threshold
                    ) / temperature
                )
                blend = self.persistent_memory_blend * confidence
            else:
                candidate = cold_start.detach() + torch.randn_like(
                    cold_start
                ) * self.persistent_memory_noise_std
                keep = (
                    torch.rand(
                        batch_size, 1, 1, device=tokens.device
                    ) < self.persistent_memory_train_prob
                ).to(tokens.dtype)
                blend = self.persistent_memory_blend * keep
            previous = blend * candidate + (1.0 - blend) * cold_start
            frame_outputs = []
            for frame_index in range(steps):
                output = self.slot_binder(
                    tokens[:, frame_index],
                    points[:, frame_index],
                    previous_slots=previous,
                )
                previous = output['slots']
                frame_outputs.append(output)
            outputs = {
                key: torch.stack(
                    [output[key] for output in frame_outputs], dim=1
                )
                for key in frame_outputs[0]
            }
        else:
            outputs = self.slot_binder.forward_sequence(tokens, points)

        slots = outputs['slots']
        if use_memory:
            self._slot_memory = slots[:, -1].detach()
            self._slot_memory_age = (
                self._slot_memory_age + 1 if memory_valid else 1
            )
        return outputs if return_outputs else slots

    def _relation_contextualize(self, slots, centroids):
        batch_size, steps, num_slots, slot_dim = slots.shape
        latent_actions = slots.new_zeros(
            batch_size, steps, num_slots, 16
        )
        if steps > 1:
            forward_delta = slots[:, 1:] - slots[:, :-1]
            latent_actions[:, :-1] = self.slot_idm_mlp(forward_delta)
            latent_actions[:, 1:] = self.slot_idm_mlp(-forward_delta)

        flat_slots = slots.reshape(-1, num_slots, slot_dim)
        logits = self.interaction_head(
            flat_slots,
            centroids.reshape(-1, num_slots, 3),
            latent_actions.reshape(-1, num_slots, 16),
        )
        logits = 0.5 * (logits + logits.transpose(1, 2))
        gates = self.relation_gate_floor + (
            1.0 - self.relation_gate_floor
        ) * logits.sigmoid()
        diagonal = torch.eye(
            num_slots, dtype=torch.bool, device=slots.device
        ).unsqueeze(0)
        gates = torch.where(diagonal, torch.ones_like(gates), gates)
        if 0 < self.relation_topk < num_slots - 1:
            scores = gates.detach().masked_fill(diagonal, -1.0)
            indices = scores.topk(self.relation_topk, dim=-1).indices
            support = torch.zeros_like(gates, dtype=torch.bool)
            support.scatter_(-1, indices, True)
            support = support | support.transpose(1, 2) | diagonal
            gates = torch.where(
                support,
                gates,
                torch.full_like(gates, self.relation_gate_floor),
            )
        weights = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        messages = torch.bmm(weights, flat_slots).reshape_as(slots)
        return slots + torch.tanh(self.relation_token_scale) * (
            messages - slots
        )

    def _encode_visual_sequence(self, observations, batch_size, steps):
        """Encode a short observation sequence with temporal slot propagation."""
        if not self.use_dynaslots:
            return self.vis_encoder(observations)

        slots = self._bind_visual_sequence(
            observations, batch_size, steps, use_memory=False
        )
        return slots.reshape(batch_size * steps, -1)

    def _encode_condition_tokens(self, observations, batch_size, steps,
                                 use_memory=False, simulate_memory=False):
        slot_balance_loss_weight = float(
            getattr(self, "slot_balance_loss_weight", 0.0)
        )
        use_relation_tokens = bool(getattr(self, "use_relation_tokens", False))
        use_centroid_tokens = bool(getattr(self, "use_centroid_tokens", False))
        use_motion_tokens = bool(getattr(self, "use_motion_tokens", False))
        use_uncertainty_tokens = bool(
            getattr(self, "use_uncertainty_tokens", False)
        )
        use_kinematic_tokens = bool(getattr(self, "use_kinematic_tokens", False))
        kinematic_reliability_gate = bool(
            getattr(self, "kinematic_reliability_gate", False)
        )
        bound = self._bind_visual_sequence(
            observations,
            batch_size,
            steps,
            use_memory=use_memory,
            simulate_memory=simulate_memory,
            return_outputs=(
                use_relation_tokens
                or use_centroid_tokens
                or use_motion_tokens
                or use_uncertainty_tokens
                or use_kinematic_tokens
                or slot_balance_loss_weight > 0.0
            ),
        )
        self._last_slot_balance_loss = None
        if (
            slot_balance_loss_weight > 0.0
            and isinstance(bound, dict)
        ):
            assignments = bound['assignments'].clamp_min(1e-8)
            occupancy = assignments.mean(dim=-1).mean(dim=(0, 1))
            occupancy = occupancy.clamp_min(1e-8)
            self._last_slot_balance_loss = math.log(float(self.num_slots)) + (
                occupancy * occupancy.log()
            ).sum()
        if use_relation_tokens:
            slots = self._relation_contextualize(
                bound['slots'], bound['centroids']
            )
        elif use_centroid_tokens:
            slots = bound['slots']
        else:
            slots = bound['slots'] if isinstance(bound, dict) else bound
        slot_tokens = (
            slots
            + self.slot_index_embedding
            + self.observation_time_embedding[:, :steps]
            + self.slot_type_embedding
        )
        if use_centroid_tokens:
            slot_tokens = slot_tokens + self.centroid_token_proj(
                bound['centroids']
            )
        if use_motion_tokens:
            # A one-step slot derivative supplies object motion without
            # introducing persistent state or a learned pairwise graph.
            slot_velocity = torch.zeros_like(slots)
            if steps > 1:
                slot_velocity[:, 1:] = slots[:, 1:] - slots[:, :-1]
            slot_tokens = slot_tokens + self.motion_token_proj(slot_velocity)
        confidence_features = None
        if use_uncertainty_tokens or kinematic_reliability_gate:
            # Occupancy and concentration expose whether a slot is reliable.
            assignments = bound['assignments'].clamp_min(1e-8)
            occupancy = bound['activations']
            point_weights = assignments / assignments.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            entropy = -(
                point_weights * point_weights.log()
            ).sum(dim=-1)
            entropy = entropy / math.log(max(assignments.shape[-1], 2))
            confidence_features = torch.stack(
                [occupancy, (1.0 - entropy).clamp(0.0, 1.0)], dim=-1
            )
        if use_uncertainty_tokens:
            slot_tokens = slot_tokens + self.uncertainty_token_proj(
                confidence_features
            )
        if use_kinematic_tokens:
            # Centroid velocity is less sensitive to feature-space drift than
            # slot differences.  Reliability gating suppresses pseudo-motion
            # from diffuse or nearly empty assignments.
            centroids = bound['centroids']
            centroid_velocity = torch.zeros_like(centroids)
            if steps > 1:
                centroid_velocity[:, 1:] = (
                    centroids[:, 1:] - centroids[:, :-1]
                )
            speed = centroid_velocity.norm(dim=-1, keepdim=True)
            kinematic_features = torch.cat(
                [centroid_velocity, speed], dim=-1
            )
            if kinematic_reliability_gate:
                confidence = confidence_features.mean(dim=-1, keepdim=True)
                confidence = confidence.clamp(0.0, 1.0)
                kinematic_features = kinematic_features * confidence
            slot_tokens = slot_tokens + self.kinematic_token_proj(
                kinematic_features
            )
        state_tokens = self.sta_encoder(observations).reshape(
            batch_size, steps, -1
        )
        state_tokens = self.state_token_proj(state_tokens).unsqueeze(2)
        state_tokens = (
            state_tokens
            + self.observation_time_embedding[:, :steps]
            + self.state_type_embedding
        )
        tokens = torch.cat([slot_tokens, state_tokens], dim=2)
        tokens = self.condition_token_norm(tokens)
        return tokens.reshape(batch_size, -1, tokens.shape[-1])

    def _regularize_condition_tokens(self, tokens, steps):
        """Perturb slot tokens during BC training while preserving state tokens."""
        if not self.training or (
            self.condition_token_dropout <= 0.0
            and self.condition_token_noise_std <= 0.0
        ):
            return tokens
        tokens = tokens.reshape(
            tokens.shape[0], steps, self.num_slots + 1, tokens.shape[-1]
        )
        slot_tokens = tokens[:, :, :self.num_slots]
        if self.condition_token_noise_std > 0.0:
            slot_tokens = slot_tokens + torch.randn_like(slot_tokens) * (
                self.condition_token_noise_std
            )
        if self.condition_token_dropout > 0.0:
            keep = torch.rand(
                *slot_tokens.shape[:-1], 1,
                device=slot_tokens.device,
            ) >= self.condition_token_dropout
            slot_tokens = slot_tokens * keep.to(slot_tokens.dtype)
        tokens = torch.cat([slot_tokens, tokens[:, :, self.num_slots:]], dim=2)
        return tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory

    def _conditional_action_sample(
            self, condition_data, condition_mask,
            local_cond=None, global_cond=None, **kwargs):
        if self.num_action_samples == 1:
            return self.conditional_sample(
                condition_data,
                condition_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                **kwargs,
            )

        samples = self.num_action_samples
        batch_size = condition_data.shape[0]
        condition_data = condition_data.repeat_interleave(samples, dim=0)
        condition_mask = condition_mask.repeat_interleave(samples, dim=0)
        if local_cond is not None:
            local_cond = local_cond.repeat_interleave(samples, dim=0)
        if global_cond is not None:
            global_cond = global_cond.repeat_interleave(samples, dim=0)
        trajectories = self.conditional_sample(
            condition_data,
            condition_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **kwargs,
        )
        return trajectories.reshape(
            batch_size, samples, *trajectories.shape[1:]
        ).mean(dim=1)


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            if self.use_slot_tokens:
                global_cond = self._encode_condition_tokens(
                    this_nobs,
                    B,
                    To,
                    use_memory=self.persistent_slots,
                )
            else:
                vis_features = self._encode_visual_sequence(this_nobs, B, To)
                state_features = self.sta_encoder(this_nobs)
                nobs_features = torch.cat([vis_features, state_features], dim=-1)
            if self.use_slot_tokens:
                pass
            elif "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self._conditional_action_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _action_loss_weights(self, horizon, device, dtype):
        weights = torch.ones(horizon, device=device, dtype=dtype)
        if self.executed_action_loss_weight == 1.0:
            return weights
        start = self.n_obs_steps - 1
        end = min(start + self.n_action_steps, horizon)
        weights[start:end] = self.executed_action_loss_weight
        return weights / weights.mean()

    def compute_loss(self, batch):
        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            if self.use_slot_tokens:
                global_cond = self._encode_condition_tokens(
                    this_nobs,
                    batch_size,
                    self.n_obs_steps,
                    use_memory=False,
                    simulate_memory=self.persistent_slots,
                )
                global_cond = self._regularize_condition_tokens(
                    global_cond, self.n_obs_steps
                )
            else:
                vis_features = self._encode_visual_sequence(
                    this_nobs, batch_size, self.n_obs_steps
                )
                state_features = self.sta_encoder(this_nobs)
                nobs_features = torch.cat([vis_features, state_features], dim=-1)

            if self.use_slot_tokens:
                pass
            elif "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
    

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        if self.executed_action_loss_weight != 1.0:
            action_weights = self._action_loss_weights(
                horizon, loss.device, loss.dtype
            )
            loss = loss * action_weights.view(1, horizon, 1)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        bc_loss = loss.mean()
        loss = bc_loss

        # Keep visual fine-tuning from undoing the anti-collapse constraint
        # learned during DynaSlots pretraining. This is deliberately much
        # weaker than the pretraining term: behavior cloning remains the main
        # objective, and the regularizer only prevents permanent slot death.
        slot_balance_loss = self._last_slot_balance_loss
        if slot_balance_loss is not None and self.slot_balance_loss_weight > 0.0:
            # Policy ``compute_loss`` is intentionally epoch-agnostic in the
            # existing trainer, so use a constant weak coefficient here. The
            # pretraining branch has the explicit warm-up schedule.
            balance_scale = 1.0
            loss = loss + (
                self.slot_balance_loss_weight
                * balance_scale
                * slot_balance_loss
            )
        else:
            balance_scale = 0.0
        

        loss_dict = {
                'bc_loss': bc_loss.item(),
                'loss_total': loss.item(),
                'slot_balance_loss': (
                    float(slot_balance_loss.item())
                    if slot_balance_loss is not None else 0.0
                ),
                'slot_balance_scale': float(balance_scale),
            }
        
        return loss, loss_dict
