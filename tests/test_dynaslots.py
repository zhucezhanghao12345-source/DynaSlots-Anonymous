import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from omegaconf import OmegaConf

from dynaslots.model.diffusion.conditional_unet1d import (
    ConditionalResidualBlock1D,
    ConditionalUnet1D,
    TokenCrossAttention,
)
from dynaslots.model.transformer.DiT1D import (
    DiTFeaturePredictor,
    RelationGatedGraphEncoder,
)
from dynaslots.model.vision.slot_binder import SlotBinder, SlotInteractionHead
from dynaslots.dataset.adroit_dataset import AdroitDataset
from dynaslots.model.vision.pointtransformer_v1.pointTransformer_v1_noKNN import (
    PurePointTransformerExtractor,
)
from dynaslots.policy.dynaslots import DynaSlots
from dynaslots.policy.dynaslots_policy import DynaSlotsPolicy


def test_slot_binder_competition_and_temporal_shapes():
    torch.manual_seed(0)
    binder = SlotBinder(slot_dim=12, num_slots=3, num_iterations=2)
    tokens = torch.randn(2, 3, 10, 12, requires_grad=True)
    coordinates = torch.randn(2, 3, 10, 3)

    output = binder.forward_sequence(tokens, coordinates)

    assert output["slots"].shape == (2, 3, 3, 12)
    assert output["centroids"].shape == (2, 3, 3, 3)
    assert output["activations"].shape == (2, 3, 3)
    assert output["assignments"].shape == (2, 3, 3, 10)
    torch.testing.assert_close(
        output["assignments"].sum(dim=2),
        torch.ones(2, 3, 10),
        atol=1e-6,
        rtol=1e-6,
    )

    output["slots"].square().mean().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_slot_balance_loss_prefers_dataset_level_uniform_usage():
    # The regularizer should be zero for uniform aggregate usage and positive
    # for a collapsed assignment, while preserving point-wise sharpness.
    uniform = torch.full((2, 3, 4, 8), 0.25)
    collapsed = torch.zeros_like(uniform)
    collapsed[:, :, 0] = 1.0
    uniform_loss, uniform_mass = DynaSlots._slot_balance_loss(uniform)
    collapsed_loss, collapsed_mass = DynaSlots._slot_balance_loss(collapsed)
    assert uniform_loss.item() < 1e-6
    assert collapsed_loss.item() > 1.0
    torch.testing.assert_close(uniform_mass, torch.full((4,), 0.25))
    torch.testing.assert_close(collapsed_mass, torch.tensor([1., 0., 0., 0.]))


def test_dit_supports_baseline_and_cross_slot_inputs():
    torch.manual_seed(1)
    baseline = DiTFeaturePredictor(
        input_dim=8, hidden_size=16, depth=1, num_heads=4
    )
    baseline_output = baseline(
        torch.randn(2, 8),
        torch.randint(0, 10, (2,)),
        torch.randn(2, 8),
        torch.randn(2, 16),
    )
    assert baseline_output.shape == (2, 8)

    slot_model = DiTFeaturePredictor(
        input_dim=8,
        hidden_size=16,
        depth=2,
        num_heads=4,
        num_tokens=3,
        graph_depth=1,
        param_type="eps",
    )
    noisy_slots = torch.randn(2, 3, 8, requires_grad=True)
    slot_output = slot_model(
        noisy_slots,
        torch.randint(0, 10, (2, 3)),
        torch.randn(2, 3, 8),
        torch.randn(2, 3, 16),
        centroids=torch.randn(2, 3, 3),
    )
    assert slot_output.shape == (2, 3, 8)
    slot_output.mean().backward()
    assert torch.isfinite(noisy_slots.grad).all()


def test_relation_gates_are_sparse_differentiable_and_keep_self_edges():
    torch.manual_seed(3)
    encoder = RelationGatedGraphEncoder(
        hidden_size=16,
        num_heads=4,
        depth=2,
        topk=2,
        gate_floor=0.05,
    )
    tokens = torch.randn(2, 4, 16, requires_grad=True)
    logits = torch.randn(2, 4, 4, requires_grad=True)
    gates = encoder.compute_gates(logits)
    torch.testing.assert_close(
        gates.diagonal(dim1=1, dim2=2), torch.ones(2, 4)
    )
    torch.testing.assert_close(gates, gates.transpose(1, 2))
    non_self_support = gates > 0.05 + 1e-6
    non_self_support &= ~torch.eye(4, dtype=torch.bool).unsqueeze(0)
    assert (non_self_support.sum(dim=-1) >= 2).all()

    output = encoder(tokens, logits)
    output.square().mean().backward()
    assert torch.isfinite(tokens.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_relation_gated_dit_uses_interaction_logits():
    torch.manual_seed(4)
    model = DiTFeaturePredictor(
        input_dim=8,
        hidden_size=16,
        depth=1,
        num_heads=4,
        num_tokens=4,
        graph_depth=2,
        relation_gating=True,
        relation_topk=2,
        param_type="eps",
    )
    relation_logits = torch.randn(2, 4, 4, requires_grad=True)
    output = model(
        torch.randn(2, 4, 8),
        torch.randint(0, 10, (2, 4)),
        torch.randn(2, 4, 8),
        torch.randn(2, 4, 16),
        centroids=torch.randn(2, 4, 3),
        relation_logits=relation_logits,
        relation_gate_strength=0.5,
    )
    assert output.shape == (2, 4, 8)
    output.mean().backward()
    assert relation_logits.grad is not None


def test_token_cross_attention_policy_decoder():
    torch.manual_seed(5)
    model = ConditionalUnet1D(
        input_dim=4,
        global_cond_dim=8,
        diffusion_step_embed_dim=16,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=8,
        condition_type="token_cross_attention_film",
    )
    sample = torch.randn(2, 16, 4, requires_grad=True)
    tokens = torch.randn(2, 14, 8, requires_grad=True)
    output = model(sample, torch.randint(0, 10, (2,)), global_cond=tokens)
    assert output.shape == sample.shape
    output.square().mean().backward()
    assert torch.isfinite(sample.grad).all()
    assert torch.isfinite(tokens.grad).all()


def test_token_film_is_neutral_at_initialization():
    torch.manual_seed(8)
    block = ConditionalResidualBlock1D(
        in_channels=8,
        out_channels=8,
        cond_dim=4,
        kernel_size=3,
        n_groups=4,
        condition_type="token_cross_attention_film",
    )
    x = torch.randn(2, 8, 6)
    cond = torch.randn(2, 5, 4)
    torch.testing.assert_close(block(x, cond), block(x, None))


def test_token_qk_norm_is_opt_in():
    baseline = TokenCrossAttention(8, 4, 16)
    normalized = TokenCrossAttention(8, 4, 16, qk_norm=True)
    assert isinstance(baseline.query_norm, torch.nn.Identity)
    assert isinstance(baseline.key_norm, torch.nn.Identity)
    assert isinstance(normalized.query_norm, torch.nn.LayerNorm)
    assert isinstance(normalized.key_norm, torch.nn.LayerNorm)


class _FixedTokenCondition(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, x, cond):
        return self.output.expand(x.shape[0], -1, -1)


def test_bounded_token_film_applies_tanh_to_scale():
    block = ConditionalResidualBlock1D(
        in_channels=2,
        out_channels=2,
        cond_dim=3,
        kernel_size=3,
        n_groups=1,
        condition_type="token_cross_attention_film",
        bounded_token_film=True,
    )
    block.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    block.residual_conv = torch.nn.Conv1d(2, 2, 1, bias=False)
    torch.nn.init.zeros_(block.residual_conv.weight)
    # Token FiLM output layout is [scale channels, bias channels].
    block.cond_encoder = _FixedTokenCondition(
        torch.tensor([[[10.0, 10.0, 0.0, 0.0]]])
    )
    x = torch.ones(1, 2, 1)
    output = block(x, torch.zeros(1, 1, 3))
    expected = torch.full_like(output, 1.0 + torch.tanh(torch.tensor(10.0)))
    torch.testing.assert_close(output, expected)


def test_bounded_token_film_bias_applies_tanh():
    block = ConditionalResidualBlock1D(
        in_channels=2,
        out_channels=2,
        cond_dim=3,
        kernel_size=3,
        n_groups=1,
        condition_type="token_cross_attention_film",
        bounded_token_film_bias=True,
    )
    block.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    block.residual_conv = torch.nn.Conv1d(2, 2, 1, bias=False)
    torch.nn.init.zeros_(block.residual_conv.weight)
    block.cond_encoder = _FixedTokenCondition(
        torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
    )
    x = torch.ones(1, 2, 1)
    output = block(x, torch.zeros(1, 1, 3))
    expected = torch.full_like(output, 1.0 + torch.tanh(torch.tensor(10.0)))
    torch.testing.assert_close(output, expected)


def test_action_sample_ensemble_repeats_and_averages_per_batch():
    policy = DynaSlotsPolicy.__new__(DynaSlotsPolicy)
    torch.nn.Module.__init__(policy)
    policy.num_action_samples = 2
    captured = {}

    def fake_sample(condition_data, condition_mask, **kwargs):
        captured["global_cond"] = kwargs["global_cond"].clone()
        offsets = torch.tensor([0.0, 2.0, 10.0, 14.0]).reshape(4, 1, 1)
        return condition_data + offsets

    policy.conditional_sample = fake_sample
    condition = torch.zeros(2, 3, 4)
    mask = torch.zeros_like(condition, dtype=torch.bool)
    global_cond = torch.tensor([[1.0], [2.0]])
    output = policy._conditional_action_sample(
        condition, mask, global_cond=global_cond
    )
    torch.testing.assert_close(output[0], torch.ones_like(output[0]))
    torch.testing.assert_close(output[1], torch.full_like(output[1], 12.0))
    torch.testing.assert_close(
        captured["global_cond"].flatten(),
        torch.tensor([1.0, 1.0, 2.0, 2.0]),
    )


def test_centroid_token_projection_is_neutral_and_trainable():
    policy = DynaSlotsPolicy.__new__(DynaSlotsPolicy)
    torch.nn.Module.__init__(policy)
    policy.use_relation_tokens = False
    policy.use_centroid_tokens = True
    policy.num_slots = 2
    policy.slot_index_embedding = torch.nn.Parameter(torch.zeros(1, 1, 2, 4))
    policy.observation_time_embedding = torch.nn.Parameter(torch.zeros(1, 2, 1, 4))
    policy.slot_type_embedding = torch.nn.Parameter(torch.zeros(1, 1, 1, 4))
    policy.state_type_embedding = torch.nn.Parameter(torch.zeros(1, 1, 1, 4))
    policy.state_token_proj = torch.nn.Linear(3, 4)
    policy.condition_token_norm = torch.nn.LayerNorm(4)
    policy.centroid_token_proj = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.SiLU(), torch.nn.Linear(4, 4)
    )
    torch.nn.init.zeros_(policy.centroid_token_proj[-1].weight)
    torch.nn.init.zeros_(policy.centroid_token_proj[-1].bias)

    slots = torch.randn(2, 2, 2, 4)
    centroids = torch.randn(2, 2, 2, 3, requires_grad=True)
    policy._bind_visual_sequence = lambda *args, **kwargs: {
        "slots": slots, "centroids": centroids
    }
    policy.sta_encoder = lambda observations: torch.zeros(4, 3)
    output = policy._encode_condition_tokens({}, 2, 2)
    assert output.shape == (2, 6, 4)
    output.square().mean().backward()
    assert policy.centroid_token_proj[-1].weight.grad is not None


class _IdentityTokenEncoder(torch.nn.Module):
    def forward_tokens(self, observations):
        return observations["point_cloud"]


def test_motion_and_uncertainty_token_residuals_are_differentiable():
    torch.manual_seed(11)
    policy = DynaSlotsPolicy.__new__(DynaSlotsPolicy)
    torch.nn.Module.__init__(policy)
    policy.use_relation_tokens = False
    policy.use_centroid_tokens = False
    policy.use_motion_tokens = True
    policy.use_uncertainty_tokens = True
    policy.num_slots = 2
    policy.motion_token_proj = torch.nn.Sequential(
        torch.nn.LayerNorm(4), torch.nn.Linear(4, 4)
    )
    policy.uncertainty_token_proj = torch.nn.Sequential(
        torch.nn.Linear(2, 4), torch.nn.SiLU(), torch.nn.Linear(4, 4)
    )
    slots = torch.randn(2, 2, 2, 4, requires_grad=True)
    assignments = torch.softmax(torch.randn(2, 2, 2, 6), dim=2)
    activations = (assignments.mean(dim=-1) * 2).clamp(0.0, 1.0)
    point_weights = assignments / assignments.sum(dim=-1, keepdim=True)
    entropy = -(point_weights * point_weights.clamp_min(1e-8).log()).sum(-1)
    entropy = entropy / torch.log(torch.tensor(6.0))
    features = torch.stack([activations, 1.0 - entropy], dim=-1)
    output = slots + policy.motion_token_proj(
        torch.cat([torch.zeros_like(slots[:, :1]), slots[:, 1:] - slots[:, :-1]], dim=1)
    ) + policy.uncertainty_token_proj(features)
    assert output.shape == slots.shape
    output.square().mean().backward()
    assert slots.grad is not None and torch.isfinite(slots.grad).all()


def test_persistent_slot_memory_is_updated_and_reset():
    torch.manual_seed(6)
    policy = DynaSlotsPolicy.__new__(DynaSlotsPolicy)
    torch.nn.Module.__init__(policy)
    policy.vis_encoder = _IdentityTokenEncoder()
    policy.slot_binder = SlotBinder(
        slot_dim=3, num_slots=2, num_iterations=1
    )
    policy._slot_memory = None
    policy._slot_memory_age = 0
    policy.persistent_memory_max_age = 32
    policy.persistent_memory_train_prob = 0.5
    policy.persistent_memory_noise_std = 0.05
    policy.persistent_memory_blend = 0.8
    policy.persistent_memory_similarity_threshold = 0.25
    policy.persistent_memory_temperature = 0.1

    observations = {"point_cloud": torch.randn(4, 6, 3)}
    first = policy._bind_visual_sequence(
        observations, batch_size=2, steps=2, use_memory=True
    )
    assert first.shape == (2, 2, 2, 3)
    torch.testing.assert_close(policy._slot_memory, first[:, -1])

    second = policy._bind_visual_sequence(
        observations, batch_size=2, steps=2, use_memory=True
    )
    assert second.shape == first.shape
    torch.testing.assert_close(policy._slot_memory, second[:, -1])
    policy.reset()
    assert policy._slot_memory is None
    assert policy._slot_memory_age == 0


def test_training_memory_does_not_persist_between_batches():
    torch.manual_seed(7)
    policy = DynaSlotsPolicy.__new__(DynaSlotsPolicy)
    torch.nn.Module.__init__(policy)
    policy.vis_encoder = _IdentityTokenEncoder()
    policy.slot_binder = SlotBinder(
        slot_dim=3, num_slots=2, num_iterations=1
    )
    policy._slot_memory = None
    policy._slot_memory_age = 0
    policy.persistent_memory_max_age = 32
    policy.persistent_memory_train_prob = 1.0
    policy.persistent_memory_noise_std = 0.05
    policy.persistent_memory_blend = 0.8
    policy.persistent_memory_similarity_threshold = 0.25
    policy.persistent_memory_temperature = 0.1

    observations = {"point_cloud": torch.randn(4, 6, 3)}
    output = policy._bind_visual_sequence(
        observations,
        batch_size=2,
        steps=2,
        simulate_memory=True,
    )
    assert output.shape == (2, 2, 2, 3)
    assert policy._slot_memory is None
    assert policy._slot_memory_age == 0


def test_relation_contextualization_is_residual_and_differentiable():
    torch.manual_seed(9)
    policy = DynaSlotsPolicy.__new__(DynaSlotsPolicy)
    torch.nn.Module.__init__(policy)
    policy.slot_idm_mlp = torch.nn.Sequential(
        torch.nn.LayerNorm(4),
        torch.nn.Linear(4, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 16),
    )
    policy.interaction_head = SlotInteractionHead(slot_dim=4, action_dim=16)
    policy.relation_token_scale = torch.nn.Parameter(torch.tensor(0.1))
    policy.relation_gate_floor = 0.05
    policy.relation_topk = 2
    slots = torch.randn(2, 2, 4, 4, requires_grad=True)
    centroids = torch.randn(2, 2, 4, 3)
    output = policy._relation_contextualize(slots, centroids)
    assert output.shape == slots.shape
    output.square().mean().backward()
    assert slots.grad is not None and torch.isfinite(slots.grad).all()
    assert policy.relation_token_scale.grad is not None


def test_point_transformer_exposes_projected_tokens():
    encoder = PurePointTransformerExtractor(
        in_channels=3,
        out_channels=8,
        embed_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    point_cloud = torch.randn(2, 12, 3)
    assert encoder.forward_tokens(point_cloud).shape == (2, 12, 8)
    assert encoder(point_cloud).shape == (2, 8)


def test_executed_action_loss_weights_target_consumed_chunk():
    policy = DynaSlotsPolicy.__new__(DynaSlotsPolicy)
    policy.executed_action_loss_weight = 2.0
    policy.n_obs_steps = 2
    policy.n_action_steps = 8

    weights = policy._action_loss_weights(16, torch.device("cpu"), torch.float32)

    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    assert torch.all(weights[1:9] > weights[0])
    torch.testing.assert_close(weights[0], weights[9])


def test_point_dropout_resamples_without_changing_shape_or_range():
    np.random.seed(0)
    dataset = AdroitDataset.__new__(AdroitDataset)
    dataset.point_dropout_ratio = 0.25
    point_cloud = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)

    augmented = dataset._dropout_point_cloud(point_cloud)

    assert augmented.shape == point_cloud.shape
    assert not np.shares_memory(augmented, point_cloud)
    for original_frame, augmented_frame in zip(point_cloud, augmented):
        original_points = {tuple(point) for point in original_frame}
        assert all(tuple(point) in original_points for point in augmented_frame)
        assert len({tuple(point) for point in augmented_frame}) < len(original_points)


def test_dynaslots_loss_is_finite_and_differentiable():
    torch.manual_seed(2)
    scheduler = DDIMScheduler(
        num_train_timesteps=10,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="sample",
    )
    encoder_config = OmegaConf.create({
        "in_channels": 3,
        "out_channels": 8,
        "embed_dim": 16,
        "num_layers": 1,
        "num_heads": 4,
        "dropout": 0.0,
        "final_norm": "layernorm",
    })
    model = DynaSlots(
        shape_meta=OmegaConf.create({
            "obs": {"point_cloud": {"shape": [12, 3]}},
            "action": {"shape": [2]},
        }),
        noise_scheduler=scheduler,
        horizon=3,
        encoder_output_dim=8,
        pointnet_type="pointtransformer",
        pointcloud_encoder_cfg=encoder_config,
        use_dynaslots=True,
        num_slots=3,
        slot_iterations=2,
        frame_stride=1,
        slot_graph_depth=1,
        relation_gating=True,
        relation_topk=2,
        relation_gate_warmup_epochs=10,
        fdm_d_model=16,
        vicreg_inv_weight=1.0,
        vicreg_var_weight=1.0,
        vicreg_cov_weight=0.04,
    )
    loss, metrics = model._compute_dynaslots_loss(
        {"point_cloud": torch.randn(2, 3, 12, 3)},
        epoch=0,
        inv_w=1.0,
        var_w=1.0,
        cov_w=0.04,
    )

    assert torch.isfinite(loss)
    assert set([
        "ifdm_loss",
        "reverse_ifdm_loss",
        "slot_inv_loss",
        "interaction_loss",
        "slot_entropy",
        "relation_gate_strength",
    ]).issubset(metrics)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
