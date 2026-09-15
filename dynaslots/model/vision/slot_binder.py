import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotBinder(nn.Module):
    """Bind point tokens to a temporally consistent set of object slots.

    The assignment first normalizes over slots so that slots compete for each
    point. A second normalization over points is used only for aggregation.
    This resolves an ambiguity in the paper's Eq. 1 and prevents every slot
    from independently binding the same object.
    """

    def __init__(self, slot_dim, num_slots=6, num_iterations=3, eps=1e-8):
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.num_iterations = int(num_iterations)
        self.eps = float(eps)

        self.initial_slots = nn.Parameter(
            torch.randn(1, self.num_slots, self.slot_dim) /
            self.slot_dim ** 0.5
        )
        self.token_norm = nn.LayerNorm(self.slot_dim)
        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.refine_norm = nn.LayerNorm(self.slot_dim)
        self.refine_mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.slot_dim * 2),
            nn.GELU(),
            nn.Linear(self.slot_dim * 2, self.slot_dim),
        )
        self.scale = self.slot_dim ** -0.5

    def forward(self, tokens, coordinates, previous_slots=None):
        """
        Args:
            tokens: [B, N, D] point features.
            coordinates: [B, N, 3] coordinates aligned with ``tokens``.
            previous_slots: optional [B, K, D] propagated slot state.
        Returns:
            Dictionary containing slots, centroids, activations and the
            point-to-slot assignment probabilities [B, K, N].
        """
        if tokens.ndim != 3 or coordinates.ndim != 3:
            raise ValueError("tokens and coordinates must be rank-3 tensors")
        if tokens.shape[:2] != coordinates.shape[:2]:
            raise ValueError("tokens and coordinates must share B and N")
        if tokens.shape[-1] != self.slot_dim:
            raise ValueError(
                f"expected token dim {self.slot_dim}, got {tokens.shape[-1]}"
            )

        batch_size = tokens.shape[0]
        if previous_slots is None:
            slots = self.initial_slots.expand(batch_size, -1, -1)
        else:
            if previous_slots.shape != (
                    batch_size, self.num_slots, self.slot_dim):
                raise ValueError(
                    "previous_slots must have shape "
                    f"[B, {self.num_slots}, {self.slot_dim}]"
                )
            slots = previous_slots

        normalized_tokens = self.token_norm(tokens)
        keys = self.to_k(normalized_tokens)
        values = self.to_v(normalized_tokens)

        assignment = None
        aggregate_weights = None
        for _ in range(self.num_iterations):
            queries = self.to_q(self.slot_norm(slots)) * self.scale
            logits = torch.einsum("bkd,bnd->bkn", queries, keys)

            # Each point chooses among slots. This is the competition missing
            # when Eq. 1 is interpreted as a softmax over points.
            assignment = F.softmax(logits, dim=1)

            # Normalize over points only for the weighted aggregation.
            aggregate_weights = assignment + self.eps
            aggregate_weights = aggregate_weights / aggregate_weights.sum(
                dim=-1, keepdim=True
            )
            updates = torch.einsum(
                "bkn,bnd->bkd", aggregate_weights, values
            )
            slots = slots + updates
            slots = slots + self.refine_mlp(self.refine_norm(slots))

        centroids = torch.einsum(
            "bkn,bnc->bkc", aggregate_weights, coordinates[..., :3]
        )
        # Relative assignment mass is a non-parametric occupancy estimate. It
        # cannot collapse all dynamics losses by learning activations of zero.
        activations = (assignment.mean(dim=-1) * self.num_slots).clamp(0.0, 1.0)
        return {
            "slots": slots,
            "centroids": centroids,
            "activations": activations,
            "assignments": assignment,
        }

    def forward_sequence(self, tokens, coordinates, initial_slots=None):
        """Propagate slots through a token sequence.

        Args:
            tokens: [B, T, N, D]
            coordinates: [B, T, N, 3]
        """
        if tokens.ndim != 4 or coordinates.ndim != 4:
            raise ValueError("sequence inputs must be rank-4 tensors")

        previous_slots = initial_slots
        outputs = []
        for frame_index in range(tokens.shape[1]):
            frame_output = self(
                tokens[:, frame_index],
                coordinates[:, frame_index],
                previous_slots=previous_slots,
            )
            previous_slots = frame_output["slots"]
            outputs.append(frame_output)

        return {
            key: torch.stack([output[key] for output in outputs], dim=1)
            for key in outputs[0]
        }


class SlotInteractionHead(nn.Module):
    """Predict pairwise contact/co-motion relations between slots.

    Kinematic pseudo-labels cannot identify causal direction, so the output is
    intentionally described as a relation graph rather than a causal graph.
    """

    def __init__(self, slot_dim, action_dim=16, hidden_dim=128):
        super().__init__()
        pair_dim = 2 * slot_dim + 2 * action_dim + 6
        self.network = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, slots, centroids, latent_actions):
        if slots.ndim != 3:
            raise ValueError("slots must have shape [B, K, D]")
        num_slots = slots.shape[1]

        def pairwise(tensor):
            left = tensor.unsqueeze(2).expand(-1, -1, num_slots, -1)
            right = tensor.unsqueeze(1).expand(-1, num_slots, -1, -1)
            return left, right

        slot_i, slot_j = pairwise(slots)
        centroid_i, centroid_j = pairwise(centroids)
        action_i, action_j = pairwise(latent_actions)
        pair_features = torch.cat(
            [slot_i, slot_j, centroid_i, centroid_j, action_i, action_j],
            dim=-1,
        )
        return self.network(pair_features).squeeze(-1)
