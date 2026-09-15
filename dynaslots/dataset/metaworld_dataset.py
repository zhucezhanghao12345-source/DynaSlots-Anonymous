from typing import Dict
import torch
import numpy as np
import copy
from dynaslots.common.pytorch_util import dict_apply
from dynaslots.common.replay_buffer import ReplayBuffer
from dynaslots.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from dynaslots.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from dynaslots.dataset.base_dataset import BaseDataset

class MetaworldDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            action_clip=None,
            ):
        super().__init__()
        self.zarr_path = zarr_path
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud','img'])
        self._remove_zero_padding()
        self._clip_actions(action_clip)
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def _remove_zero_padding(self):
        """Replace absent-point zero padding with deterministic real points.

        MetaWorld clouds are stored at a fixed length of 1024, but some tasks
        (especially ``sweep``) contain substantially fewer cropped points and
        were padded with all-zero rows.  Those rows are not masked by the
        PointTransformer/SlotBinder and therefore become artificial geometry.
        We clean the in-memory replay buffer once, before fitting the
        normalizer, by repeating valid points.  This keeps training and online
        FPS sampling on the same support without changing the source zarr.
        """
        points = self.replay_buffer['point_cloud']
        if points.ndim != 3 or points.shape[-1] < 3:
            return
        xyz = points[..., :3]
        valid = np.any(np.abs(xyz) > 1e-6, axis=-1)
        invalid_count = int((~valid).sum())
        if invalid_count == 0:
            return
        for frame_idx in range(points.shape[0]):
            bad = np.flatnonzero(~valid[frame_idx])
            if bad.size == 0:
                continue
            good = np.flatnonzero(valid[frame_idx])
            if good.size == 0:
                continue
            points[frame_idx, bad] = points[
                frame_idx, good[np.arange(bad.size) % good.size]
            ]
        print(
            f"[MetaworldDataset] replaced {invalid_count} zero-padded "
            f"points in {self.zarr_path}"
        )

    def _clip_actions(self, action_clip):
        """Align recorded targets with the environment action contract.

        Some MetaWorld expert policies emit a proportional-control response
        before the environment clips it to ``[-1, 1]``.  Keeping those raw
        responses in the BC target makes the normalizer fit values that can
        never be executed.  This option is deliberately opt-in and operates
        only on the in-memory replay buffer.
        """
        if action_clip is None:
            return
        if len(action_clip) != 2:
            raise ValueError(f"action_clip must be [low, high], got {action_clip}")
        low, high = (float(action_clip[0]), float(action_clip[1]))
        if low >= high:
            raise ValueError(f"action_clip low must be below high, got {action_clip}")
        actions = self.replay_buffer['action']
        before = actions.copy()
        np.clip(actions, low, high, out=actions)
        changed = int(np.count_nonzero(before != actions))
        if changed:
            print(
                f"[MetaworldDataset] clipped {changed} action values to "
                f"[{low:g}, {high:g}] in memory for {self.zarr_path}"
            )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
            'image': self.replay_buffer['img'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)
        image = sample['img'][:,].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud, 
                'agent_pos': agent_pos, 
                'image': image, 
            },
            'action': sample['action'].astype(np.float32)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
