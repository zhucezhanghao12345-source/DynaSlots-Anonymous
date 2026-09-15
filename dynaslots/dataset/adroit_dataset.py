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

class AdroitDataset(BaseDataset):
    def __init__(self,
            zarr_path=None, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            point_dropout_ratio=0.0,
            ):
        super().__init__()
        self.task_name = task_name
        self.point_dropout_ratio = float(point_dropout_ratio)
        if not 0.0 <= self.point_dropout_ratio < 1.0:
            raise ValueError("point_dropout_ratio must be in [0, 1)")
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud', 'img'])
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
        val_set.point_dropout_ratio = 0.0
        return val_set

    def _dropout_point_cloud(self, point_cloud):
        if self.point_dropout_ratio == 0.0:
            return point_cloud
        point_cloud = point_cloud.copy()
        num_points = point_cloud.shape[-2]
        num_drop = int(round(num_points * self.point_dropout_ratio))
        if num_drop == 0:
            return point_cloud
        for frame in point_cloud:
            dropped = np.random.choice(num_points, num_drop, replace=False)
            kept = np.setdiff1d(
                np.arange(num_points), dropped, assume_unique=True
            )
            replacements = np.random.choice(kept, num_drop, replace=True)
            frame[dropped] = frame[replacements]
        return point_cloud

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
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        image = sample['img'][:,].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
                'image': image,
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        data['obs']['point_cloud'] = self._dropout_point_cloud(
            data['obs']['point_cloud']
        )
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
