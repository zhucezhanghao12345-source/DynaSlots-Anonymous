# if __name__ == "__main__":
#     import sys
#     import os
#     import pathlib

#     ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
#     sys.path.append(ROOT_DIR)
#     os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.distributed as dist
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
from dynaslots.policy.dynaslots import DynaSlots
from dynaslots.dataset.base_dataset import BaseDataset
from dynaslots.common.checkpoint_util import TopKCheckpointManager
from dynaslots.common.pytorch_util import dict_apply, optimizer_to, _copy_to_cpu
from dynaslots.model.common.lr_scheduler import get_scheduler
import matplotlib.pyplot as plt

from dynaslots.common.addition import plot_history, calculate_average_metrics

OmegaConf.register_new_resolver("eval", eval, replace=True)

class DynaSlotsPretrainWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DynaSlots = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
        if distributed:
            dist.init_process_group(backend='nccl')
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
        else:
            rank = 0
            world_size = 1
            local_rank = 0
        is_main_process = rank == 0
        resumed_from_checkpoint = False
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path('pretrain_latest')
            if lastest_ckpt_path.is_file():
                if is_main_process:
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                resumed_from_checkpoint = True
                # Periodic checkpoints are written after completing
                # ``self.epoch`` and before its in-memory increment.
                if self.epoch < cfg.training.num_epochs:
                    self.epoch += 1

        # configure dataset
        dataset: BaseDataset
        print(f"Dataset: {cfg.task.dataset}")
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_loader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        train_sampler = None
        if distributed:
            train_sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=bool(train_loader_cfg.pop('shuffle', True)),
                seed=int(cfg.training.seed),
                drop_last=False,
            )
        train_dataloader = DataLoader(
            dataset, sampler=train_sampler, **train_loader_cfg
        )
        if resumed_from_checkpoint:
            # Re-express progress in optimizer steps for the current world
            # size. A single-GPU checkpoint otherwise appears 4x farther
            # through a cosine schedule after switching to four GPUs.
            self.global_step = self.epoch * len(train_dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = None
        if is_main_process:
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
            cprint(f"Validation dataset_val size: {cfg.val_dataloader.batch_size}", "yellow")
            cprint(f"Validation dataset size: {cfg.dataloader.batch_size}", "yellow")

        self.model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = None
        if is_main_process:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(f'cuda:{local_rank}' if distributed else cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)
        train_model = self.model
        if distributed:
            train_model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                # The baseline IDM remains intentionally unused by the
                # DynaSlots branch.
                find_unused_parameters=True,
            )

        # save batch for sampling
        train_sampling_batch = None

        train_history = list()

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(local_epoch_idx)
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            plot_loss = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                    leave=False, mininterval=cfg.training.tqdm_interval_sec,
                    disable=not is_main_process) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = train_model(batch, local_epoch_idx)

                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        if cfg.training.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                float(cfg.training.max_grad_norm),
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    if distributed:
                        metric_names = sorted(loss_dict)
                        metric_values = torch.tensor(
                            [raw_loss.detach().item()] + [loss_dict[k] for k in metric_names],
                            dtype=torch.float64,
                            device=device,
                        )
                        dist.all_reduce(metric_values, op=dist.ReduceOp.SUM)
                        metric_values /= world_size
                        raw_loss_cpu = metric_values[0].item()
                        loss_dict = {
                            key: metric_values[index + 1].item()
                            for index, key in enumerate(metric_names)
                        }
                    else:
                        raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    step_log.update(loss_dict)

                    plot_loss.append(loss_dict)

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if is_main_process:
                            wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break
               
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss 
                
            plot_loss_dict = calculate_average_metrics(plot_loss)
            train_history.append(plot_loss_dict)

            # checkpoint
            if is_main_process and (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint(tag='pretrain_latest')
                plot_history(train_history, self.epoch, self.output_dir, self.cfg.training.seed)

            if is_main_process:
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

        if is_main_process:
            plot_history(train_history, self.epoch-1, self.output_dir, self.cfg.training.seed)
            self.save_checkpoint(tag='pretrain_latest')
            print(f'Saved plots to {self.output_dir}')
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        if tag in ('latest', 'pretrain_latest'):
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")         
            
    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'dynaslots', 'config'))
)
def main(cfg):
    workspace = DynaSlotsPretrainWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
