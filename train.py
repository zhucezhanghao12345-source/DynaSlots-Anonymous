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
from dynaslots.policy.dynaslots_policy import DynaSlotsPolicy
from dynaslots.dataset.base_dataset import BaseDataset
from dynaslots.env_runner.base_runner import BaseRunner
from dynaslots.common.checkpoint_util import TopKCheckpointManager
from dynaslots.common.pytorch_util import dict_apply, optimizer_to, _copy_to_cpu
from dynaslots.model.diffusion.ema_model import EMAModel
from dynaslots.model.common.lr_scheduler import get_scheduler

from dynaslots.common.addition import plot_history, calculate_average_metrics

OmegaConf.register_new_resolver("eval", eval, replace=True)

class DynaSlotsPolicyWorkspace:
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
        self.model: DynaSlotsPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DynaSlotsPolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        # Keep pretrained visual features frozen by default. For the explicit
        # visual-adaptation diagnostic, use a separate low-rate parameter group
        # so the diffusion decoder retains the configured base learning rate.
        finetune_visual = bool(cfg.policy.get("finetune_visual", False))
        if finetune_visual:
            visual_scale = float(cfg.policy.get("visual_lr_scale", 0.1))
            visual_params, base_params = [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("vis_encoder.") or (
                    name.startswith("slot_binder.")
                    and bool(cfg.policy.get("finetune_slot_binder", False))
                ):
                    visual_params.append(param)
                else:
                    base_params.append(param)
            base_group = dict(params=base_params)
            visual_group = dict(params=visual_params)
            base_lr = float(cfg.optimizer.get("lr", 1.0e-4))
            visual_group["lr"] = base_lr * visual_scale
            # Hydra's `instantiate(..., params=[...])` recursively converts
            # parameter groups into DictConfig objects. Construct the
            # configured optimizer class directly so AdamW receives native
            # Python dictionaries and tensors.
            optimizer_cls = hydra.utils.get_class(cfg.optimizer._target_)
            optimizer_kwargs = OmegaConf.to_container(
                cfg.optimizer, resolve=True
            )
            optimizer_kwargs.pop("_target_", None)
            optimizer_kwargs["params"] = [base_group, visual_group]
            self.optimizer = optimizer_cls(**optimizer_kwargs)
        else:
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
            # Keep model initialization synchronized via DDP while giving
            # every rank independent diffusion noise and augmentation RNG.
            torch.manual_seed(int(cfg.training.seed) + rank)
            np.random.seed(int(cfg.training.seed) + rank)
            random.seed(int(cfg.training.seed) + rank)
        else:
            rank = 0
            world_size = 1
            local_rank = 0
        is_main_process = rank == 0
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
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
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = None
        if is_main_process:
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
            cprint(f"Validation dataset_val size: {cfg.val_dataloader.batch_size}", "yellow")
            cprint(f"Validation dataset size: {cfg.dataloader.batch_size}", "yellow")

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        # When resuming from a checkpoint, cosine schedules must be expressed
        # over the full planned training length, not just the remaining epochs
        # in this run; otherwise the resumed run restarts the schedule instead
        # of continuing it.
        total_epochs = cfg.training.num_epochs
        if cfg.training.resume and self.epoch > 0:
            total_epochs = self.epoch + cfg.training.num_epochs
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * total_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseRunner
        env_runner = None
        if is_main_process:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)

        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        
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
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        train_model = self.model
        if distributed:
            train_model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        # save batch for sampling
        train_sampling_batch = None

        train_history = list()

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
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
                    raw_loss, loss_dict = train_model(batch)
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
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
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
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()

                    plot_loss.append(loss_dict)
                    
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if is_main_process:
                            wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            plot_loss_dict = calculate_average_metrics(plot_loss)
            train_history.append(plot_loss_dict)

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout
            if is_main_process and (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None and self.epoch!=0:
                t3 = time.time()
                # runner_log = env_runner.run(policy, dataset=dataset)
                runner_log = env_runner.run(policy)
                t4 = time.time()
                # print(f"rollout time: {t4-t3:.3f}")
                # log all
                step_log.update(runner_log)

            
                
            # run validation
            if is_main_process and (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if is_main_process and (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    
                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse

            if is_main_process and env_runner is None:
                step_log['test_mean_score'] = - train_loss
                
            # Save test_mean_score to log file
            if is_main_process and 'test_mean_score' in step_log:
                log_file_path = os.path.join(self.output_dir, 'test_mean_score.log')
                with open(log_file_path, 'a') as f:
                    top3 = step_log.get('SR_test_L3')
                    top3_suffix = f", Top-3 Mean: {top3}" if top3 is not None else ""
                    f.write(
                        f"Epoch {self.epoch}, Global Step {self.global_step}: "
                        f"{step_log['test_mean_score']}{top3_suffix}\n"
                    )
            
            # checkpoint
            if is_main_process and (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt and self.epoch!=0 :
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()
                plot_history(train_history, self.epoch, self.output_dir, self.cfg.training.seed)

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
            # ========= eval end for this epoch ==========
            policy.train()
            if distributed:
                # Non-zero ranks wait while rank zero performs the serial
                # MuJoCo rollout and checkpoint I/O.
                dist.barrier()
            if is_main_process:
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log
            
        if is_main_process:
            plot_history(train_history, self.epoch-1, self.output_dir, self.cfg.training.seed)
            print(f'Saved plots to {self.output_dir}')
        if distributed:
            dist.barrier()
            dist.destroy_process_group()

    def eval(self):
        # load the latest checkpoint
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy)
        
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
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
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
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

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'dynaslots', 'config'))
)
def main(cfg):
    workspace = DynaSlotsPolicyWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
