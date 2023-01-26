'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: PPT训练流管理

example:

'''

import torch
import torch.backends.cudnn as cudnn

import torch.distributed as dist

try:
    import apex
    from apex import amp
except:
    pass
    
from loguru import logger
import datetime
import os
import time
import shutil

from utils.common import reduce_mean, remove_file
from utils.standard_tools import togpu
from model.utils.ops import clip_grads

class Trainer:
    def __init__(self, config, args, Model, Data_loader, step_update=False):
        self.is_main_process = True if  args.rank== 0 else False

        self.step_update = step_update
        self.Model = Model
        self.Dataloader = Data_loader
        
        self.config = config
        self.args = args
        self.is_distributed = args.distributed
        
        self.max_epoch = config['epoch']
        self.amp_training = args.fp16

        self.data_type = torch.float16 if args.fp16 else torch.float32

        self.file_name = os.path.join(args.output_dir, f'{datetime.date.today()}', config['experiment_name'],  config['config_name'])
        if self.is_main_process:
            logger.info("Trianing save folder: {}".format(self.file_name))
        self.best_score = None
        if not os.path.exists(self.file_name):
            os.makedirs(self.file_name)
        shutil.copyfile(args.exp_file, os.path.join(self.file_name, os.path.basename(args.exp_file)))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.visual_loss = 0.
        self.epoch_loss = 0.

        self.train_sampler = None
        self.eval_sampler = None

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()
            
    def before_train(self):
        torch.cuda.set_device(self.args.rank)
        self.model = self.Model.get_model()
        self.model.cuda(self.args.rank)
        
        if self.is_distributed:
            if not self.amp_training:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.args.rank)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                        device_ids=[self.args.rank],
                                                                        find_unused_parameters=True)
            else:
                self.model =  apex.parallel.convert_syncbn_model(self.model).to(self.args.rank)
                self.model = apex.parallel.DistributedDataParallel(self.model)        

        if self.is_main_process:
            logger.add(os.path.join(self.file_name, 'training.log'))
            logger.info("args: {}".format(self.args))
            logger.info("config:\n{}".format(self.config))

        self.optimizer = self.Model.get_optimizer(self.model)

        if self.amp_training:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

        if self.is_main_process:
            logger.info("init train loader...")
        if not  self.is_distributed:
            self.train_loader = self.Dataloader.get_train()
        else:
            self.train_loader, self.train_sampler = self.Dataloader.get_train(distributed=self.is_distributed, nprocs=self.args.nprocs)

        if self.is_main_process:
            logger.info("init eval loader...")
        if not  self.is_distributed:
            self.eval_loader = self.Dataloader.get_test()
        else:
            self.eval_loader, self.eval_sampler = self.Dataloader.get_test(distributed=self.is_distributed, nprocs=self.args.nprocs)

        if not self.step_update:
            self.max_iter = self.max_epoch
        else:
            self.max_iter = len(self.train_loader)*self.max_epoch
        
        if self.is_main_process:
            logger.info("max_iter: {}".format(self.max_iter))

        if isinstance(self.optimizer, list):
            self.lr_scheduler = []
            for sub_optimizer in self.optimizer:
                self.lr_scheduler.append(self.Model.get_lr_scheduler(sub_optimizer, self.max_iter))
        else:
            self.lr_scheduler = self.Model.get_lr_scheduler(self.optimizer, self.max_iter)
        self.loss_function = self.Model.get_loss_func().cuda(self.args.rank)

        if self.args.pretrained is not None:
            logger.info('use pretrain model {}'.format(self.args.pretrained))
            self.pretrained_init_weight()
            self.start_epoch = 0
        else:
            self.resume_train()

        self.model.train()
        
        self.iter = 0
        if self.is_main_process:
            logger.info("Training start...")

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()
            
    def before_epoch(self):
        if self.is_main_process:
            logger.info("---> start train epoch{}".format(self.epoch + 1))
        self.model.train()
        self.visual_loss = 0.
        self.epoch_loss = 0.

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        if self.eval_sampler is not None:
            self.eval_sampler.set_epoch(self.epoch)

    def train_in_iter(self):
        self.iter = 0
        self.epoch_iter = len(self.train_loader)
        self.start_time = time.time()
        for inps, targets in self.train_loader:
            self.before_iter()
            self.train_one_iter(inps, targets)
            self.after_iter()

    def before_iter(self):
        self.iter += 1
            
    def train_one_iter(self, inps, targets):
        if not isinstance(inps ,dict):
            inps = inps.cuda(non_blocking=True)
            inps.requires_grad = False
        elif isinstance(inps ,dict):
            inps = togpu(inps)

        if not isinstance(targets ,dict) and not isinstance(targets, list):
            targets = targets.cuda(non_blocking=True)
            targets.requires_grad = False
        elif isinstance(targets ,dict):
            targets = togpu(targets)

        logic = self.model.forward(inps)
        loss_outputs = self.loss_function(logic, targets)

        loss = loss_outputs

        if isinstance(self.optimizer, list):
            for sub_optimizer in self.optimizer:
                sub_optimizer.zero_grad()
        else: 
            self.optimizer.zero_grad()

        if self.is_distributed:
            dist.barrier()
            iter_loss= reduce_mean(loss, self.args.nprocs)
            self.visual_loss += iter_loss.item()
            self.epoch_loss += iter_loss.item()
        else:
            self.visual_loss += loss.item()
            self.epoch_loss += loss.item()

        if self.amp_training:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if torch.isfinite(loss).item():
            _ = clip_grads(self.model.parameters())
        else:
            NotImplementedError("loss type error!can't backward!")

        if isinstance(self.optimizer, list):
            for sub_optimizer in self.optimizer:
                sub_optimizer.step()
        else:
            self.optimizer.step()

    def after_iter(self):
        if isinstance(self.lr_scheduler, list):
            for sub_lr_scheduler  in self.lr_scheduler:
                if sub_lr_scheduler is not None and self.step_update:
                    sub_lr_scheduler.step()
        else:
            if self.lr_scheduler is not None and self.step_update:
                self.lr_scheduler.step()
            
        if self.iter % self.args.visual_batch_interval == 0 and self.is_main_process:
            interval_use_time = time.time() - self.start_time
            self.start_time = time.time()
            left_time = interval_use_time*((self.epoch_iter - self.iter) / self.args.visual_batch_interval)
            left_minut = left_time/60.0
            left_hours =  left_minut/60.0

            logger.info(
                "Epoch {}/{} batch {}/{} lr: {:.5f}".format(self.epoch+1, self.max_epoch, self.iter, self.epoch_iter,
                                                                                                self.optimizer.state_dict()['param_groups'][0]['lr'] if not isinstance(self.optimizer, list) else self.optimizer[0].state_dict()['param_groups'][0]['lr']
                                                                                                )
            )
            logger.info(
                "Loss of batch: {:.4f} ETA:{:.2f} hours \n".format(self.visual_loss/self.args.visual_batch_interval, left_hours)
            )
            self.visual_loss = 0.
            
    def after_epoch(self):
        if self.is_main_process:
            logger.info(
                "Epoch {} Training Loss is {:.4f}".format(self.epoch+1, self.epoch_loss/self.iter)
            )
        
        if isinstance(self.lr_scheduler, list):
            for sub_lr_scheduler  in self.lr_scheduler:
                if self.lr_scheduler is not None and not self.step_update:
                    sub_lr_scheduler.step()
        else:
            if self.lr_scheduler is not None and not self.step_update:
                self.lr_scheduler.step()

        if (self.epoch + 1) % self.args.save_interval == 0:
            self.save_ckpt(ckpt_name="epoch_{}_{:.4f}".format(self.epoch+1, self.epoch_loss/self.iter))

        if (self.epoch + 1) % self.args.eval_interval == 0 and  (self.epoch + 1) > self.args.start_eval:
            if self.is_main_process:
                logger.info("Starting evaluation ....")
            self.evaluate_and_save_model(self.config.get('loss_criteria', True))

    def after_train(self):
        if self.is_main_process:
            logger.info(
                "Training of experiment is done and the best score is {:.2f}".format(
                    self.best_score)
            )
        if self.is_main_process:
            torch.save(self.model, os.path.join(self.file_name,  "test_model.pth"))
        if self.is_distributed:
            dist.destroy_process_group()

    def pretrained_init_weight(self):
        self.model.load_pretrained_model(self.args.pretrained)

    def resume_train(self):
        if self.args.resume:
            if self.is_main_process:
                logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=torch.device( 'cpu'))
            self.load_ckpt(ckpt["model"])
            if isinstance(self.optimizer, list):
                self.optimizer[0].load_state_dict(ckpt["optimizer"][0])
                self.optimizer[1].load_state_dict(ckpt["optimizer"][1])
            else:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if isinstance(self.lr_scheduler, list):
                self.lr_scheduler[0].load_state_dict(ckpt["scheduler"][0])
                self.lr_scheduler[1].load_state_dict(ckpt["scheduler"][1])
            else:
                self.lr_scheduler.load_state_dict(ckpt["scheduler"])
            if self.amp_training and "amp" in ckpt:
                amp.load_state_dict(ckpt["amp"])
            self.start_epoch = ckpt["start_epoch"]
            if self.is_main_process:
                logger.info(
                    "loaded checkpoint '{}' (epoch {})".format(
                        self.args.resume, self.start_epoch
                    )
                ) 
        else:
            if self.args.ckpt is not None:
                if self.is_main_process:
                    logger.info("loading checkpoint for finetuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=torch.device( 'cpu'))["model"]
                self.load_ckpt(ckpt)

            self.start_epoch = 0

    def load_ckpt(self, ckpt):
        match_tensor = 0
        model_state_dict = self.model.state_dict()
        load_dict = {}
        for key_model, v in model_state_dict.items():
            if key_model not in ckpt:
                if key_model[7:] not in ckpt:
                    if self.is_main_process:
                        logger.warning(
                            "{} is not in the ckpt. Please double check and see if this is desired.".format(
                                key_model
                            )
                        )
                    continue
                else:
                    v_ckpt = ckpt[key_model[7:]]
                    if v.shape != v_ckpt.shape:
                        if self.is_main_process:
                            logger.warning(
                                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                                    key_model, v_ckpt.shape, key_model, v.shape
                                )
                            )
                            continue
                    match_tensor += 1
                    load_dict[key_model] = v_ckpt
            else:
                v_ckpt = ckpt[key_model]
                if v.shape != v_ckpt.shape:
                    if self.is_main_process:
                        logger.warning(
                            "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                                key_model, v_ckpt.shape, key_model, v.shape
                            )
                        )
                        continue
                match_tensor += 1
                load_dict[key_model] = v_ckpt

        self.model.load_state_dict(load_dict, strict=False)
        if self.is_main_process:
            print("Model inital tensors: {}/{}".format(match_tensor, len(ckpt)))

    def evaluate_and_save_model(self, loss_criteria=True):
        score = self.Model.eval_model(self.model, self.eval_loader, self.loss_function, self.args.rank,
                                                                        self.is_distributed, self.is_main_process, nprocs=self.args.nprocs, loss_criteria=loss_criteria)
        if self.best_score is None:
            if self.is_main_process:
                logger.info("Best score init: {:.4f}".format(score))
                self.best_score = score
                self.save_ckpt("best_epoch_{}_{:.4f}.pth".format(self.epoch+1, score), update_best_ckpt=True)
            
        elif score < self.best_score and loss_criteria:
            if self.is_main_process:
                logger.info("Best score update: {:.4f} --> {:.4f}".format(self.best_score, score))
                r_flag = remove_file(self.file_name, 'best_epoch')
                self.save_ckpt("best_epoch_{}_{:.4f}.pth".format(self.epoch+1, score), update_best_ckpt=True)
                self.best_score = min(self.best_score, score)

        elif score > self.best_score and not loss_criteria:
            if self.is_main_process:
                logger.info("Best score update: {:.4f} --> {:.4f}".format(self.best_score, score))
                r_flag = remove_file(self.file_name, 'best_epoch')
                self.save_ckpt("best_epoch_{}_{:.4f}.pth".format(self.epoch+1, score), update_best_ckpt=True)
                self.best_score = max(self.best_score, score)
            
        else:
            if self.is_main_process:
                logger.info("Current score: {:.4f} Best socre {:.4f}. Do not update best model".format(score, self.best_score))
    
    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.is_main_process:
            logger.info("Save weights to {} \n".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": self.model.state_dict() if not self.is_distributed else self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict() if not isinstance(self.optimizer, list) else [op.state_dict() for op in self.optimizer],
                "scheduler": self.lr_scheduler.state_dict() if not isinstance(self.lr_scheduler, list) else [lr.state_dict() for lr in self.lr_scheduler]
            }
            if self.amp_training:
                ckpt_state["amp"] = amp.state_dict()
            
            self.save_checkpoint(
                                ckpt_state,
                                update_best_ckpt,
                                self.file_name,
                                ckpt_name
                                )
            
    def save_checkpoint(self, state, is_best, save_dir, model_name=""):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, model_name + "_ckpt.pth")
        if is_best:
            best_filename = os.path.join(save_dir, model_name)
            torch.save(state, best_filename)
        else:
            torch.save(state, filename)
