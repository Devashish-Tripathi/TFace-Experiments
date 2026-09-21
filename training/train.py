# Modifications Copyright (C) 2026 Devashish Tripathi
# Originally licensed under Apache 2.0 by Tencent.

import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# added
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)



import logging
import torch
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import torch.optim as optim
import torch.nn.init as init
from torchkit.util import AverageMeter, Timer
from torchkit.util import accuracy_dist
from torchkit.util import AllGather
from torchkit.hooks.learning_rate_hook import adjust_lr
from torchkit.loss import get_loss
from torchkit.task import BaseTask
from torchkit.head import get_head
from torchkit.util import get_class_split
from torchkit.backbone import get_model
from utils import NoisyActivation, images_to_batch

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s: %(message)s')


# added
from argparse import ArgumentParser


class TrainTask(BaseTask):
    """ TrainTask in distfc mode, which means classifier shards into multi workers
    """
    def __init__(self, cfg_file):
        super(TrainTask, self).__init__(cfg_file)

        # added
        os.makedirs(self.cfg['MODEL_ROOT'], exist_ok= True)
        os.makedirs(self.cfg['LOG_ROOT'], exist_ok= True)


    def make_model(self):
        """ build training backbone and heads
        """
        backbone_name = self.cfg['BACKBONE_NAME']
        backbone_model = get_model(backbone_name)
        self.backbone = backbone_model(self.input_size, input_channel=189)
        self.backbone.cuda()
        logging.info("{} Backbone Generated".format(backbone_name))

        embedding_size = self.cfg['EMBEDDING_SIZE']
        self.class_shards = []
        metric = get_head(self.cfg['HEAD_NAME'], dist_fc=self.dist_fc)

        for name, branch in self.branches.items():
            class_num = self.class_nums[name]
            class_shard = get_class_split(class_num, self.world_size)
            self.class_shards.append(class_shard)
            logging.info('Split FC: {}'.format(class_shard))

            init_value = torch.FloatTensor(embedding_size, class_num)
            init.normal_(init_value, std=0.01)
            head = metric(in_features=embedding_size,
                          gpu_index=self.rank,
                          weight_init=init_value,
                          class_split=class_shard,
                          scale=branch.scale,
                          margin=branch.margin)
            del init_value
            head = head.cuda()
            self.heads[name] = head

    def loop_step(self, epoch, nonoise):
        """
        load_data
            |
        extract feature
            |
        optimizer step
            |
        print log and write summary
        """
        backbone, heads, noise_model = self.backbone, list(self.heads.values()), self.noise_model
        noise_model.train()
        backbone.train()  # set to training mode
        for head in heads:
            head.train()

        batch_sizes = self.batch_sizes
        am_losses = [AverageMeter() for _ in batch_sizes]
        am_top1s = [AverageMeter() for _ in batch_sizes]
        am_top5s = [AverageMeter() for _ in batch_sizes]
        t = Timer()
        for step, samples in enumerate(self.train_loader):
            # call hook function before_train_iter
            self.call_hook("before_train_iter", step, epoch)
            backbone_opt, head_opts, noise_opt = self.opt['backbone'], list(self.opt['heads'].values()), self.noise_opt

            inputs = samples[0].cuda(non_blocking=True)
            labels = samples[1].cuda(non_blocking=True)

            inputs = images_to_batch(inputs)
            inputs = inputs.detach()
            if not nonoise:
                inputs = noise_model(inputs)

            all_features, all_labels = self.backbone_forward(backbone, inputs, labels, batch_sizes)
            losses = []
            for i in range(len(batch_sizes)):
                # PartialFC need update optimizer state in training process
                if self.pfc: 
                    outputs, labels, original_outputs = self.partialfc_head_forward(heads[i], all_features[i],
                                                                                    all_labels[i], head_opts[i])
                else:
                    outputs, labels, original_outputs = self.general_head_forward(heads[i], all_features[i],
                                                                                  all_labels[i])

                loss = self.loss(outputs, labels) * self.branch_weights[i]
                losses.append(loss)
                prec1, prec5 = accuracy_dist(self.cfg,
                                             original_outputs.data,
                                             all_labels[i],
                                             self.class_shards[i],
                                             topk=(1, 5))
                am_losses[i].update(loss.data.item(), all_features[i].size(0))
                am_top1s[i].update(prec1.data.item(), all_features[i].size(0))
                am_top5s[i].update(prec5.data.item(), all_features[i].size(0))

            # update summary and log_buffer
            scalars = {
                'train/loss': am_losses,
                'train/top1': am_top1s,
                'train/top5': am_top5s,
            }
            self.update_summary({'scalars': scalars})
            log = {
                'loss': am_losses,
                'prec@1': am_top1s,
                'prec@5': am_top5s,
            }
            self.update_log_buffer(log)

            # compute loss
            total_loss = sum(losses)
            # compute gradient and do SGD
            if nonoise:
                total_opts = [backbone_opt] + head_opts
            else:
                total_opts = [backbone_opt, noise_opt] + head_opts

            self.backward_and_update(total_loss, total_opts, self.scaler)
            
            # PartialFC need update weight and weight_norm manually
            if self.pfc:
                for head in heads:
                    head.update()

            cost = t.get_duration()
            self.update_log_buffer({'time_cost': cost})

            # call hook function after_train_iter
            self.call_hook("after_train_iter", step, epoch)

    def get_noise_opt(self, noise_model):
        optimizer = optim.Adam(list(noise_model.module.parameters()), lr=self.cfg['LRS_NOISE'][0])
        return optimizer

    def prepare(self, nolocs, usesens, senspth, epstrain):
        """ common prepare task for training
        """
        self.make_inputs()
        self.make_model()
        self.loss = get_loss('DistCrossEntropy').cuda()
        self.opt = self.get_optimizer()
        self.register_hooks()
        self.pfc = self.cfg['HEAD_NAME'] == 'PartialFC'
        # changed
        self.noise_model = NoisyActivation(sensitivity=usesens, sens_pth= senspth, donot_use_loc= nolocs, budget_mean= epstrain).cuda()

    # added
    def save_ckpt(self, epoch):
        super().save_ckpt(epoch)
        if self.rank == 0:
            noise_path = os.path.join(self.cfg['MODEL_ROOT'], f"Noise_Epoch_{epoch}_checkpoint.pth")
            torch.save(self.noise_model.module.state_dict(), noise_path)

    def train(self, params):
        """
        make inputs
            |
        make model
            |
        make loss function
            |
        make optimizer
            |
        make auto mix precision grad scalar
            |
        register hooks
            |
        Distributed Data Parallel mode
            |
        loop_step
        """
        nonoise, nolocs, usesens, senspth, epstrain = params
        self.prepare(nolocs, usesens, senspth, epstrain)
        self.call_hook("before_run")
        self.backbone = DistributedDataParallel(self.backbone, device_ids=[self.local_rank])
        self.noise_model = DistributedDataParallel(self.noise_model, device_ids=[self.local_rank], find_unused_parameters=True)
        self.noise_opt = self.get_noise_opt(self.noise_model)
        for epoch in range(self.start_epoch, self.epoch_num):
            self.call_hook("before_train_epoch", epoch)
            adjust_lr(epoch, self.cfg['LRS_NOISE'], self.cfg['STAGES'], self.noise_opt)
            self.loop_step(epoch, nonoise)
            self.call_hook("after_train_epoch", epoch)
        self.call_hook("after_run")


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    
    # added
    parser = ArgumentParser()
    parser.add_argument('--yaml_name', help='name of the training yaml file', default= 'train.yaml')
    parser.add_argument('--no_noise', action='store_true', help='toggle off noise adding module. for debugging')
    parser.add_argument('--eps_train', type= float, default= 4.0, help='training epsilon value')
    parser.add_argument('--no_locs', action='store_true', help='toggle off locs in NoisyActivation. Don\'t use this if you want as close as baseline')
    parser.add_argument('--use_sensitivity', action='store_true', help='Disable to follow the baseline (sensitivity=1)')
    parser.add_argument('--sens_pth', type=str, default=None, help='Path to dataset sensitivity tensor. Make using make_sense.py script!')

    args, unknown = parser.parse_known_args()

    # added
    params = [args.no_noise, args.no_locs, args.use_sensitivity, args.sens_pth, args.eps_train]

    # modified
    task = TrainTask(os.path.join(task_dir, args.yaml_name))
    task.init_env()
    task.train(params)


if __name__ == '__main__':
    main()
