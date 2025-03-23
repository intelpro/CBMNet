import torch
from utils.utils import *
from torch.optim import AdamW
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import datetime
import argparse
import os
from utils.dataloader_bsergb import *
from models.model_manager import OurModel
import torch.optim as optim
from utils.flow_utils import *



def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_epochs', type = int, default=301)
    parser.add_argument('--end_epochs_flow', type = int, default=100)
    parser.add_argument('--batch_size', type = int, default=1)
    parser.add_argument('--val_batch_size', type = int, default=1)
    # training params
    parser.add_argument('--voxel_num_bins', type = int, default=16)
    parser.add_argument('--crop_size', type = int, default=256)
    parser.add_argument('--learning_rate', type = float, default=1e-4)
    parser.add_argument('--mode', type = str, default='flow')
    parser.add_argument('--flow_tb_debug', type = str2bool, default='True')
    parser.add_argument('--flow_tb_viz', type = str2bool, default='True')
    parser.add_argument('--warp_tb_debug', type = str2bool, default='True')
    ## val folder
    parser.add_argument('--val_mode', type = str2bool, default='False')
    parser.add_argument('--val_skip_num_list', default=[1, 3])
    # model discription
    parser.add_argument('--model_folder', type=str, default='final_models')
    parser.add_argument('--model_name', type=str, default='ours')
    parser.add_argument('--use_smoothness_loss', type=str2bool, default='True')
    parser.add_argument('--smoothness_weight', type = float, default=10.0)
    # data loading params
    parser.add_argument('--num_threads', type = int, default=12)
    parser.add_argument('--experiment_name', type = str, default='train_bsergb_networks')
    parser.add_argument('--tb_update_thresh', type = int, default=1)
    parser.add_argument('--data_dir', type = str, default = '/media/mnt2/bs_ergb')
    parser.add_argument('--use_multigpu', type=str2bool, default='True')
    parser.add_argument('--train_skip_num_list', default=[1, 3])
    # loading module
    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self._init_counters()
        self._init_tensorboard()
        self._init_dataloader()
        self._init_model()
        self._init_optimizer()
        self._init_scheduler()
        self._init_metrics()

    def _init_counters(self):
        self.tb_iter_cnt = 0
        self.tb_iter_cnt_val = 0
        self.tb_iter_cnt2 = 0
        self.tb_iter_cnt2_val = 0
        self.tb_iter_thresh = self.args.tb_update_thresh
        self.batchsize = self.args.batch_size
        self.start_epoch = 0
        self.end_epoch = self.args.total_epochs
        self.best_psnr = 0.0 
        self.start_epoch_flow = 0
        self.end_epoch_flow = self.args.end_epochs_flow
        self.start_epoch_joint = self.args.end_epochs_flow + 1

    def _init_tensorboard(self):
        timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M')
        tb_path = os.path.join('./experiments', f"{timestamp}-{self.args.experiment_name}")
        self.tb = SummaryWriter(tb_path, flush_secs=1)

    def _init_dataloader(self):
        ## train set
        train_set = get_BSERGB_train_dataset(self.args.data_dir, self.args.train_skip_num_list, mode='3_TRAINING')
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_threads, pin_memory=True, drop_last=True)
        ## val set
        val_set_dict = get_BSERGB_val_dataset(self.args.data_dir, self.args.val_skip_num_list, mode='1_TEST')
        # make loader per skip
        self.val_loader_dict = {}
        for skip_num, val_dataset in val_set_dict.items():
            self.val_loader_dict[skip_num] = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.args.val_batch_size,
                shuffle=False,
                num_workers=self.args.num_threads,
                pin_memory=True
            )

    def _init_model(self):
        self.model = OurModel(self.args)
        self.model.initialize(self.args.model_folder, self.args.model_name)

        if torch.cuda.is_available():
            self.model.cuda()

        if self.args.use_multigpu:
            self.model.use_multi_gpu()

    def _init_optimizer(self):
        params = self.model.get_optimizer_params()
        self.optimizer = AdamW(params, lr=self.args.learning_rate)

    def _init_scheduler(self):
        if self.args.mode == 'joint':
            milestones = [40, 60]
        elif self.args.mode == 'flow':
            milestones = [30]
        else:
            milestones = []

        if milestones:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=0.5
            )

    def _init_metrics(self):
        self.PSNR_calculator = PSNR()
        self.SSIM_calculator = SSIM()

    def mode_classify(self):
        # Mode override by argument
        if self.args.mode == 'joint':
            mode = 'joint'
        elif self.epoch <= self.end_epoch_flow:
            mode = 'flow'
        elif self.start_epoch_joint <= self.epoch <= self.end_epoch:
            mode = 'joint'
        else:
            raise ValueError(f"Invalid epoch {self.epoch} for mode classification.")
        self.model.set_mode(mode)
        # Automatically freeze flownet if in joint mode
        if mode == 'joint':
            self.model.fix_flownet()
        return mode

    def train(self):
        for self.epoch in trange(self.start_epoch, self.end_epoch, desc='epoch progress'):
            self.model.train()
            mode_now = self.mode_classify()

            for _, sample in enumerate(tqdm(self.train_loader, desc='train progress')):
                self.train_step(sample, mode=mode_now)

            if self.epoch % 10 == 0 and mode_now == 'joint':
                psnr_val, _ = self.val_joint(self.epoch)
                if psnr_val > self.best_psnr:
                    self.best_psnr = psnr_val
                    self.save_model(self.epoch, best=True)
                    print(f"[Best Model Updated] Epoch {self.epoch} - PSNR: {psnr_val:.2f}")

            self.scheduler.step()


    def train_step(self, sample, mode):
        # --- Move batch to device and zero optimizer ---
        sample = batch2device(sample)
        self.optimizer.zero_grad()

        # --- Set input for model ---
        self.model.set_train_input(sample)

        # --- Forward pass and compute loss ---
        self.model.forward_nets()
        if mode == 'flow':
            loss = self.model.get_flow_loss()
        elif mode == 'joint':
            loss = self.model.get_multi_scale_loss()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # --- Backpropagation and optimization ---
        loss.backward()
        self.optimizer.step()

        # --- Update training status ---
        self.model.update_loss_meters(mode)
        self.tb_iter_cnt += 1

        if self.batchsize * self.tb_iter_cnt > self.tb_iter_thresh:
            self.log_train_tb(mode)

        # --- Clean up ---
        del sample

    
    def log_train_tb(self, mode):
        def add_scalar(tag, value):
            self.tb.add_scalar(tag, value, self.tb_iter_cnt2)

        def add_image(tag, image):
            self.tb.add_image(tag, image, self.tb_iter_cnt2)

        def add_flow_image(tag, flow_tensor):
            flow_img = flow_to_image(flow_tensor.detach().cpu().permute(1, 2, 0).numpy()).transpose(2, 0, 1)
            add_image(tag, flow_img)

        # --- Log loss values ---
        add_scalar('train_progress/loss_total', self.model.loss_handler.loss_total_meter.avg)
        add_scalar('train_progress/loss_flow', self.model.loss_handler.loss_flow_meter.avg)
        add_scalar('train_progress/loss_warp', self.model.loss_handler.loss_warp_meter.avg)
        add_scalar('train_progress/loss_smoothness', self.model.loss_handler.loss_smoothness_meter.avg)

        # --- Log interpolation input images ---
        add_image('train_image/clean_image_first', self.model.batch['image_input0'][0])
        add_image('train_image/clean_image_last', self.model.batch['image_input1'][0])
        add_image('train_image/interp_gt', self.model.batch['clean_gt_images'][0])

        # --- Log predicted optical flow (estimated) ---
        if self.args.flow_tb_viz:
            add_flow_image('train_flow/flow_t0_est', self.model.outputs['flow_out']['flow_t0_dict'][0][0])
            add_flow_image('train_flow/flow_t1_est', self.model.outputs['flow_out']['flow_t1_dict'][0][0])

        # --- Debug intermediate flow results ---
        if self.args.flow_tb_debug:
            add_flow_image('train_flow_debug_0/flow_event', self.model.outputs['flow_out']['event_flow_dict'][0][0])
            add_flow_image('train_flow_debug_0/flow_image', self.model.outputs['flow_out']['image_flow_dict'][0][0])
            add_flow_image('train_flow_debug_0/flow_fusion', self.model.outputs['flow_out']['fusion_flow_dict'][0][0])
            add_image('train_flow_debug_0/event_flow_mask', self.model.outputs['flow_out']['mask_dict'][0][0])

        # --- Joint training-specific logging ---
        if mode == 'joint':
            add_scalar('train_progress/loss_image', self.model.loss_handler.loss_image_meter.avg)
            add_image('train_image/interp_out', self.model.outputs['interp_out'][0][0])
        elif mode == 'flow':
            # --- Warp output visualization ---
            if self.args.warp_tb_debug:
                add_image('train_warp_output/warp_image_0t', self.model.batch['imaget_est0_warp'][0][0])
                add_image('train_warp_output/warp_image_t1', self.model.batch['imaget_est1_warp'][0][0])
                add_image('train_warp_output/warp_image_gt', self.model.batch['clean_gt_images'][0])

        # --- Update counters and reset meters ---
        self.tb_iter_cnt2 += 1
        self.tb_iter_cnt = 0
        self.model.loss_handler.reset_meters()

    def val_joint(self, epoch):
        # Total and per-interval metric meters
        psnr_total = AverageMeter()
        ssim_total = AverageMeter()
        psnr_interval = AverageMeter()
        ssim_interval = AverageMeter()

        # Set model to evaluation mode
        self.model.eval()
        # set model mode
        self.model.set_mode('joint')

        with torch.no_grad():
            for skip_num, val_loader in self.val_loader_dict.items():
                for _, sample in enumerate(tqdm(val_loader, desc=f'val skip {skip_num}')):
                    sample = batch2device(sample)
                    self.model.set_test_input(sample)
                    self.model.forward_joint_test()

                    gt = sample['clean_middle']
                    pred = self.model.test_outputs['interp_out']

                    psnr = self.PSNR_calculator(gt, pred).mean().item()
                    ssim = self.SSIM_calculator(gt, pred).mean().item()

                    psnr_interval.update(psnr)
                    ssim_interval.update(ssim)

                    psnr_total.update(psnr)
                    ssim_total.update(ssim)

                # Log per interval result
                self.tb.add_scalar(f'val_progress/BSERGB/{skip_num}skip/avg_psnr_interp', psnr_interval.avg, epoch)
                self.tb.add_scalar(f'val_progress/BSERGB/{skip_num}skip/avg_ssim_interp', ssim_interval.avg, epoch)

                psnr_interval.reset()
                ssim_interval.reset()

            # Log total result
            self.tb.add_scalar('val_progress/BSERGB/average/avg_psnr_interp', psnr_total.avg, epoch)
            self.tb.add_scalar('val_progress/BSERGB/average/avg_ssim_interp', ssim_total.avg, epoch)

        torch.cuda.empty_cache()
        self.model.test_outputs = {}
        return psnr_total.avg, ssim_total.avg

    def save_model(self, epoch):
        combined_state_dict = {
            'epoch': self.epoch,
            'model_state_dict': self.model.net.state_dict(), 
            'Optimizer_state_dict' : self.optimizer.state_dict(), 
            'Scheduler_state_dict' :  self.scheduler.state_dict()}
        torch.save(combined_state_dict, os.path.join(self.model.save_path, 'best_model_' +  str(epoch) + '_ep.pth'))
    

if __name__=='__main__':
    args = get_argument()
    trainer = Trainer(args)
    if args.val_mode == True:
        trainer.val_joint(0)
    else:
        trainer.train()
