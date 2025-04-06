import torch
from utils.utils import *
from tqdm import tqdm
import argparse
import os
from utils.dataloader_bsergb import *
from models.model_manager import OurModel
from utils.flow_utils import *
import torchvision.utils as vutils


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--voxel_num_bins', type=int, default=16)
    parser.add_argument('--flow_tb_debug', type=str2bool, default='False')
    parser.add_argument('--flow_tb_viz', type=str2bool, default='True')
    parser.add_argument('--warp_tb_debug', type=str2bool, default='True')
    parser.add_argument('--val_mode', type=str2bool, default='False')
    parser.add_argument('--val_skip_num_list', default=[1, 3])
    parser.add_argument('--model_folder', type=str, default='final_models')
    parser.add_argument('--model_name', type=str, default='ours_large')
    parser.add_argument('--use_smoothness_loss', type=str2bool, default='True')
    parser.add_argument('--smoothness_weight', type=float, default=10.0)
    parser.add_argument('--num_threads', type=int, default=12)
    parser.add_argument('--experiment_name', type=str, default='test_bsergb_dataset')
    parser.add_argument('--tb_update_thresh', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='/home/user/dataset/bsergb_interpolation_v2/')
    parser.add_argument('--ckpt_path',  type=str,   default='pretrained_model/Ours_Large_BSERGB.pth')
    parser.add_argument('--use_multigpu', type=str2bool, default='True')
    parser.add_argument('--train_skip_num_list', default=[1, 3])
    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self._init_model()
        self._init_metrics()
        self._init_dataloader()

    def _init_dataloader(self):
        val_set_dict = get_BSERGB_val_dataset(self.args.data_dir, self.args.val_skip_num_list, mode='1_TEST')
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
        ckpt = torch.load(self.args.ckpt_path, map_location='cpu')
        self.model.load_model(ckpt['model_state_dict'])

        if torch.cuda.is_available():
            self.model.cuda()

        if self.args.use_multigpu:
            self.model.use_multi_gpu()

    def _init_metrics(self):
        self.PSNR_calculator = PSNR()
        self.SSIM_calculator = SSIM()

    def test_joint(self, epoch=0):
        psnr_total = AverageMeter()
        ssim_total = AverageMeter()
        psnr_interval = AverageMeter()
        ssim_interval = AverageMeter()

        self.model.eval()
        self.model.set_mode('joint')

        os.makedirs('./outputs', exist_ok=True)
        os.makedirs(f'./logs/{self.args.experiment_name}', exist_ok=True)

        with torch.no_grad():
            for skip_num, val_loader in self.val_loader_dict.items():
                output_save_path = f'./outputs/net_out/{skip_num}skip'
                gt_save_path = f'./outputs/gt/{skip_num}skip'
                os.makedirs(output_save_path, exist_ok=True)
                os.makedirs(gt_save_path, exist_ok=True)
                for i, sample in enumerate(tqdm(val_loader, desc=f'val skip {skip_num}')):
                    sample = batch2device(sample)
                    self.model.set_test_input(sample)
                    self.model.forward_joint_test()

                    gt = sample['clean_middle']
                    pred = self.model.test_outputs['interp_out']

                    psnr = self.PSNR_calculator(gt, pred).mean().item()
                    ssim = self.SSIM_calculator(gt, pred).mean().item()
                    print(psnr)

                    psnr_interval.update(psnr)
                    ssim_interval.update(ssim)
                    psnr_total.update(psnr)
                    ssim_total.update(ssim)

                    # 이미지 저장
                    output_name = os.path.join(output_save_path, str(i).zfill(5) + '.png')
                    vutils.save_image(pred[0], output_name)
                    # gt 저장
                    gt_name = os.path.join(gt_save_path, str(i).zfill(5) + '.png')
                    vutils.save_image(gt[0], gt_name)

                print(f"[Skip {skip_num}] PSNR: {psnr_interval.avg:.2f}, SSIM: {ssim_interval.avg:.4f}")
                psnr_interval.reset()
                ssim_interval.reset()

        avg_psnr = psnr_total.avg
        avg_ssim = ssim_total.avg

        print(f"\n[Test Summary] Avg PSNR: {avg_psnr:.2f}, Avg SSIM: {avg_ssim:.4f}")

        # 로그 저장
        log_path = os.path.join('./logs', self.args.experiment_name, f'test_result_epoch{epoch}.txt')
        with open(log_path, 'w') as f:
            f.write(f"Experiment: {self.args.experiment_name}\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Average PSNR: {avg_psnr:.2f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")

        torch.cuda.empty_cache()
        self.model.test_outputs = {}
        return avg_psnr, avg_ssim


if __name__ == '__main__':
    args = get_argument()
    trainer = Trainer(args)
    trainer.test_joint(epoch=0)