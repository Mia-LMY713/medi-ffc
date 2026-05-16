import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import glob
import re
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from models_rddcnn import DnCNN, DnCNN_skip
import utils
from torch.optim.lr_scheduler import MultiStepLR
from data_generator import DenoisingDataset
from warnings import filterwarnings
from skimage import util
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from PIL import Image
from collections import defaultdict
import pandas as pd
from datetime import datetime
filterwarnings('ignore')


# Params
parser = argparse.ArgumentParser(description='DnCNN-ffc')
parser.add_argument('--model', default='RDDCNN_ffc2_skip', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--patch_size', default=40, type=int, help='batch size')
parser.add_argument('--train_data', default='./datasets/train/RDDCNN_train_data', type=str, help='path of train data') # 调用数据
parser.add_argument('--label_data', default='./datasets/train/RDDCNN_train_SPCT_data', type=str, help='path of train data') # 调用数据
parser.add_argument('--sigma', default=50, type=int, help='noise level')
parser.add_argument('--noise_range_low', default=0, type=int, help='noise lower bound in blind mode')
parser.add_argument('--noise_range_high', default=55, type=int, help='noise higher bound in blind mode')
parser.add_argument('--epoch', default=100, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--mode', default='B', type=str, help='training mode S for known noise level, B for unknown')
parser.add_argument('--test_dir', default='./datasets/test/RDDCNN_test_data', type=str, help='path of test dataset')
parser.add_argument('--test_gd_dir', default='./datasets/test/RDDCNN_test_sharp', type=str, help='path of test dataset')
args = parser.parse_args()

batch_size = args.batch_size
patch_size = args.patch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma

def evaluate(model, res_dict,epoch):

    model.eval()  # evaluation mode
    with torch.no_grad():
        psnr_preds = []
        psnr_oris=[]
        ssim_preds=[]
        ssim_oris=[]
        for im in os.listdir(args.test_dir):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):

                img_gd=np.array(Image.open(os.path.join(args.test_gd_dir, im[:-4]+'_sharp.png')))
                gd = img_gd[:,:,0]/255.0
                np.random.seed(seed=8)  # for reproducibility

                if args.mode == 'S':
                    noise = torch.randn(gd.shape).mul_(args.sigma / 255.0)
                    x = torch.tensor(gd).float() + noise
                    x=x.view(1, -1, x.shape[0], x.shape[1])
                if args.mode == 'P':
                    o_gd = (gd * 255).astype(np.uint8)
                    x = torch.from_numpy(util.random_noise(o_gd, "poisson")).float()
                    x=x.view(1, -1, x.shape[0], x.shape[1])
                if args.mode == 'B':
                    noise = torch.zeros(gd.shape)
                    stdN = np.random.uniform(args.noise_range_low, args.noise_range_high, size=1)

                    sizeN = noise.shape
                    noise = torch.randn(sizeN).mul_(stdN[0] / 255.0)
                    x = torch.tensor(gd).float() + noise
                    x = x.view(1, -1, x.shape[0], x.shape[1])
                if args.mode == 'gd':
                    img_x = np.array(Image.open(os.path.join(args.test_dir, im))).astype(np.float32)
                    x = img_x[:, :, 0] / 255.0
                    x = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])


                start_time = time.time()

                torch.cuda.synchronize()
                x = x.cuda()
                pred = model(x)  # inference
                pred = pred.cpu()
                pred = pred.detach().numpy().astype(np.float32)
                pred = np.squeeze(pred)  # .transpose(1,2,0)
                torch.cuda.synchronize()
                x=x.cpu().detach().numpy().astype(np.float32)
                x=np.squeeze(x)

                psnr_pred = peak_signal_noise_ratio(gd, pred)
                ssim_pred = structural_similarity(gd, pred)
                psnr_ori = peak_signal_noise_ratio(gd, x)
                ssim_ori = structural_similarity(gd, x)
                psnr_preds.append(psnr_pred)
                ssim_preds.append(ssim_pred)
                psnr_oris.append(psnr_ori)
                ssim_oris.append(ssim_ori)
        psnr_pred_avg = np.mean(psnr_preds)
        ssim_pred_avg = np.mean(ssim_preds)
        psnr_ori_avg = np.mean(psnr_oris)
        ssim_ori_avg = np.mean(ssim_oris)
        print('evaluate mean:',psnr_pred_avg,  psnr_ori_avg, ssim_pred_avg, ssim_ori_avg)
        res_dict['epoch'].append(epoch)
        res_dict['psnr_pred_avg'].append(psnr_pred_avg)
        res_dict['ssim_pred_avg'].append(ssim_pred_avg)
        res_dict['psnr_ori_avg'].append(psnr_ori_avg)
        res_dict['ssim_ori_avg'].append(ssim_ori_avg)
        return res_dict


class sum_squared_error(_Loss):
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, reduction='sum').div_(2)


def main():
    print('===> Building model')
    model = DnCNN_skip(image_channels=1)
    # model = DnCNN(image_channels=1)

    model.train()
    criterion_mse = sum_squared_error()
    if cuda:
        model = model.cuda()
        # device_ids = [0]
        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        criterion_mse = criterion_mse.cuda()

    initial_epoch=0
    # initial_epoch = utils.findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    # if initial_epoch > 0:
    #     print('resuming by loading epoch %03d' % initial_epoch)
    #     model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
    #     # model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates

    res_dict=defaultdict(list)
    DDataset = DenoisingDataset(data_dir=args.train_data, label_dir=args.label_data,
                                 patch_size=args.patch_size)
    DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True,
                         batch_size=batch_size, shuffle=True)
    print(f'dataset size: {len(DDataset)} patches')

    for epoch in range(initial_epoch, n_epoch):
        epoch_loss = 0
        start_time = time.time()

        for n_count, batch in enumerate(DLoader):
            optimizer.zero_grad()

            if args.mode == 'S':
                _,batch_y=batch
                noise = torch.randn(batch_y.size()).mul_(args.sigma/255.0)
                batch_x = batch_y + noise
            if args.mode=='P':
                _, batch_y=batch
                obatch_y=(batch_y*255).numpy().astype(np.uint8)
                batch_x = torch.from_numpy(util.random_noise(obatch_y, "poisson")).float()
            if args.mode == 'B':
                _, batch_y=batch
                noise = torch.zeros(batch_y.size())
                stdN = np.random.uniform(args.noise_range_low, args.noise_range_high, size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0, :, :, :].size()
                    noise[n, :, :, :] = torch.randn(sizeN).mul_(stdN[n]/255.0)
                batch_x = batch_y + noise
            if args.mode == 'gd':
                batch_x, batch_y = batch

            if cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            loss = criterion_mse(model(batch_x), batch_y) / batch_size
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, len(DDataset)//batch_size, loss.item()/batch_size))
        elapsed_time = time.time() - start_time
        scheduler.step()  # step to the learning rate in this epoch

        utils.log('epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        res_dict=evaluate(model, res_dict, epoch)
        res=pd.DataFrame(res_dict)
        res.to_csv(os.path.join(save_dir, 'results.csv'), index=None)
        model.train()


if __name__ == "__main__":
    save_dir = os.path.join('./models', args.model + '_'+args.mode +str(args.sigma)+'_'+datetime.now().strftime("%m-%d %H:%M:%S"))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    main()
