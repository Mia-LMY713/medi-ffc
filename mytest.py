import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import glob
import numpy as np
import torch
import utils
import torch.nn as nn
from models_rddcnn import DnCNN, DnCNN_skip
from skimage.io import imread, imsave
import time
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from PIL import Image
from skimage import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='/home/blb/blbdata/medi', type=str, help='path of test dataset')
    parser.add_argument('--set_names', default=['RDDCNN_test_data'], help='test dataset names')
    parser.add_argument('--sigma', default=30, type=int, help='noise level')
    parser.add_argument('--model_dir', default=r'./models/RDDCNN_ffc2_P50_11-06 08:18:14', help='directory of the model') # 调用路径修改
    parser.add_argument('--mode', default='B', type=str, help='the model name')
    parser.add_argument('--noise_range_low', default=0, type=int, help='noise lower bound in blind mode')
    parser.add_argument('--noise_range_high', default=55, type=int, help='noise higher bound in blind mode')
    parser.add_argument('--result_dir', default='./results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the de-noised image, 1 or 0')
    parser.add_argument('--test_dir', default='./datasets/test/RDDCNN_test_data', type=str,
                        help='path of test dataset')
    parser.add_argument('--test_gd_dir', default='.datasets/test/RDDCNN_test_sharp', type=str,
                        help='path of test dataset')
    return parser.parse_args()


def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    dirname=os.path.dirname(path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        result=(result-result.min())/(result.max()-result.min())
        result*=255
        # result=np.clip(result, 0, 1)*255
        img=Image.fromarray(result.astype(np.uint8))
        img.save(path)
        # imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def main():
    args = parse_args()
    model = DnCNN(image_channels=1).cuda()
    # device_ids = [0]
    # model = nn.DataParallel(model, device_ids=device_ids)

    model_name = 'model_006.pth'
    model.load_state_dict(torch.load(os.path.join(args.model_dir, model_name)))

    model.eval()  # evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()
    with torch.no_grad():
        ans = []

        psnr_preds = []
        ssim_preds=[]
        psnr_oris=[]
        ssim_oris=[]
        for im in os.listdir(args.test_dir):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):


                img_gd = np.array(Image.open(os.path.join(args.test_gd_dir, im[:-4] + '_sharp.png')))
                gd = img_gd[:, :, 0] / 255.0
                np.random.seed(seed=8)  # for reproducibility

                if args.mode == 'S':
                    noise = torch.randn(gd.shape).mul_(args.sigma / 255.0)
                    x = torch.tensor(gd).float() + noise
                    x = x.view(1, -1, x.shape[0], x.shape[1])
                if args.mode == 'P':
                    o_gd = (gd * 255).astype(np.uint8)
                    x = torch.from_numpy(util.random_noise(o_gd, "poisson")).float()
                    x = x.view(1, -1, x.shape[0], x.shape[1])
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
                x = x.cpu().detach().numpy().astype(np.float32)
                x = np.squeeze(x)

                psnr_pred = peak_signal_noise_ratio(gd, pred)
                ssim_pred = structural_similarity(gd, pred)
                psnr_ori = peak_signal_noise_ratio(gd, x)
                ssim_ori = structural_similarity(gd, x)
                psnr_preds.append(psnr_pred)
                ssim_preds.append(ssim_pred)
                psnr_oris.append(psnr_ori)
                ssim_oris.append(ssim_ori)
                if args.save_result:

                    name, ext = os.path.splitext(im)
                    # show(np.hstack((y, x_)))  # show the image
                    save_result(pred, path=os.path.join(args.result_dir, model.model_name(), name+'pred'+'_%.3f'% psnr_pred +ext))  # save the denoised image
                    #save_result(gd, path=os.path.join(args.result_dir, model.model_name(), name +'gd'+'_%.3f'% psnr_ori+ ext))
        psnr_avg = np.mean(psnr_preds)
        print(psnr_avg)


if __name__ == "__main__":
    main()
