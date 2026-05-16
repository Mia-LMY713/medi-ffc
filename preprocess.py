from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
import os
from PIL import Image
'''
img1 = Image.open('E:/数据集/image/JPCLN001.png')
img2 = Image.open('E:/数据集/gt/JPCLN001.png')
img3 = Image.open('E:/RDDCNN_train_data/cxy_00001.png')
img4 = Image.open('E:/RDDCNN_test_sharp/crown_cbct_0_sharp.png')

#img1 = np.array(Image.open('JPCLN001.png'))
#img2 = np.array(Image.open('gt.png'))
    print(len(img1.split()))
    print(len(img2.split()))
    print(len(img3.split()))
    print(len(img4.split()))
    print(psnr(img1, img2))
'''
if __name__ == "__main__":

    input_folder = 'E:/数据集/image/'
    output_folder = 'E:/数据集/image_4/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 打开图像文件
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            # 转换为单通道灰度图像（去噪任务只需单通道）
            gray_img = img.convert('L')

            # 保存转换后的图像
            output_path = os.path.join(output_folder, filename)
            gray_img.save(output_path)

    print('图像批处理完成！')
