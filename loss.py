import torch
import matplotlib.pyplot as plt

# 定义一个函数来加载.pth文件并提取损失数据
def load_losses_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint['losses']

# 定义多个.pth文件的路径
checkpoint_paths = ['/home/lmy/medi-ffc/models/RDDCNN_ffc2_skip_S15_04-03 08:37:58/model_001.pth',
                    '/home/lmy/medi-ffc/models/RDDCNN_ffc2_skip_S15_04-03 08:37:58/model_002.pth']

# 初始化一个列表来存储所有的损失数据
all_losses = []

# 依次加载每个.pth文件并提取损失数据
for checkpoint_path in checkpoint_paths:
    losses = load_losses_from_checkpoint(checkpoint_path)
    all_losses.append(losses)

# 创建横轴（训练步骤或训练周期）和纵轴（损失值）
epochs = range(1, len(all_losses[0]) + 1)

# 绘制损失曲线
for i, losses in enumerate(all_losses):
    plt.plot(epochs, losses, label=f'Checkpoint {i+1}')

plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
