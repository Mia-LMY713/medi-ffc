# FDENet: Frequency-Domain Enhanced Network for Low-Dose CT Image Denoising

A deep learning method for low-dose CT (LDCT) image denoising based on transform-domain algorithms. FDENet integrates **Fast Fourier Convolution (FFC)** for global frequency-domain feature extraction, **Deformable Convolution v2** for local adaptive feature extraction, and a multi-branch skip-connection fusion strategy to effectively suppress CT noise while preserving structural details.

## Architecture

FDENet consists of the following key modules:

- **FFC Block**: Two layers of Fast Fourier Convolution with BatchNorm and activation for capturing global frequency-domain features
- **DB (Deformable Block)**: Modulated Deformable Convolution v2 + ReLU for adaptive local feature extraction
- **EB (Enhancement Block)**: Multi-layer convolution with BatchNorm and dilated convolution for multi-scale feature enhancement
- **RB (Reconstruction Block)**: Final 3×3 convolution for image reconstruction
- **Concat Fusion**: Channel concatenation + 1×1 convolution for multi-branch feature fusion (replacing simple addition)

The network follows a residual learning strategy: the output is `y - f(y)`, where `f(y)` predicts the noise component.

### Overall Architecture

![Overall Architecture](Experiment%20Results/Overall%20Architecture.png)

### Robust Deformed Network Architecture

![Robust Deformed Network Architecture](Experiment%20Results/Robust%20Deformed%20Network%20Architecture.png)

## Dataset

Experiments use the **2016 NIH-AAPM-Mayo Clinic Low-Dose CT** dataset:

- **Source**: Mayo Medical Center public dataset for LDCT image quality evaluation
- **Content**: 10 anonymous patients, abdominal 512×512 images
  - Low-dose CT (LDCT): 1mm and 3mm slices
  - Normal-dose CT (NDCT): 120kV tube voltage, 200mAs
- **Simulation**: Quarter-dose LDCT simulated by adding Poisson noise to normal-dose projections
- **Training set**: 9 patients, 450 image pairs (3mm slices)
- **Test set**: 100 image pairs (randomly selected)

### Dataset Samples

![Mayo Clinic Low-Dose CT Dataset](Experiment%20Results/Mayo%20Clinic%20Low%20Dose%20CT%20Dataset.png)

## Experimental Settings

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA A100 (40GB) |
| Framework | PyTorch |
| Optimizer | Adam |
| Batch size | 8 |
| Patch size | 40×40 |
| Training patches | ~118,124 |
| Epochs | 100 |
| Initial learning rate | 1e-3 |
| LR schedule | MultiStepLR (milestones: 30, 60, 90; gamma: 0.2) |
| Loss weights | λ₁=0.2, λ₂=0.8 |

## Results

Quantitative comparison on the Mayo LDCT dataset (3mm slices):

| Method | PSNR (dB) | SSIM | GMSD |
|--------|-----------|------|------|
| LDCT | 30.2805 | 0.8585 | 0.0876 |
| REDCNN | 33.2021 | 0.9109 | 0.0584 |
| ED-CNN | 33.2217 | 0.9081 | 0.0600 |
| CNCL | 33.1521 | **0.9186** | 0.0587 |
| QAE | 33.6436 | 0.9161 | **0.0581** |
| CTformer | 33.3942 | 0.9141 | 0.0589 |
| **FDENet (Ours)** | **33.8115** | 0.9184 | 0.0587 |

FDENet achieves the **highest PSNR** among all compared methods while maintaining competitive SSIM and GMSD scores.

### Visual Comparison

![Denoising Results](Experiment%20Results/Denoising%20Results.png)

### Local Detail Comparison

![Local Denoising Results](Experiment%20Results/Local%20Denoising%20Results.png)

### ROI Zoom-in Comparison

![ROI Denoising Results](Experiment%20Results/ROI%20denoising%20results.png)

## Project Structure

```
medi-ffc/
├── train.py              # Training script
├── mytest.py             # Testing / inference script
├── models.py             # Network architectures (DnCNN, DnCNN_skip/FDENet)
├── ffc.py                # Fast Fourier Convolution modules
├── deform_conv_v2.py     # Deformable Convolution v2 (torchvision.ops)
├── data_generator.py     # Dataset class with dynamic patch loading
├── utils.py              # Utility functions (PSNR, weight init, etc.)
├── preprocess.py         # Data preprocessing
├── loss.py               # Loss visualization
└── README.md
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.8
- torchvision >= 0.9 (for `torchvision.ops.deform_conv2d`)
- numpy
- opencv-python
- scikit-image
- Pillow
- pandas

Install dependencies:

```bash
pip install torch torchvision numpy opencv-python scikit-image Pillow pandas
```

## Usage

### Data Preparation

Organize your dataset as follows:

```
datasets/
├── train/
│   ├── RDDCNN_train_data/       # Low-dose CT training images
│   └── RDDCNN_train_SPCT_data/  # Normal-dose CT labels
└── test/
    ├── RDDCNN_test_data/        # Low-dose CT test images
    └── RDDCNN_test_sharp/       # Normal-dose CT ground truth
```

### Training

```bash
python train.py \
    --model RDDCNN_ffc2_skip \
    --batch_size 8 \
    --patch_size 40 \
    --epoch 100 \
    --lr 1e-3 \
    --mode gd \
    --train_data ./datasets/train/RDDCNN_train_data \
    --label_data ./datasets/train/RDDCNN_train_SPCT_data \
    --test_dir ./datasets/test/RDDCNN_test_data \
    --test_gd_dir ./datasets/test/RDDCNN_test_sharp
```

**Training modes:**
- `gd`: Use ground-truth paired data (LDCT input + NDCT label)
- `S`: Fixed Gaussian noise level (sigma)
- `B`: Blind denoising (random noise level)
- `P`: Poisson noise

### Testing

```bash
python mytest.py \
    --model_dir ./models/YOUR_MODEL_DIR \
    --test_dir ./datasets/test/RDDCNN_test_data \
    --test_gd_dir ./datasets/test/RDDCNN_test_sharp \
    --mode gd
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{FDENet2024,
  title={FDENet: Frequency-Domain Enhanced Network for Low-Dose CT Image Denoising},
  year={2024}
}
```

## License

This project is for academic research purposes only.
