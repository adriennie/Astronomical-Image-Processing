# Denoising Astronomical Images with a Convolutional Autoencoder to Improve YOLO-Style Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)

**Author:** Adrika Pradhan (B.Tech Computer Science, AI & ML, University of Petroleum and Energy Studies)

## At a glance

- **Goal:** test whether denoising astronomical images with a convolutional autoencoder improves the output of a small YOLO-style detector.
- **Pipeline:** two stages. A convolutional autoencoder is trained first to remove synthetic Gaussian noise. A YOLO-inspired detector is then trained on noisy images and, separately, on denoised images, and the two are compared.
- **Data:** DeepSpaceYOLO images (galaxies, nebulae and other structures), resized to 128 x 128, with synthetic Gaussian noise (sigma = 0.2) added.
- **Framework:** TensorFlow / Keras, trained from scratch (no pretrained weights).
- **Status:** a small research study. The detection results use placeholder labels and should be read as a relative comparison, not as detection accuracy on real annotations (see [Limitations](#limitations)).

## Results

| Metric | Value | Notes |
|---|---|---|
| PSNR (denoised vs. clean) | 27.52 dB | Test-set average |
| SSIM (denoised vs. clean) | 0.88 | Test-set average |
| Mean IoU, detector on noisy inputs | 0.51 | Placeholder labels |
| Mean IoU, detector on denoised inputs | 0.69 | Placeholder labels |

Comparison with other methods on the same test subset (values for the baselines are read from the comparison figure in the paper, so treat them as approximate):

| Method | PSNR | SSIM | IoU |
|---|---|---|---|
| This work | 27.52 dB | 0.88 (highest) | 0.69 (highest) |
| DnCNN | about 29 dB | lower | lower |
| U-Net | about 28 dB | lower | lower |
| Standard autoencoder | lower | lower | about 0.55 |
| YOLOv1 | very low | very low | about 0.42 |

In short: the proposed model gives the highest SSIM and IoU in this comparison, while DnCNN and U-Net score higher on PSNR.

## Method

### Data
- DeepSpaceYOLO dataset: astronomical images with YOLO-format annotations, captured by Stellina and Vespera smart telescopes.
- Images are loaded with OpenCV, resized to 128 x 128 and normalised to [0, 1].
- Noisy inputs are created by adding Gaussian noise with a standard deviation of 0.2 to the clean images.
- 80/20 train/test split, with 10% of the training portion used for validation.

### Denoising autoencoder
- **Encoder:** Conv2D(32) and Conv2D(64), each with 3 x 3 kernels, ReLU and 'same' padding, each followed by 2 x 2 max pooling.
- **Decoder:** two Conv2DTranspose layers (64, then 32 filters, stride 2), then a Conv2D(3) layer with sigmoid activation.
- **Training:** MSE loss, Adam optimiser, learning rate 1e-3, 20 epochs, batch size 32.

### Detector
- A small CNN with convolutional and max-pooling layers and a final 1 x 1 convolution with sigmoid activation.
- Output is a 16 x 16 grid; each cell predicts 5 values (object presence, centre x and y, width, height).
- The same detector is trained twice, once on noisy images and once on denoised images, using the same labels, and compared with mean IoU.

### Metrics
- **PSNR** and **SSIM** compare denoised images to the clean originals.
- **IoU** measures overlap between predicted and reference boxes.

## Limitations

Please read these before using or citing the numbers.

- **Placeholder labels for detection.** The detector was trained with placeholder ("dummy") YOLO labels so that the noisy and denoised conditions could be compared under identical settings. The IoU values (0.51 and 0.69) therefore show the *relative* effect of denoising in this setup. They are not accuracy on real object annotations.
- **Synthetic noise.** Noise is simulated Gaussian noise (sigma = 0.2), not real observational noise from atmosphere, cosmic rays or sensors.
- **Two stages, not joint training.** The autoencoder and detector are trained one after the other. Joint end-to-end training is future work.
- **Pixel metrics on dark images.** Astronomical images are mostly dark background, so PSNR and SSIM can be high even if faint objects are smoothed away. Inspect the difference images in the notebook for fine detail.
- **Variation across the test set.** PSNR varies widely between images (roughly 10 to 30 dB in the per-image plot), so the average hides difficult cases.
- **Baselines.** The comparison figure gives approximate values only, and baseline training details are not fully documented in this repository.

## Repository structure

```
.
├── research paper/arXiv_paper.tex                              # manuscript source (LaTeX)
├── astronomical-image-processing-and-noise-reduction.ipynb     # notebook: training, evaluation, figures
├── LICENSE
└── README.md
```

## Reproducing the experiments

1. **Environment:** Python 3.8+ and TensorFlow 2.x (tested with TensorFlow 2.10 or newer). The main libraries are TensorFlow/Keras, NumPy and OpenCV.
2. **Data:** the DeepSpaceYOLO dataset is not included in this repository. Download it from its Zenodo release and place the images and YOLO-format annotations in a `data/` folder.
3. **Run:** open `astronomical-image-processing-and-noise-reduction.ipynb` and run the cells in order. It trains the autoencoder, trains the detector on noisy and denoised inputs, and produces the metrics and figures.

## Future work

- Train the denoiser and detector jointly.
- Replace placeholder labels with the real DeepSpaceYOLO annotations and report precision and recall alongside IoU.
- Use realistic noise from real telescope observations.
- Report results over several random seeds, and add ablations by noise level.
- Compare against stronger baselines under identical training settings.

## Citation

```bibtex
@misc{pradhan_denoising_astronomical,
  author = {Pradhan, Adrika},
  title  = {Denoising Astronomical Images with Convolutional Autoencoders to Improve YOLO-Based Object Detection},
  year   = {YEAR},
  note   = {Manuscript source in research paper/arXiv_paper.tex}
}
```

## Acknowledgments

The author thanks the DeepSpaceYOLO dataset team and acknowledges the Luxembourg National Research Fund (FNR grant 15872557) for dataset development.

## License and contact

Released under the MIT License (see [LICENSE](LICENSE)). Questions: adrika.pradhan@usi.ch
