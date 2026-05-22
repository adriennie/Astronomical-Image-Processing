# Denoising Astronomical Images with Convolutional Autoencoders to Improve YOLO-Based Object Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
Short repository README for the accompanying research paper. This repository contains the LaTeX source for the paper, experimental code snippets, and materials to reproduce core results reported in the manuscript.

**Authors:** Adrika Pradhan
## Abstract

This work proposes a dual-task deep learning pipeline that integrates a convolutional autoencoder for image denoising with a YOLO-inspired object detector tailored to astronomical imagery. Experiments on the DeepSpace YOLO dataset show improved image quality (PSNR, SSIM) and higher object detection performance compared to baseline pipelines that omit denoising.
## How to cite

If you use this repository or the method in publications, please cite the paper source in `research paper/arXiv_paper.tex` or the published DOI when available. Example BibTeX (edit fields as appropriate):
```
@article{pradhan2024denoising,
  title={Denoising Astronomical Images with Convolutional Autoencoders to Improve YOLO-Based Object Detection},
## Repository structure

- [research paper/arXiv_paper.tex](research%20paper/arXiv_paper.tex): LaTeX source for the manuscript
- [LICENSE](LICENSE): Project license
- [astronomical-image-processing-and-noise-reduction.ipynb](astronomical-image-processing-and-noise-reduction.ipynb): exploratory notebook

## Reproducibility overview
- **Environment:** Python 3.8+, TensorFlow 2.x (tested with TF 2.10+)
- **Install:** create a virtualenv and install desired packages (example below)

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
- **Data:** This repo does not contain the full DeepSpace YOLO dataset. Place training and annotation files in a `data/` folder following YOLO format.
- **Run training (example):** `python train_autoencoder.py --data data/ --epochs 20`

## Key files and reproduction notes
- `research paper/arXiv_paper.tex`: manuscript source and figures
- `astronomical-image-processing-and-noise-reduction.ipynb`: examples for visualizing denoising and detection results

## Results summary
- Example reported metrics (paper): PSNR 27.52 dB, SSIM 0.88, IoU 0.69. See the manuscript and notebook for full tables and figures.

## Next steps for maintainers
- Add `requirements.txt` with exact package versions
- Add scripts `train_autoencoder.py` and `eval_detection.py` if you want runnable training/evaluation

## License & Contact
This project is MIT licensed — see [LICENSE](LICENSE). For questions contact: adrika2607@gmail.com

# Denoising Astronomical Images with Convolutional Autoencoders to Improve YOLO-Based Object Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![DOI](https://zenodo.org/badge/DOI/YOUR_DOI_HERE.svg)](https://zenodo.org/badge/latestdoi/YOUR_DOI_HERE)

## Author

Adrika Pradhan
B.Tech Computer Science (Artificial Intelligence & Machine Learning)
University of Petroleum and Energy Studies
<adrika2607@gmail.com>

## Abstract

Astronomical imaging is crucial for expanding our knowledge of the universe, enabling the discovery, categorization, and examination of celestial bodies. However, significant noise, caused by atmospheric distortion, cosmic ray interference, and sensor limitations, remains a challenge, particularly for detecting faint or low-contrast objects. Traditional denoising techniques are computationally efficient but often ineffective in highly noisy environments and can damage important image features.

This study introduces a novel dual-task deep learning architecture that addresses this limitation by integrating a YOLO-inspired object detection network with a convolutional autoencoder for noise reduction. The architecture is specifically customized for astronomical images and trained from scratch using the domain-specific DeepSpace YOLO dataset. The proposed method demonstrates superior performance over baseline approaches, achieving an average PSNR of 27.52 dB, SSIM of 0.88, and a notable increase in object detection accuracy. This end-to-end pipeline contributes to the development of automated and reliable analysis tools for future astronomical surveys.

## Keywords

Astronomical imaging, deep learning, dual-task learning, convolutional autoencoder, CNN, object detection, YOLO, DeepSpace YOLO dataset, image denoising, noise-aware object detection, celestial object detection, image preprocessing, MSE, PSNR, SSIM, IoU, Adam optimizer, supervised learning, feature extraction, latent space representation, Gaussian noise simulation, real-time detection, bounding box regression, image quality metrics, astronomical object localization.

## Table of Contents

1. [Introduction](#introduction)
2. [Related Work](#related-work)
3. [Methodology](#methodology)
    * [Denoising Autoencoder Architecture](#denoising-autoencoder-architecture)
    * [Object Detection Module](#object-detection-module)
    * [Training Procedure](#training-procedure)
4. [Dataset](#dataset)
5. [Experimental Setup](#experimental-setup)
6. [Key Results](#key-results)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Conclusion](#conclusion)
9. [Future Work](#future-work)
10. [License](#license)
11. [Acknowledgments](#acknowledgments)
12. [References](#references)

## 1. Introduction
>
> *Provide a comprehensive introduction to the research area, highlighting the significance of astronomical image processing and the challenges associated with noise.  Elaborate on the limitations of existing denoising methods and the motivation behind using a deep learning-based approach.  Clearly state the objectives and contributions of your work.*

## 2. Related Work
>
> *Conduct a thorough review of existing literature on astronomical image denoising and object detection. Discuss traditional methods, as well as recent advances in deep learning for these tasks. Critically analyze the strengths and weaknesses of existing approaches and position your work within the context of the current state-of-the-art.*

## 3. Methodology

The proposed framework integrates two key components: a convolutional autoencoder for image denoising and a YOLO-inspired CNN for object detection. Both modules are trained from scratch on the DeepSpace YOLO dataset to develop task-specific and noise-aware features, aiming to enhance the quality of astronomical photographs through AI-based noise reduction algorithms.

### 3.1. Denoising Autoencoder Architecture

The denoising module employs a sophisticated convolutional autoencoder (CAE) based on an encoder-decoder paradigm.

* **Encoder:** Gradually reduces spatial dimensions of noisy input images ($128\times128\times3$ pixels) and extracts hierarchical features using convolutional layers (with ReLU activation) and max-pooling layers. This ensures computational efficiency, translational invariance, and retention of dominant features. Example layers: `Conv2D(32, (3, 3), activation='relu', padding='same')`, `MaxPooling2D((2, 2), padding='same')`.
* **Decoder:** Performs the inverse operation, reconstructing a clean image from the latent representation by progressively increasing spatial dimensions using upsampling layers (e.g., transposed convolutional layers) and refining details with convolutional layers. The final output layer uses a Conv2D layer with a sigmoid activation function to normalize pixel values to [0, 1]. Example layers: `Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')`, `Conv2D(3, (3, 3), activation='sigmoid', padding='same')`.

  # Encoder

    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

  # Decoder

    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    output_img = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = tf.keras.models.Model(input_img, output_img)
    return autoencoder

### 3.2. Object Detection Module

The object detection module is inspired by the YOLO (You Only Look Once) architecture, using a CNN backbone to extract features.

* It consists of multiple convolutional layers and max-pooling layers for feature extraction and spatial downsampling.
* The final detection layer outputs a $16\times16$ spatial grid, where each cell predicts 5 values: class label, center coordinates (x,y), and bounding box width and height. A sigmoid activation function is used for object presence prediction.
* The loss function typically combines classification loss (cross-entropy) and bounding box regression loss (e.g., MSE or IoU loss).

### 3.3. Training Procedure

* **Denoising Autoencoder Training:** The autoencoder is trained end-to-end using Mean Squared Error (MSE) as the loss function and the Adam optimizer with a learning rate of $1\times10^{-3}$. A train-test split of 80-20 was used, with a 10% validation split during training for 20 epochs with a batch size of 32.  Early stopping can be implemented to prevent overfitting.
* **Object Detection Module Training:**  The model was trained comparatively using original noisy images and denoised images (from the autoencoder) with dummy YOLO labels to assess the impact of noise reduction on detection accuracy.  Consider using transfer learning by pre-training the object detection module on a larger dataset (e.g., COCO) before fine-tuning on the DeepSpace YOLO dataset.
* **Dual-Task Training (Optional):** Explore joint training of the autoencoder and object detection module. This can potentially lead to better performance by allowing the denoising module to learn features that are more relevant for object detection.

## 4. Dataset

The DeepSpace YOLO dataset consists of annotated astronomical images in YOLO format, specifically prepared for astronomical object detection, including labels for galaxies, nebulae, and other space structures. Images were resized to $128\times128$ pixels, and artificial Gaussian noise with a standard deviation of 0.2 was added to mimic observational noise.

> *Provide detailed statistics about the dataset, including the number of images, the number of objects per image, the distribution of object classes, and any data augmentation techniques used.*
>
> *Explain how the Gaussian noise was added (e.g., using NumPy) and justify the choice of the standard deviation value.*

## 5. Experimental Setup

> *Describe the hardware and software used for your experiments. Include details about the GPU, CPU, RAM, and operating system.  Specify the versions of TensorFlow, Keras, and other relevant libraries.*
>
> *Explain the hyperparameter tuning process, including the range of values explored for learning rate, batch size, and other hyperparameters. Justify the final choice of hyperparameters based on validation performance.*

## 6. Key Results

The method demonstrates improved performance over baseline methods:

* **Average PSNR:** 27.52 dB
* **SSIM:** 0.88
* **IoU:** 0.69 (from Table I)
* Significant increase in object detection accuracy.

> *Present your results in a clear and concise manner, using tables and figures to illustrate the performance of your method. Compare your results with baseline methods and discuss the statistical significance of the improvements.*
>
> *Include examples of denoised images and object detection results to visually demonstrate the effectiveness of your approach.*
>
> *Create a table summarizing the performance metrics for different configurations of your model (e.g., with and without denoising, different noise levels).*
>
> *Discuss any limitations of your method and potential areas for improvement.*

## 7. Evaluation Metrics

The performance of both denoising and object detection networks was evaluated using:

* **Peak Signal-to-Noise Ratio (PSNR):** A measure of the quality of the reconstructed image compared to the original image. Higher PSNR values indicate better image quality. Defined as:

> *Provide a more detailed explanation of each evaluation metric and its relevance to your research problem. Discuss the advantages and disadvantages of each metric and justify your choice of metrics.*
>
> *Explain how the metrics were calculated and any pre-processing steps that were applied to the images or bounding boxes.*

## 8. Conclusion

> *Summarize the key findings of your research and highlight the contributions of your work. Discuss the implications of your results for the field of astronomical image processing and object detection. Reiterate the advantages of your proposed method over existing approaches.*

## 9. Future Work

> *Outline potential directions for future research, such as exploring different autoencoder architectures, using more sophisticated noise models, or applying your method to other types of astronomical images. Suggest ways to improve the performance of your method and address its limitations. Consider investigating the use of generative adversarial networks (GANs) for denoising.*

## 10. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 11. Acknowledgments

> *Acknowledge any individuals or organizations that provided support for your research. This may include funding sources, collaborators, or advisors.*

## 12. References
