# Denoising Astronomical Images with Convolutional Autoencoders to Improve YOLO-Based Object Detection

## Author
[cite_start]Adrika Pradhan [cite: 820]
B.Tech Computer Science (Artificial Intelligence & Machine Learning)
University of Petroleum and Energy Studies
adrika2607@gmail.com

## Abstract
Astronomical imaging is crucial for expanding our knowledge of the universe, enabling the discovery, categorization, and examination of celestial bodies. [cite_start]However, significant noise, caused by atmospheric distortion, cosmic ray interference, and sensor limitations, remains a challenge, particularly for detecting faint or low-contrast objects[cite: 820, 821]. [cite_start]Traditional denoising techniques are computationally efficient but often ineffective in noisy environments and can damage important image features[cite: 822].

[cite_start]This study introduces a novel dual-task deep learning architecture that addresses this limitation by integrating a YOLO-inspired object identification network with a convolutional autoencoder for noise reduction[cite: 823]. [cite_start]The architecture is specifically customized for astronomical images and trained from scratch using the domain-specific DeepSpace YOLO dataset[cite: 824]. [cite_start]The proposed method demonstrates superior performance over baseline approaches, achieving an average PSNR of 27.52 dB, SSIM of 0.88, and a notable increase in object detection accuracy[cite: 825]. [cite_start]This end-to-end pipeline contributes to the development of automated and reliable analysis tools for future astronomical surveys[cite: 826].

## Keywords
[cite_start]Astronomical imaging, deep learning, dual-task learning, convolutional autoencoder, CNN, object detection, YOLO, DeepSpace YOLO dataset, image denoising, noise-aware object detection, celestial object detection, image preprocessing, MSE, PSNR, SSIM, IoU, Adam optimizer, supervised learning, feature extraction, latent space representation, Gaussian noise simulation, real-time detection, bounding box regression, image quality metrics, astronomical object localization. [cite: 827]

## Methodology

[cite_start]The proposed framework integrates two key components: a convolutional autoencoder for image denoising and a YOLO-inspired CNN for object detection[cite: 872, 876]. [cite_start]Both modules are trained from scratch on the DeepSpace YOLO dataset to develop task-specific and noise-aware features, aiming to enhance the quality of astronomical photographs through AI-based noise reduction algorithms[cite: 873, 874, 875].

### Denoising Autoencoder Architecture
[cite_start]The denoising module employs a sophisticated convolutional autoencoder (CAE) based on an encoder-decoder paradigm[cite: 889, 891].
* [cite_start]**Encoder:** Gradually reduces spatial dimensions of noisy input images ($128\times128\times3$ pixels) and extracts hierarchical features using convolutional layers (with ReLU activation) and max-pooling layers[cite: 922, 923, 924, 925, 927]. [cite_start]This ensures computational efficiency, translational invariance, and retention of dominant features[cite: 900, 901, 902, 937, 938, 939, 940, 941].
* [cite_start]**Decoder:** Performs the inverse operation, reconstructing a clean image from the latent representation by progressively increasing spatial dimensions using upsampling layers (e.g., transposed convolutional layers) and refining details with convolutional layers[cite: 903, 904, 905, 906, 945, 946]. [cite_start]The final output layer uses a Conv2D layer with a sigmoid activation function to normalize pixel values to [0, 1][cite: 947, 957].
* [cite_start]**Training:** The autoencoder is trained end-to-end using Mean Squared Error (MSE) as the loss function and the Adam optimizer with a learning rate of $1\times10^{-3}$[cite: 912, 915, 917, 918, 944, 959, 960, 961, 962, 963]. [cite_start]A train-test split of 80-20 was used, with a 10% validation split during training for 20 epochs with a batch size of 32[cite: 887, 970, 972, 974].

### Object Detection Module
[cite_start]The object detection module is inspired by the YOLO (You Only Look Once) architecture, using a CNN backbone to extract features[cite: 999, 1000].
* [cite_start]It consists of multiple convolutional layers and max-pooling layers for feature extraction and spatial downsampling[cite: 1001, 1003, 1015].
* [cite_start]The final detection layer outputs a $16\times16$ spatial grid, where each cell predicts 5 values: class label, center coordinates (x,y), and bounding box width and height[cite: 1006, 1009, 1012, 1014]. [cite_start]A sigmoid activation function is used for object presence prediction[cite: 1016, 1017].
* [cite_start]**Training:** The model was trained comparatively using original noisy images and denoised images (from the autoencoder) with dummy YOLO labels to assess the impact of noise reduction on detection accuracy[cite: 1019, 1020, 1021].

## Dataset
[cite_start]The DeepSpace YOLO dataset [cite: 824, 867] [cite_start]consists of annotated astronomical images in YOLO format, specifically prepared for astronomical object detection, including labels for galaxies, nebulae, and other space structures[cite: 865, 878]. [cite_start]Images were resized to $128\times128$ pixels and artificial Gaussian noise with a standard deviation of 0.2 was added to mimic observational noise[cite: 881, 882].

## Key Results
[cite_start]The method demonstrates improved performance over baseline methods[cite: 825]:
* [cite_start]**Average PSNR:** 27.52 dB [cite: 825]
* [cite_start]**SSIM:** 0.88 [cite: 825]
* [cite_start]**IoU:** 0.69 [cite: 438] (from Table I)
* [cite_start]Significant increase in object detection accuracy[cite: 825].

## Evaluation Metrics
The performance of both denoising and object detection networks was evaluated using:
* [cite_start]**Peak Signal-to-Noise Ratio (PSNR)** [cite: 619]
* [cite_start]**Structural Similarity Index (SSIM)** [cite: 619]
* [cite_start]**Intersection over Union (IoU)** [cite: 620, 621]
