# Steganalysis Kaggle Alaska2 Competition 
Pytorch implementation of a steganalysis network as part of the Alaska2 Kaggle competition.


## Introduction
Steganography is the science of hiding messages into images by changing its encoding coefficients in the DCT domain. Such changes slightly alter the pixel values in the spatial domain but are not noticeable by the human eye. Steganalysis is the study of detecting whether an image has been altered through steganography, i.e whether it contains "stego noise".

The [Alaska2 competiton][alaska] consists in classifying images within two classes: original or altered through steganography. A labelled dataset of 75k 'series' of images is provided. Each 'series' of images contains an original image (unaltered) and three alterations of the original image, via three steganographic methods: UERD, JUNIWARD and JMiPOD. Hence, the dataset contains a total of 75 * 4 = 300k images. The Testset is composed of 10k images. The competition's metric is a weighted AUC (more details [here][weighted_AUC])

Please note that due to the limited amount of resources available to process such a large dataset, the number of experiments/runs were very limited.

## Litterature Review

Before the era of deep learning, steganalysis methods involved extracting characteristic features from an image to feed them to a classifier. Hyun et al. use histograms of wavelet subbands extracted from images as an input of an MLP binary classifier [1]. In Pevny et al., the stego noise is exposed with high pass filters and a Markov chain is trained to learn the transition probabilities between adjacent pixel values. Subsets of these transitions probabilities are used as features for an SVM classifier [2]. 

Since stego noise is a high frequency (rapidly changing) noise, the use of high pass filters increases the Signal to Noise Ratio (SNR), by removing low-frequency components, as seen in [2]. The Spatial Rich Model (SRM) is a set of 30 high pass filters of the 1st, 2nd and 3rd order [3]. In many recent papers using CNNs for Steganalysis (which as one could expect, proved to work better than other methods), the SRM filters are used as the first layer [4,5,6]. Furthermore, using a truncation layer for the activation of the SRM filters allows to filter out large elements in the image, which contain no information about the stego noise, leading to a faster training and a higher accuracy overall [7]. The most commonly used truncation layer is the Linear Truncation Unit (TLU) which is defined as follows [6]:

<img src="https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/blob/master/ReadMe_imgs/TLU.png" width="200" height="70">

Where T is the truncation threshold (hyperparameter). Wu et al. propose an alternative to the TLU in [6], the Single-Valued Truncation (STL) leading to better results with their network:

<img src="https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/blob/master/ReadMe_imgs/STL.png" width="200" height="50">

The idea behind the STL is that since the truncated values provide useless information, setting them to the same value allows reducing the total variance. 


## Model

The model is composed of the 30 SRM filters, followed by a truncation layer and a pretrained EfficientNet-B2. Although the SRM layer consists in 'only' 30 filters, they increased the number of FLOPS of the overall model by half, compared to merely using an efficient net, as detailed in this [file][flops].

The choice of the truncation layer is detailed in another section. Using a binary classification led to bad accuracies, so 4 classes are outputted instead (original, UERD, JUNIWARD and JMiPOD). The loss function is the negative log-likelihood loss.

## Training Specs

Images are augmented using horizontal flips with probability 0.5. A single GPU is used: NVIDIA RTX 2080 Ti. 80% of the data are used for training and 20% for validating. The data are shuffled prior to the train/validation split to ensure a homogenous repartition.

The learning rates are initially set to 0.001 for the Efficient Net and to a much smaller value, 0.00001, for the SRM layer, to impede the SRM filters from straying too far from their original values. The lrs are automatically adjusted by means of a plateau scheduler, with a decreasing factor of 0.5, a patience of 1, and using the validation loss as the tracked metric.

At first, it was attempted to train the model using the four variants of each image within the same batch. However, this led to very high accuracies for the trainset, and near randomness for the validation set. Our interpretation is this problem originates from the fact that the batch norm layers use the means and variances of the batch during training: since the four variants of the same image are very similar, and hence, have very similar means/variances, normalising them using their own means/variances led to removing the useless information, which significantly increased the SNR, by exposing the stego noise (explaining the very high accuracies for the trainset). However, when setting the model to eval mode, the batch norm layers use the running means/variances computed during training and therefore, it led to poor results. In other words, the network did not learn to find if an image contains stego noise but rather given several variants of the same image (including the original), find which ones contain noise.

To address this issue, the batches are split into 4 sub-batches, each sub-batch containing only one version of each image. To improve the training, the model's weights are updated only after inferring all 4 sub-batches, so that the gradients are affected by the four versions of each image. A batch of 4 is used, leading to 4 (batch size) * 4 (sub-batch size) = 16 images per update. However, doing so is more memory consuming compared to simply using a batch of 16, leading to GPU out of memory error. This was remedied using the [large scale model][lsm] module of IBM, which swaps automatically unused tensors from the GPU to the CPU memory, at the cost of time. Another downside of such an approach is that the batch norm layers were using means/variances computed on a smaller batch (hence less representative).

## Truncation Layer

The model is trained over one epoch with different truncation layers and hyperparameters. The assumption that two variants of the model (with different parameters) would compare the same when training for one epoch and for many epochs is made, considering the limited amount of time and resources available for this project. To ensure repeatability between the experiments, the dataset is shuffled with a seed fixed to 42 before the train/validation split, without further shuffling. Figure 1 shows the evolution of the training losses and weighted AUCs over one epoch. *stlX* and *tluX* correspond to the truncation layers described in the Literature Review, with a truncation threshold set to X. The Xs values experimented were ranging from 5 to 11 as commonly used in papers [5,6,7]. Two additional runs are also present: without a truncation layer (*noTruncation*) which is equivalent to setting the truncation threshold to +∞, and without the SRM layer (*noSRM*).

|Loss|Weighted AUC|Legend|
|--|--|--|
| <img src="https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/blob/master/ReadMe_imgs/1epoch_loss.svg" width="400" height="200"> | <img src="https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/blob/master/ReadMe_imgs/1epoch_weighted_AUC.svg" width="400" height="200"> | <img src="https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/blob/master/ReadMe_imgs/legend.png" width="100" height="120"> |
<p align="center">
  <i>
  Figure 1: Results of different variants of the model after 1 epoch
  </i>
</p>
  
Only one run is performed with the STL since it leads to the worst results by far of all runs. However, as expected, using a tlu provides an obvious advantage over the absence of truncation layer or SRM layer. The different thresholds used for the tlu provide very similar results: with more resources/time, it would have been interesting to pursue these runs over more epochs to observe a difference. Since it is suggested in [7] that the threshold should not be larger than 8, the tlu11 was rejected. The tlu8 is chosen over the tlu5, to ensure no useful information would be truncated.

## Results

The results of the training are shown in Figure 2. The score (Weighted AUC) on the Testset is 0.879.

|Training Loss|Training Weighted AUC|Learning Rate|
|--|--|--|
| <img src="https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/blob/master/ReadMe_imgs/train_loss.svg" width="400" height="200"> | <img src="https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/blob/master/ReadMe_imgs/train_weighted_AUC.svg" width="400" height="200"> | <img src="https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/blob/master/ReadMe_imgs/val_lr.svg" width="400" height="200"> |

|Validation Loss|Validation Weighted AUC|
|--|--|
| <img src="https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/blob/master/ReadMe_imgs/val_loss.svg" width="400" height="200"> | <img src="https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/blob/master/ReadMe_imgs/val_weighted_AUC.svg" width="400" height="200"> |

<p align="center">
  <i>
  Figure 2: Training and validation results
  </i>
</p>

With more resources, it would have been interesting to train the model more than one time and alter the truncation layer, truncation threshold, learning rate scheduling (seems to be decreasing a bit too fast in our training), batches (batch of 16 instead of 4 sub-batches), etc...

## Unsuccessful experiment: Denoising Autoencoder

*Description of the very first experiment, which unfortunately led to poor results. The remaining pieces of code about this can be found in [unused][unused]*

Since for all altered images, the original image is provided, the first idea was to use a denoising autoencoder (DAE) to remove the stego noise. The DAE was taking as input an original or an altered image and the output was compared to the original image for the computation of the loss function. Hence, one could expect the DAE to consistently output an image without stego noise. Consequently, the input image steganalysis could be performed by comparing it to the output. Two different approaches were experimented. In the first one, the output image was substracted to the input of the DAE to expose the stego noise, and the resulting image was passed through an efficient net. In the second approach, the input and the output were compared via a Siamese network. It was also tried to first use an SRM layer, and pass all 30 channels to the DAE.

When visualising the data, it was noticed that the noise left out by the DAE was about 10 times larger than the stego noise, which explains the poor classification results. Furthermore, removing the stego noise from an image is actually a much more complex problem than merely telling whether it is there or not.

## References
[1] Hyun, S. H., Park, T. H., Jeong, B. G., Kim, Y. S., & Eom, I. K. (2010). Feature Extraction for Steganalysis using Histogram Change of Wavelet Subbands. In Proceeding of The 25th International Technical Conference on Circuits/Systems, Computers and Communications (ITC-CSCC 2010), Pattaya, Thailand, JULY (pp. 4-7).

[2] Pevny, T., Bas, P., & Fridrich, J. (2010). Steganalysis by subtractive pixel adjacency matrix. IEEE Transactions on information Forensics and Security, 5(2), 215-224.

[3] Fridrich, J., & Kodovsky, J. (2012). Rich models for steganalysis of digital images. IEEE Transactions on Information Forensics and Security, 7(3), 868-882.

[4] Xu, X., Sun, Y., Tang, G., Chen, S., & Zhao, J. (2016, September). Deep learning on spatial rich model for steganalysis. In International Workshop on Digital Watermarking (pp. 564-577). Springer, Cham.

[5] Ye, J., Ni, J., & Yi, Y. (2017). Deep learning hierarchical representations for image steganalysis. IEEE Transactions on Information Forensics and Security, 12(11), 2545-2557.

[6] Wu, S., Zhong, S. H., Liu, Y., & Liu, M. (2019). CIS-Net: A Novel CNN Model for Spatial Image Steganalysis via Cover Image Suppression. arXiv preprint arXiv:1912.06540.

[7] Lu, Y. Y., Yang, Z. L. O., Zheng, L., & Zhang, Y. (2019, September). Importance of Truncation Activation in Pre-Processing for Spatial and Jpeg Image Steganalysis. In 2019 IEEE International Conference on Image Processing (ICIP) (pp. 689-693). IEEE.

[autoencoder]: https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/tree/master/unused

[alaska]: https://www.kaggle.com/c/alaska2-image-steganalysis

[weighted_AUC]: https://www.kaggle.com/c/alaska2-image-steganalysis/overview/evaluation

[flops]: https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/blob/master/utils/SRM_flops_calculator.py

[lsm]: https://github.com/IBM/pytorch-large-model-support

[unused]: https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/tree/master/unused
