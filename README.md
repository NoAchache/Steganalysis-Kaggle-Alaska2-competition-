# Steganalysis Kaggle Alaska2 Competition 
Pytorch implementation of a steganalysis network as part of the Alaska2 Kaggle competition.


## Introduction
Steganography is the science of hidding messages into images by changing its encoding coefficients in the DCT domain. Such changes slightly alter the pixel values in the spatial domain but are not noticeable by the human eye. Steganalysis is the study of detecting wether an image has been altered through steganography, i.e wheter it contains "stego noise".

The [Alaska2 competiton][alaska] consists in classifying images within two classes: original or altered with steganography. A labelled dataset of 75k series of images is provided. Each serie of images contains an orignal image and three alterations of the original image through three different steganographic methods: UERD, JUNIWARD and JMiPOD. Hence, the dataset contains a total of 75 * 4 = 300k images. The Testset is composed of 10k images.

Please note that due to the limited amount of resources available to process such a large dataset, the amount of experiments/runs were very limited.

## Litterature Review

*Although a Litterature Review may not be appropriated in a ReadMe, this short section helps to better understand the choices taken for the experiments.*

Before the era of deep learning, steganalysis methods involved extracting characteristic features from an image to feed them to a classifier. Hyun et al., use histograms of wavelet subbands extracted from images as an input of an MLP binary classifier [1]. In Pevny et al., the stego noise is exposed with high pass filters and a Markov chain is trained to learn the transition probabilities between adjacent pixel values. Subsets of these transitions probabilities are used as features for a SVM classifier [2]. 

Since stego noise is a high frequency (rapidly changing) noise, using high pass filters increases the SNR, by removing low frequency components, as seen in [2]. The Spatial Rich Model (SRM) is a set of 30 high pass filters of the 1st, 2nd and 3rd order [3]. In many recent papers using CNNs for Steganalysis (which as one could expect, proved to work better than other methods), the first layers are initialized with the values of the SRM filters [4,5,6]. Furthermore, using a trunctation layer for the activation of the SRM filters allows to filter out large elements in the image, which contain no information about the stego noise, leading to a faster training and a higher accuraccy [7]. The most commonly used is the Linear Truncation Unit (TLU) which is defined as follows [6]:

<img src="https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/blob/master/ReadMe_imgs/TLU.png" width="200" height="70">

Wu et al. propose an alternative to the TLU in [6], the Single-Valued Truncation (STL) leading to better results:

<img src="https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/blob/master/ReadMe_imgs/STL.png" width="200" height="50">

The idea behind the STL is that since the truncated values provide useless information, setting them to the same value allows to reduce the total variance. 


## Model

The model is composed of the 30 SRM filters, followed by a truncation layer and a pretrained EfficientNet-B2. The choice of the truncation layer is detailled in another section. Using a binary classification led to bad accuraccies, so 4 classes were outputted instead (orignal, UERD, JUNIWARD and JMiPOD). The negative log likelihood loss function was used.

## Training Specs

Images were augmented using simple horizontal flips with a probability 0.5. A single gpu was used: NVIDIA RTX 2080 Ti.

Scheduler and lr

At first, it was attempted to train the model using the four variants of each cover images within the same batch. However, this led to very high accuracies for the trainset, and near randomness for the validation set. My interpretation is this problem originates from the fact that the batch norm layers use the means and variances of the batch during training. Since the four variants of the same image are very similar, and hence, have very similar means/variances, normalising them using their own means/variances led to removing all the useless information, which significantely increased the SNR, by exposing the stego noise (explaining the very high accuracies for the trainset). However, when setting the model to eval mode, the batch norm layers use the running means/variances computed during training, and therefore gives poor results. In other words, the network did not learn to find if an image contains stego noise, but rather given several variants of the same image (including the original), find which ones contain noise.

To adress this issue, the batches were splitted into 4 sub-batches, each sub-batch containing only one version of each image. To improve the training, the model's weights were updated only after infering all 4 sub-batches, so that the gradients were affected by the four versions of each image. A batch of 4 was used, leading to 4 (batch size) * 4 (sub-batch size) = 16 images per update. However, doing so was more memory consuming than simply using a batch of 16 leading to gpu out of memory error. This was remedied using the [large scale model][lsm] module of IBM, swapping automatically unused tensors from the gpu to the cpu memory (at the cost of time). Another downside of such approach is that the batch norm layers were using means/variances computed on a smaller batch (hence less representative).


## Unsuccessful experiment: Denoising Autoencoder (unsucessful: very early convergence)

Since for all altered images, the original image was provided, the first idea was to use a denoising autoencoder (DAE) to remove the stego noise. The DAE was taking as input an orginal or an altered image. The output was compared to the original image for the computation of the loss function. Hence, one could expect the DAE to consistentely output an image without stego noise. Therefore, the input image steganalysis could be performed by comparing it to the output. Two different approaches were experimented:

noise much bigger than stego (10 times larger)
more complex problem

code in unused

Several variants:
-
-
-



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

[lsm]: https://github.com/IBM/pytorch-large-model-support
