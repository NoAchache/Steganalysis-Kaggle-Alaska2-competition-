# Steganalysis Kaggle Alaska2 Competition 
Pytorch implementation of a steganalysis network as part of the Alaska2 Kaggle competition.


## Introduction
Steganography is the science of hidding messages into images by changing its encoding coefficients in the DCT domain. Such changes slightly alter the pixel values in the spatial domain but are not noticeable by the human eye. Steganalysis is the study of detecting wether an image has been altered through steganography, i.e wheter it contains "stego noise".

The [Alaska2 competiton][alaska] consists in classifying images within two classes: original or altered with steganography. A labelled dataset of 75k series of images is provided. Each serie of images contains an orignal image and three alterations of the original image through three different steganographic methods: UERD, JUNIWARD and JMiPOD. Hence, the dataset contains a total of 75 * 4 = 300k images.

Please note that due to the limited amount of resources available to process such a large dataset, the amount of experiments/runs were very limited.

## Litterature Review

Although a Litterature Review may not be appropriated in a ReadMe, this short section helps to better understand the choices taken for the experiments.

Before the era of deep learning, steganalysis methods involved extracting characteristic features from an image to feed them to a classifier. Hyun et al., use histograms of wavelet subbands extracted from images as an input of an MLP binary classifier [1]. In Pevny et al., the stego noise is exposed with high pass filters and a Markov chain is trained to learn the transition probabilities between adjacent pixel values. Subsets of these transitions probabilities are used as features for a SVM classifier [2]. 

Since stego noise is a high frequency (rapidly changing) noise, using high pass filters increases the SNR, by removing low frequency components, as seen in [2]. The Spatial Rich Model (SRM) is a set of 30 high pass filters of the 1st, 2nd and 3rd order [3]. In many recent papers using CNNs for Steganalysis (which as one could expect, proved to work better than other methods), the first layers are initialized with the values of the SRM filters [4,5,6]. Furthermore, using a trunctation layer for the activation of the SRM filters allows to filter out large elements in the image, which contain no information about the stego noise, leading to a faster training and a higher accuraccy [7].

## First experiment: Denoising Autoencoder (unsucessful: very early convergence)
The first idea was to use a denoising autoencoder, since the orignal image was provided for all altered images (

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
