# Steganalysis Kaggle Alaska2 Competition 
Pytorch implementation of a steganalysis network as part of the Alaska2 Kaggle competition.


## Introduction
Steganography is the science of hidding messages into images by changing its encoding coefficients in the DCT domain. Such changes slightly alter the pixel values in the spatial domain but are not noticeable by the human eye. Steganalysis is the study of detecting wether an image has been altered through steganography, i.e wheter it contains "stego noise".

The [Alaska2 competiton][alaska] consists in classifying images within two classes: original or altered with steganography. A labelled dataset of 75k series of images is provided. Each serie of images contains an orignal image and three alterations of the original image through three different steganographic methods: UERD, JUNIWARD and JMiPOD. Hence, the dataset contains a total of 75 * 4 = 300k images.

Please note that due to the limited amount of resources available to process such a large dataset, the amount of experiments/runs were very limited.

## Litterature Review
Before the era of deep learning, steganalysis methods involved extracting characteristic features from an image to feed them to a classifier. Hyun et al., use histograms of wavelet subbands extracted from images as an input of an MLP binary classifier [1]. In Pevny et al., the stego noise is exposed with high pass filters and a Markov chain is trained to learn the transition probabilities between adjacent pixel values. Subsets of these transitions probabilities are used as features for a SVM classifier [2]. Using high pass filters proved to be an efficient manner to increase the signal to noise ratio of steganalyser since stego noise is a high frequency (rapidly changing) noise. 

## First experiment: Denoising Autoencoder (unsucessful: very early convergence)
The first idea was to use a denoising autoencoder, since the orignal image was provided for all altered images (

## References
[1] Hyun, S. H., Park, T. H., Jeong, B. G., Kim, Y. S., & Eom, I. K. (2010). Feature Extraction for Steganalysis using Histogram Change of Wavelet Subbands. In Proceeding of The 25th International Technical Conference on Circuits/Systems, Computers and Communications (ITC-CSCC 2010), Pattaya, Thailand, JULY (pp. 4-7).

[2] Pevny, T., Bas, P., & Fridrich, J. (2010). Steganalysis by subtractive pixel adjacency matrix. IEEE Transactions on information Forensics and Security, 5(2), 215-224.

[autoencoder]: https://github.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/tree/master/unused

[alaska]: https://www.kaggle.com/c/alaska2-image-steganalysis
