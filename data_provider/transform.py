import cv2
import numpy as np


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for t in self.transforms:
            imgs = t(imgs)
        return imgs


class RandomMirror(object):
    def __call__(self, images):
        if np.random.randint(2):
            for i, image in enumerate(images):
                images[i] = np.ascontiguousarray(image[:, ::-1])
        return images


class PreProcessing(object):
    def __call__(self, images):
        for i, image in enumerate(images):
            image = image.astype(np.float32)
            image /= 255.0
            images[i] = image.transpose(2, 0, 1)
        return images


class ResizeAndPadding(object):
    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, images):
        for i, image in enumerate(images):
            h, w, _ = image.shape

            # resize image until either its height or width matches the dimensions of the output image

            if h / w > self.sizes[1] / self.sizes[0]:
                w_new = int(w * self.sizes[1] / h)
                h_new = self.sizes[1]
                image = cv2.resize(image, dsize=(w_new, h_new))

            elif w / h > self.sizes[0] / self.sizes[1]:
                h_new = int(h * self.sizes[0] / w)
                w_new = self.sizes[0]
                image = cv2.resize(image, dsize=(w_new, h_new))

            else:
                w_new, h_new = self.sizes[0], self.sizes[1]
                image = cv2.resize(image, dsize=(w_new, h_new))

            max_w_i = np.max([w_new, self.sizes[0]])
            max_h_i = np.max([h_new, self.sizes[1]])

            # Pad the image to match the input size if it is smaller
            im_padded = np.zeros((max_h_i, max_w_i, 3), dtype=np.uint8)
            im_padded[:h_new, :w_new, :] = image.copy()
            images[i] = im_padded

        return images


class BaseTransform(object):
    def __init__(self, sizes):
        self.base_transform = Compose([
            ResizeAndPadding(sizes),
            PreProcessing()
        ])

    def __call__(self, images):
        return self.base_transform(images)

class AugmentatedTransform(object):
    def __init__(self, sizes):
        self.augmentation = Compose([
            RandomMirror(),
            ResizeAndPadding(sizes),
            PreProcessing()
        ])

    def __call__(self, images):
        return self.augmentation(images)