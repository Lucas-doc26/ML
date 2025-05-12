import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import tensorflow.image as tf_img

def calculete_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def calculete_ssim(image1, image2):
    return tf_img.ssim(image1, image2, max_val=1.0).numpy()

def calculete_psnr(image1, image2):
    return tf.image.psnr(image1, image2, max_val=1.0)

def calculete_ncc(image1, image2):
    image1_mean = np.mean(image1)
    image2_mean = np.mean(image2)
    numerator = np.sum((image1 - image1_mean) * (image2 - image2_mean))
    denominator = np.sqrt(np.sum((image1 - image1_mean)**2) * np.sum((image2 - image2_mean)**2))
    return numerator / denominator