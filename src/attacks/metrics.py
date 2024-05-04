import numpy as np
import tensorflow as tf


def ssim(img1, img2, max_val=1.0):
    '''Calculate SSIM (Structural Similarity Index Measure) for grayscale images'''
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    # Define necessary constants
    K1 = 0.01
    K2 = 0.03
    L = max_val
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Calculate mean
    mu1 = tf.reduce_mean(img1)
    mu2 = tf.reduce_mean(img2)

    # Calculate variance
    sigma1_sq = tf.reduce_mean(tf.square(img1 - mu1))
    sigma2_sq = tf.reduce_mean(tf.square(img2 - mu2))

    # Calculate covariance
    sigma12 = tf.reduce_mean((img1 - mu1) * (img2 - mu2))

    # Calculate SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return (numerator / denominator).numpy()
def ncc(img1, img2):
    img1 = np.array(img1.squeeze(), dtype=np.float64)
    img2 = np.array(img2.squeeze(), dtype=np.float64)

    # Ensure images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape.")

    # Compute mean of each image
    mean_img1 = tf.reduce_mean(img1)
    mean_img2 = tf.reduce_mean(img2)

    # Compute normalized images
    norm_img1 = img1 - mean_img1
    norm_img2 = img2 - mean_img2

    # Compute NCC numerator
    numerator = tf.reduce_sum(norm_img1 * norm_img2)

    # Compute NCC denominators
    denominator_img1 = tf.reduce_sum(tf.square(norm_img1))
    denominator_img2 = tf.reduce_sum(tf.square(norm_img2))

    # Compute NCC
    ncc_value = numerator / tf.sqrt(denominator_img1 * denominator_img2)
    return ncc_value.numpy()

def psnr(img1, img2):
    # Ensure images are numpy arrays and have dtype=float64
    img1 = np.array(img1, dtype=np.float64)
    img2 = np.array(img2, dtype=np.float64)

    # Calculate mean squared error
    mse = np.mean((img1 - img2) ** 2)

    # Ensure MSE is not zero to avoid division by zero
    if mse == 0:
        return float('inf')

    # Calculate PSNR
    max_pixel = np.max(img1)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr

