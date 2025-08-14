### 10/15/2023: Make new versions of the model functions that train with Datasets - first attempt failed
import numpy as np
import scipy.ndimage
# lenses with centralized positions for use in task-specific estimations
def load_single_lens_uniform(size=32, sigma=0.8):
    one_lens = np.zeros((size, size))
    one_lens[16, 16] = 1
    one_lens = scipy.ndimage.gaussian_filter(one_lens, sigma=sigma)
    one_lens /= np.sum(one_lens)
    return one_lens

def load_two_lens_uniform(size=32, sigma=0.8):
    two_lens = np.zeros((size, size))
    two_lens[16, 16] = 1 
    two_lens[7, 9] = 1
    two_lens = scipy.ndimage.gaussian_filter(two_lens, sigma=sigma)
    two_lens /= np.sum(two_lens)
    return two_lens

def load_three_lens_uniform(size=32, sigma=0.8):
    three_lens = np.zeros((size, size))
    three_lens[16, 16] = 1
    three_lens[7, 9] = 1
    three_lens[23, 21] = 1
    three_lens = scipy.ndimage.gaussian_filter(three_lens, sigma=sigma)
    three_lens /= np.sum(three_lens)
    return three_lens

def load_four_lens_uniform(size=32, sigma=0.8):
    four_lens = np.zeros((size, size))
    four_lens[16, 16] = 1
    four_lens[7, 9] = 1
    four_lens[23, 21] = 1
    four_lens[8, 24] = 1
    four_lens = scipy.ndimage.gaussian_filter(four_lens, sigma=sigma)
    four_lens /= np.sum(four_lens)
    return four_lens
def load_five_lens_uniform(size=32, sigma=0.8):
    five_lens = np.zeros((size, size))
    five_lens[16, 16] = 1
    five_lens[7, 9] = 1
    five_lens[23, 21] = 1
    five_lens[8, 24] = 1
    five_lens[21, 5] = 1
    five_lens = scipy.ndimage.gaussian_filter(five_lens, sigma=sigma)
    five_lens /= np.sum(five_lens)
    return five_lens

def load_six_lens_uniform(size=32, sigma=0.8):
    six_lens = np.zeros((size, size))
    six_lens[16, 16] = 1
    six_lens[7, 9] = 1
    six_lens[23, 21] = 1
    six_lens[8, 24] = 1
    six_lens[21, 5] = 1
    six_lens[27, 13] = 1 
    six_lens = scipy.ndimage.gaussian_filter(six_lens, sigma=sigma)
    six_lens /= np.sum(six_lens)
    return six_lens

def load_seven_lens_uniform(size=32, sigma=0.8):
    seven_lens = np.zeros((size, size))
    seven_lens[16, 16] = 1
    seven_lens[7, 9] = 1
    seven_lens[23, 21] = 1
    seven_lens[8, 24] = 1
    seven_lens[21, 5] = 1
    seven_lens[27, 13] = 1 
    seven_lens[4, 16] = 1
    seven_lens = scipy.ndimage.gaussian_filter(seven_lens, sigma=sigma)
    seven_lens /= np.sum(seven_lens)
    return seven_lens

def load_eight_lens_uniform(size=32, sigma=0.8):
    eight_lens = np.zeros((size, size))
    eight_lens[16, 16] = 1
    eight_lens[7, 9] = 1
    eight_lens[23, 21] = 1
    eight_lens[8, 24] = 1
    eight_lens[21, 5] = 1
    eight_lens[27, 13] = 1 
    eight_lens[4, 16] = 1
    eight_lens[16, 26] = 1
    eight_lens = scipy.ndimage.gaussian_filter(eight_lens, sigma=sigma)
    eight_lens /= np.sum(eight_lens)
    return eight_lens

def load_nine_lens_uniform(size=32, sigma=0.8):
    nine_lens = np.zeros((size, size))
    nine_lens[16, 16] = 1
    nine_lens[7, 9] = 1
    nine_lens[23, 21] = 1
    nine_lens[8, 24] = 1
    nine_lens[21, 5] = 1
    nine_lens[27, 13] = 1
    nine_lens[4, 16] = 1
    nine_lens[16, 26] = 1
    nine_lens[14, 7] = 1
    nine_lens = scipy.ndimage.gaussian_filter(nine_lens, sigma=sigma)
    nine_lens /= np.sum(nine_lens)
    return nine_lens