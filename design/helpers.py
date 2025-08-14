import numpy as np
import struct
from array import array
from os.path  import join
import matplotlib.pyplot as plt
import torch
from scipy import stats
import cleanplots

# From Kaggle: https://www.kaggle.com/code/hojjatk/read-mnist-dataset
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels  


# Helper function to show a list of images with their relating titles
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

def make_random_mnist_tiles(images, grid_size=3, num_grids=10):
    """
    images: np.ndarray of shape [batch, 32, 32]
    grid_size: number of tiles per side (3 for 3x3)
    num_grids: number of mosaics to make
    returns: np.ndarray of shape [num_grids, grid_size*32, grid_size*32]
    """
    tile_size = images.shape[1]  # should be 32
    out_imgs = np.zeros((num_grids, grid_size*tile_size, grid_size*tile_size), dtype=images.dtype)

    for g in range(num_grids):
        # Pick random images without replacement
        idxs = np.random.choice(images.shape[0], size=grid_size**2, replace=False)
        chosen = images[idxs]  # shape [grid_size^2, 32, 32]

        # Arrange into a grid
        grid_rows = []
        for r in range(grid_size):
            row_imgs = chosen[r*grid_size:(r+1)*grid_size]
            row_cat = np.concatenate(row_imgs, axis=1)  # concat along width
            grid_rows.append(row_cat)
        full_img = np.concatenate(grid_rows, axis=0)  # concat rows along height

        out_imgs[g] = full_img

    return out_imgs

def make_image_tiles_torch(images, grid_size=3, num_grids=100):
    """
    images: tensor [N, H, W]
    Returns: tiled images [num_grids, grid_size*H, grid_size*W]
    """
    batch, H, W = images.shape
    out_imgs = torch.zeros((num_grids, grid_size*H, grid_size*W), dtype=images.dtype)

    for g in range(num_grids):
        # Pick random images without replacement
        idxs = torch.randperm(batch)[:grid_size**2]
        chosen = images[idxs]  # [grid_size**2, H, W]

        # Arrange into a grid
        rows = []
        for r in range(grid_size):
            row = torch.cat([img for img in chosen[r*grid_size:(r+1)*grid_size]], dim=1)  # concat width
            rows.append(row)
        full_img = torch.cat(rows, dim=0)  # concat height
        out_imgs[g] = full_img

    return out_imgs
 
# def compute_mse(img1, img2):
#     return np.mean((np.astype(img1, np.float32) - np.astype(img2, np.float32)) ** 2)

# def compute_PSNR(ground_truth, img, max_val=255):
#     # find mse
#     mse = compute_mse(ground_truth, img)
    
#     # mse=0 means no noise
#     if mse == 0:
#         return 100
#     return 20 * np.log10(max_val / np.sqrt(mse))  

def calculate_95_ci(data):
    n_lenslets, n_samples = data.shape
    means = data.mean(axis=1)
    stderr = data.std(axis=1, ddof=1) / np.sqrt(n_samples)

    t_crit = stats.t.ppf(0.975, df=(n_samples - 1))
    lower_bound = means - t_crit * stderr
    upper_bound = means + t_crit * stderr

    return means, lower_bound, upper_bound

def plot_metrics_with_ci(metric_list, metric_names):
    """
    Plots mean and 95% CI for multiple metrics side by side.
    
    metric_list: list of arrays, each of shape (n_lenslets, n_samples)
    metric_names: list of strings, names of each metric
    """
    n_metrics = len(metric_list)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]  # ensure axes is iterable

    for i, data in enumerate(metric_list):
        means, lower, upper = calculate_95_ci(data)
        x = np.arange(len(means))
        
        axes[i].plot(x, means, color='blue', label='Mean')
        axes[i].fill_between(x, lower, upper, color='blue', alpha=0.2, label='95% CI')
        axes[i].set_title(metric_names[i])
        axes[i].set_xlabel("Lenslet index")
        axes[i].set_ylabel(metric_names[i])
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def plot_metric_grid(metric_array, tv_vals, ylabel, title_prefix):
    """
    metric_array: shape (9 lenslets, 8 tv values, 100 samples)
    tv_vals: list of TV parameter values
    ylabel: string label for y-axis
    title_prefix: string for subplot titles
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.ravel()

    for lens_idx in range(9):
        ax = axes[lens_idx]
        data = metric_array[lens_idx]  # shape (8, 100)

        mean_vals, lower_vals, upper_vals = calculate_95_ci(data)

        ax.plot(tv_vals, mean_vals, marker='o', color='blue', label="Mean")
        ax.fill_between(tv_vals, lower_vals, upper_vals,
                        color='blue', alpha=0.2, label="95% CI")

        ax.set_xscale('log')
        ax.set_xlabel("TV Value")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} - Lenslet {lens_idx}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(ylabel + '.png', dpi=300)
    plt.show()