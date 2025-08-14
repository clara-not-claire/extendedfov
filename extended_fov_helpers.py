"""
helper functions for extended fov.

includes:
- zero_pad
- normalize
- convolve_rgb
- convolve_mono
- center_crop
- forward
- load
- rescale_img

- get_extent_increase
- generate_edges
- get_midpoints
- get_widthpoints
- get_coords
- plot_extents

- compute_tamura

- check_coords
- plot_rml_irs
- generate_rml_irs
- save_final_im
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from skimage.transform import rescale, resize
from skimage import color
import skimage.feature
from scipy.ndimage import sobel
from skimage.color import rgb2gray

from PIL import Image as im 
import os
import cv2
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import Slider, Button, RadioButtons
import random
from ipywidgets import interact, interact_manual
import ipywidgets as widgets

# import recon.load_scripts_extended as loading
# from recon.helper_fns_extended import *

WHITE = (255, 255, 255)
SEED = 8
psf_index = 0
random.seed(SEED)


def zero_pad(img, ref = (0, 0)):
    im_shape = img.shape
    if type(ref) == tuple:
        ref_shape = ref
    else:
        ref_shape = ref.shape

    ax0_pad = np.abs(im_shape[0] - ref_shape[0]) // 2
    ax1_pad = np.abs(im_shape[1] - ref_shape[1]) // 2

    ## grayscale
    if len(im_shape) == 2:
        return np.pad(img, ((ax0_pad, ax0_pad), (ax1_pad, ax1_pad)))
    return np.pad(img, ((ax0_pad, ax0_pad), (ax1_pad, ax1_pad), (0, 0)))

def normalize(img):
    return img / np.linalg.norm(img.ravel()) #l1 or l2?

def convolve_rgb(img, psf, channels=3, mode='full'):
    img_lst = []
    for i in range(channels):
        img_lst.append(sig.convolve(img[:, :, i], psf[:, :, i], mode=mode))
    img_lst = np.array(img_lst)
    img_lst = np.stack(img_lst, axis=2)
    return img_lst

def convolve_mono(img, psf, mode='full'):
    return sig.convolve(img, psf, mode=mode)

def center_crop(img, dim=(0, 0)):
    # (x, y, channels)
    x, y = img.shape[0:2]
    new_x, new_y = dim[0:2]
    x_start = x//2 - new_x//2
    y_start = y//2 - new_y//2

    if len(img.shape) == 2:
        return img[x_start:x_start + new_x, y_start:y_start + new_y]
    
    return img[x_start:x_start + new_x, y_start:y_start + new_y, :]


def forward(img, psf, psf_name='psf', gray=False, name='', save_path='.', crop=True):
    # print("Input Image Dim: ", img.shape)
    # print("PSF Dim: ", psf.shape)
    if len(img.shape) > 2:
        img = img[:, :, 0:3]

    if len(psf.shape) > 2:
        psf = psf[:, :, 0:3]
    
    # try WIHTOUT zeropadding, because zero-padding affects the image distribution
    # draw a flow for myself to visualize the process
    img_padded = img #zero_pad(img, psf)
    psf = normalize(psf)

    # if scale:
    #     img_padded = rescale(img_padded, rescale, anti_aliasing=True, channel_axis=2)
    # # if resize:
    # #     img_padded = resize(img_padded, (resize, resize, 3), anti_aliasing=True)
    
    ## TODO: convolve 2d
    if gray:
        if len(img_padded.shape) > 2:
            img_padded = color.rgb2gray(img_padded)
        if len(psf.shape) > 2:
            psf = color.rgb2gray(psf)
        convolved_img = convolve_mono(img_padded, psf, mode='valid')
    else:
        convolved_img = convolve_rgb(img_padded, psf)
    if crop:
        cropped_img = center_crop(convolved_img, dim=psf.shape)
    else: 
        cropped_img = convolved_img

    cropped_img = cropped_img / np.max(cropped_img)

    # print(np.min(cropped_img), np.max(cropped_img))

    # print("Padded Image Dim: ", img_padded.shape)
    # print("Convolved Image Dim: ", convolved_img.shape)
    # print("Cropped Image Dim: ", cropped_img.shape)
    f_name = f'{name}-{psf_name}-measurement'

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    img_save_path = f'{save_path}/{f_name}.npy'
    img_save_path_png = f'{save_path}/{f_name}.png'

    np.save(img_save_path, cropped_img)

    plt.imsave(img_save_path_png, cropped_img)
    # print(f"Saved at {img_save_path}")
    return cropped_img, img_save_path, f_name

def load(path, psf='./3-6-24_rml_mono8_3000.tiff', gray=True):
    if path.endswith('.npy'):
        img = np.load(path)
    else:
        img = plt.imread(path)
    if gray:
        psf = plt.imread(psf)
    else:
        psf= plt.imread('./2-27-24_rml_rgb.tiff')
    # print("Ground Truth Shape:", img.shape)
    # print("PSF Shape:", psf.shape)
    # plt.figure()
    # plt.imshow(img, cmap='gray')
    # plt.title(f"Original Image {img.shape[:2]}")

    # plt.figure()
    # plt.imshow(psf, cmap='gray')
    # plt.title(f"PSF {psf.shape[:2]}")
    return img, psf

def rescale_img(img, scale, ref, gray = False, channel_axis=2):
    img_shape = img.shape
    ref_shape = ref.shape
    scale_x = (img_shape[0] * scale) / ref_shape[0]
    scale_y = (img_shape[1] * scale) / ref_shape[1]
    if gray:
        return rescale(img, scale, anti_aliasing=True), (scale_x, scale_y)

    return rescale(img, scale, anti_aliasing=True, channel_axis=channel_axis), (scale_x, scale_y)

# Returns Horizontal(min, max), Vertical(min, max)

def get_extent_increase(crop_path, uncropped_path, params,rgb = True, show=True, actual_shape=False):
    ## Load Cropped, Uncropped
    cropped = np.load(crop_path)
    uncropped = np.load(uncropped_path)

    uncropped = np.clip(uncropped/np.max(uncropped), 0,1) # normalize

    # Original Dimensions
    if actual_shape:
        v = actual_shape[0]
        h = actual_shape[1]
    else:
        v = cropped.shape[0]
        h = cropped.shape[1]

    if rgb:
        uncropped = uncropped[:, :, 0]
        
    edges = generate_edges(uncropped, **params)
    print("Min: ", np.min(edges), "Max: ", np.max(edges))

    v_mid, h_mid = get_midpoints(uncropped)

    # h_coords, v_coords = get_coords(edges, v_mid, h_mid)

    h_coords, v_coords = get_widthpoints(edges)

    print("Horizontal Coords: ", h_coords)
    print("Vertical Coords: ", v_coords)

    h_diff = (h_coords[1] - h_coords[0])
    v_diff = (v_coords[1] - v_coords[0])

    if h_diff < cropped.shape[1]:
        h_diff = cropped.shape[1]
        h_coords = (h_mid - h_diff // 2, h_mid + h_diff // 2)
    if v_diff < cropped.shape[0]:
        v_diff = cropped.shape[0]
        v_coords = (v_mid - v_diff // 2, v_mid + v_diff // 2)

    print("Vertical Difference: {}".format(v_diff))
    print("Horizontal Difference: {}".format(h_diff))
        
    h_increase = h_diff / h
    v_increase = v_diff / v

    print("Horizontal Extent Increase: {}".format(h_increase))
    print("Vertical Extent Increase: {}".format(v_increase))

    if show:
        plot_extents(uncropped, cropped, edges, h_coords, v_coords, h_increase, v_increase)
    
    return edges, h_increase, v_increase
    

## Generate Edges
def generate_edges(image, sigma=1.8, low_threshold=0.3, high_threshold=1.0, show=True):
    edges = skimage.feature.canny(
        image=image,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,)

    if show:
        skimage.io.imshow(edges)

    return edges

def get_midpoints(image):
    vert = image.shape[0]
    horiz = image.shape[1]
    v_mid = vert // 2
    h_mid = horiz // 2

    print("Shape: ", image.shape, "Vertical: ", v_mid, "Horizontal: ", h_mid)

    return v_mid, h_mid

def get_widthpoints(image):
    # Sweep over horizontal axis to find vertical extent
    v_min = 1e6
    v_max = 0
    for i in range(image.shape[1]):
        curr_slice = np.where(image[:, i] > 0)
        if len(curr_slice[0]) == 0:
            continue
        curr_min = np.min(curr_slice)
        curr_max = np.max(curr_slice)

        if curr_min < v_min:
            v_min = curr_min
        if curr_max > v_max:
            v_max = curr_max
    
    # Sweep over vertical axis to find horizontal extent
    h_min = 1e6
    h_max = 0
    for i in range(image.shape[0]):
        curr_slice = np.where(image[i, :] > 0)
        if len(curr_slice[0]) == 0:
            continue
        curr_min = np.min(curr_slice)
        curr_max = np.max(curr_slice)

        if curr_min < h_min:
            h_min = curr_min
        if curr_max > h_max:
            h_max = curr_max
    
    return (h_min, h_max), (v_min, v_max)

def get_coords(edges, v_mid, h_mid):
    # Get middle slices of image, horizontally and vertically
    v_slice = edges[:, h_mid]
    h_slice = edges[v_mid, :]

    # Non-zero entries should be outlines
    h_points = np.where(h_slice > 0)
    v_points = np.where(v_slice > 0)

    # Image extent should be the first and last coordinates (min, max, mid)
    h_coords = (np.min(h_points), np.max(h_points), h_mid)
    v_coords = (np.min(v_points), np.max(v_points), v_mid)

    return h_coords, v_coords


def plot_extents(uncropped, cropped, edges, h_coords, v_coords, h_increase, v_increase):
    fig, ax = plt.subplots()
    ax.imshow(uncropped, cmap='gray')
    ax.imshow(edges, cmap='Oranges', alpha=0.7)

    v_mid, h_mid = get_midpoints(uncropped)
    ax.axvline(x = cropped.shape[1] // 2 + h_mid, color = 'b', linestyle = '-', label='Original') 
    ax.axvline(x = h_mid - cropped.shape[1] // 2, color = 'b', linestyle = '-') 
    ax.axhline(y = cropped.shape[0] // 2 + v_mid, color = 'b', linestyle = '-') 
    ax.axhline(y = v_mid - cropped.shape[0] // 2, color = 'b', linestyle = '-') 

    ax.axvline(x = h_coords[0], color = 'r', linestyle = '-', label='Extended') 
    # ax.annotate(text=str(h_coords[0]),xy=(h_coords[0], uncropped.shape[0] - 50), size=8, color='r')
    ax.axvline(x = h_coords[1], color = 'r', linestyle = '-') 
    # plt.axvline(x = h_coords[2], color = 'g', linestyle = '-') 

    ax.axhline(y = v_coords[0], color = 'r', linestyle = '-') 
    ax.axhline(y = v_coords[1], color = 'r', linestyle = '-') 
    # plt.axhline(y = v_coords[2], color = 'g', linestyle = '-') 
    
    txt = "Horizontal Extent Increase: %0.2fx | Limits: (%d, %d)\nVertical Extent Increase: %0.2fx | Limits: (%d, %d)" % (h_increase, h_coords[0], h_coords[1], v_increase, v_coords[0], v_coords[1])
    ax.set_title('FOV Extension')
    plt.figtext(0.5, 0.001, txt, wrap=True, horizontalalignment='center', fontsize=8)
    ax.legend(loc='lower right')

    plt.tight_layout(pad=2)

def compute_tamura(img, return_gradient_mag=False):
    # computes the Tamura Coefficient sparsity metric
    # make sure input is np.float32 or np.float64 for accurate results
    if len(img.shape) > 2:
        img = color.rgb2gray(img)
    assert len(img.shape) == 2 # must be a 2D image for correct behavior
    sobel_h = sobel(img, 0)
    sobel_v = sobel(img, 1)
    magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    magnitude_normed = magnitude / np.max(magnitude) # normalize is technically not needed but whatever, good practice
    tamura = np.sqrt(np.std(magnitude_normed) / np.mean(magnitude_normed)) # square root of ratio of SD and mean of gradient magnitude image
    if return_gradient_mag:
        return magnitude, tamura
    return tamura

# TODO: Figure out, what defines sensor increase?
def get_sensor_pct(sensor_dims):
    # Crop Dims

    # Get Widest horizontal and vertical (increase will be this box)

    # Calculate: crop / sensor size
    # Calculate: uncropped / sensor size
    # % Increase: (uncropped - crop) / sensor size
    return


## PSF GENERATION
def check_coords(x, y, coords, r, max):
    for i in range(len(x)):
        if np.linalg.norm(np.array(coords) - np.array((x[i], y[i]))) < 2 * r + max:
            return False
    return True

def plot_rml_irs(truncate, num_points, r, extent_h, extent_v, shape_h, shape_v, save_png=False, save_tiff=False, save_npy=False, reset_index = False):
    global psf_index
    global name

    # double check dimensionality: which index is x,y; h,v

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25)
    im_lst = []
    shape = (shape_h, shape_v)

    x = []
    y = []

    ## Step by 4s so FISTA doesn't break
    for _ in range(num_points):
        dist_to_point = False
        while not dist_to_point:
            coords = (random.randrange(shape_h//2 - extent_h // 2, shape_h//2  + extent_h // 2, 4), random.randrange(shape_h//2  - extent_v // 2, shape_h//2 + extent_v // 2, 4))           
            dist_to_point = check_coords(x, y, coords, r, shape_h // 20)
        
        x.append(coords[0])
        y.append(coords[1])

    ## Find actual extent
    ### TODO: will need to account for the radius of the circle
    ### Need a way to encode radius.
    extent_h = np.max(x) - np.min(x)
    extent_v = np.max(y) - np.min(y)

    for (x,y) in zip(x, y):
        blur = random.randrange(0, shape_h//25, 5) / 10
        grid = np.full(shape, 0, np.uint8)
        rad = int(r + blur * 2)
        # print(blur, rad)
        cv2.circle(grid,(x,y), rad, WHITE, -1)
        im_lst.append(gaussian_filter(grid, sigma=blur, truncate=truncate))
        # blurred = gaussian_filter(grid, sigma=blur, truncate=truncate)

    final_im = np.zeros(shape, np.uint8)

    for img in im_lst:
        final_im += img
    # blurred = im.fromarray(blurred)
    irs = ax.imshow(final_im, cmap='gray')
    tamura_coeff = np.round(compute_tamura(final_im), 2)
    print('Tamura Coefficient: ', tamura_coeff)
    name = 'rml-psf-{}-{}-{}-{}x{}-{}x{}-s{}-num{}'.format(num_points, r, truncate, extent_h, extent_v, shape[0], shape[1], SEED, psf_index)
    name = f'{name}-t{tamura_coeff:.2f}'
    psf_index += 1
    ax.set_title(name)

    if save_npy:
        np.save('./psfs/npy/{}.npy'.format(name), final_im)
    if save_png:
        final_im = im.fromarray(final_im)
        final_im.save('./psfs/png/{}.png'.format(name))
    if save_tiff:
        final_im.save('./psfs/tiff/{}.tiff'.format(name), format="TIFF", save_all=True)

    if reset_index:
        psf_index = 0

def generate_rml_irs(final_im, bounds, truncate, num_points, r, extent_h, extent_v):
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25)
    im_lst = []

    x = []
    y = []

    ## Step by 4s so FISTA doesn't break
    for _ in range(num_points):
        dist_to_point = False
        while not dist_to_point:
            coords = (random.randrange(bounds[0], bounds[1], 4), random.randrange(bounds[2], bounds[3], 4))
            dist_to_point = check_coords(x, y, coords, r - 50)
        
        x.append(coords[0])
        y.append(coords[1])

    ## Find actual extent
    # extent_h = np.max(x) - np.min(x)
    # extent_v = np.max(y) - np.min(y)

    for (x,y) in zip(x, y):
        blur = random.randrange(10, 50, 5) / 10
        grid = np.full(shape, 0, np.uint8)
        rad = int(r + blur * 1.5)
        # print(blur, rad)
        cv2.circle(grid,(x,y), rad, WHITE, -1)
        im_lst.append(gaussian_filter(grid, sigma=blur, truncate=truncate))
        # blurred = gaussian_filter(grid, sigma=blur, truncate=truncate)

    for img in im_lst:
        final_im += img
    # blurred = im.fromarray(blurred)
    irs = ax.imshow(final_im)
    name = 'rml-psf-{}-{}-{}-{}x{}-{}x{}-s{}'.format(num_points, r, truncate, extent_h, extent_v, shape[0], shape[1], SEED)
    ax.set_title(name)

    return final_im, name
    
def save_final_im(final_im, name):
    if not os.path.exists(f'./psfs/{name}'):
        os.makedirs(f'./psfs/{name}')
    np.save('./psfs/{}/{}.npy'.format(name, name), final_im)
    final_im = im.fromarray(final_im)
    final_im.save('./psfs/{}/{}.png'.format(name, name))
    final_im.save('./psfs/{}/{}.tiff'.format(name, name), format="TIFF", save_all=True)

def get_bounding_box(image, threshold=0):
    """
    Get the bounding box of nonzero values in the image, with an optional threshold.

    Parameters:
    - image: np.ndarray, the input image.
    - threshold: float, values below this threshold are considered zero.

    Returns:
    - (x_min, x_max, y_min, y_max): tuple, the bounding box coordinates.
    """
    # Apply thresholding
    image = color.rgb2gray(image) if len(image.shape) > 2 else image
    binary_image = image > threshold

    # Find nonzero indices
    nonzero_indices = np.argwhere(binary_image)

    if nonzero_indices.size == 0:
        return None  # No nonzero values found

    # Get bounding box coordinates
    x_min, y_min = np.min(nonzero_indices, axis=0)
    x_max, y_max = np.max(nonzero_indices, axis=0)

    return x_min, x_max, y_min, y_max

def plot_bounding_box(image, bbox, ax=None):
    """
    Plot the bounding box on the image.

    Parameters:
    - image: np.ndarray, the input image.
    - bbox: tuple, the bounding box coordinates (x_min, x_max, y_min, y_max).
    - ax: matplotlib.axes.Axes, optional, the axes to plot on. If None, a new figure is created.
    """
    if bbox is None:
        print("No bounding box to plot.")
        return

    x_min, x_max, y_min, y_max = bbox

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(image, cmap='gray')
    rect = plt.Rectangle((y_min, x_min), y_max - y_min, x_max - x_min, 
                            edgecolor='red', facecolor='none', linewidth=1, )
    ax.add_patch(rect)
    ax.set_title("Thresholded Bounding Box")
    plt.show()
# Plot four points on each of the corners
# remaining points should be random
# then, the calculated extents should match!
# need a better saving scheme of the points...

"""def recon_loop(type, subpath, f=2, object_name='penguin_custom/'):
    SOURCE_PATH = '/Users/clarahung/repos/lensless-dataset/'
    MEASUREMENT_PATH = SOURCE_PATH + subpath + type #+ object_name
    MEASUREMENT_LST = [MEASUREMENT_PATH + 'measurements/' + x + '/' + x + '-measurement.npy' for x in os.listdir(MEASUREMENT_PATH+ 'measurements/')][1:]
    measurement_names = [x for x in os.listdir(MEASUREMENT_PATH+ 'measurements/')]
    PSF_PATH = MEASUREMENT_PATH + 'psfs/'
    PSF_LST = [PSF_PATH + x for x in os.listdir(PSF_PATH) if '.png' in x]
    measurement_names.sort()
    MEASUREMENT_LST.sort()
    PSF_LST.sort()
    print(MEASUREMENT_LST)

    results_dir = MEASUREMENT_PATH + object_name
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    for i in range(len(MEASUREMENT_LST)):
        result_name = f'{results_dir}/{measurement_names[i]}.png'
        uncropped_result = f'{results_dir}/uncropped_{measurement_names[i]}.png'
        psf, img, mask = preprocess(PSF_LST[i], MEASUREMENT_LST[i], f, gray_image=gray_image)

        fista = fista_spectral_numpy(psf[:,:,1:2], mask[:,:,1:2], gray=gray_image) # green channel only

        # set FISTA parameters
        fista.iters = 200
        # Default: tv, Options: 'native' for native sparsity, 'non-neg' for enforcing non-negativity only
        fista.prox_method = 'tv'  
        fista.tv_lambda  = 1e-2 #1e-2  , 1e-3, 1e-1
        fista.tv_lambdaw = 0.01
        fista.print_every = 20

        out_img = fista.run(img)
        plotted_img = preplot(out_img[0][0])

        #final_im = contrast_stretch(out_img[0][0], 0.8)
        plt.imsave(result_name, plotted_img)
        plt.imsave(uncropped_result, preplot(out_img[0][1]))
        plt.imshow(plotted_img, cmap='gray')
        plt.title(f'FISTA after {fista.iters} iterations')"""
