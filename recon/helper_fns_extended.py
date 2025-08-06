# LK file to help read in images clearly, 2/21/2023
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.transform import rescale, resize

def preprocess(psfname, imgname, f=8, gray_image=False, convert2gray = False, psf_scale = (0, 0)):
    # img = np.load(imgname, allow_pickle=True)
    if imgname[-3:] == 'npy':
        img = np.load(imgname)
    else:
        img = plt.imread(imgname)
    if psfname[-3:] == 'npy':
        psf0 = np.load(psfname)
    else:
        psf0 = plt.imread(psfname)
    print("PSF shape: ", psf0.shape)
    print("Image shape: ", img.shape)
    # print(np.min(psf0[:, :, 3]), np.max(psf0[:, :, 3]))

    # Rescale PSF if rescale factor is not zero
    if psf_scale != (0, 0):
        psf0 = resize(psf0, psf_scale, anti_aliasing=True)
        print("Rescaled PSF by a factor of:", psf_scale)
    
    if convert2gray:
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # psf0 = cv2.cvtColor(psf0, cv2.COLOR_RGB2GRAY)
        # gray_image = True
        print("Converted RGB images to grayscale..")

        img = np.average(img, axis=2)
        psf0 = np.average(psf0, axis=2)
        print("Gray Image Shape:", img.shape)

    # if grayscale, tile image
    if img.ndim == 2:
        img = np.tile(img[..., np.newaxis], (1, 1, 3))
        print("Tiled grayscale image to 3 channels..")
    if psf0.ndim == 2:
        psf0 = np.tile(psf0[..., np.newaxis], (1, 1, 3))
        print("Tiled PSF to 3 channels..")
        print("tiled shape: ", psf0.shape)

    # background subtract PSF
    # 10/23/2023 for multi-color channels need to background subtract from PSF each channel separately
    if not gray_image:
        #bg = np.mean(psf0[:100, :100])
        #psf0 = psf0 - bg
        #img = img - bg
        bg_channels = np.mean(psf0[:100, :100, :3], axis=(0, 1))
        psf0 = psf0[:, :, :3] - bg_channels
        
        # bg_channels = np.mean(psf0[:100, :100], axis=(0, 1))
        # psf0 = psf0 - bg_channels
        print("Background channels: ", bg_channels.shape)
        print(np.max(bg_channels), np.min(bg_channels))
        img = img - bg_channels

    else:
        bg = np.mean(psf0[:100, :100])
        psf0 = psf0 - bg
        img = img - bg
    #downsample images 
    ds = f
    psf = cv2.resize(psf0, (psf0.shape[1]//ds, psf0.shape[0]//ds))
    img = cv2.resize(img, (psf0.shape[1]//ds, psf0.shape[0]//ds))

    #normalize images. why does PSF get divided by norm and image get divided by max?
    if not gray_image:
        img = np.asarray(img / np.max(img, axis=(0,1)))
        psf = np.asarray(psf / np.linalg.norm(psf, axis=(0,1))) # does normalizing each axis do better? seems to reduce haze
    else:
        img = np.asarray(img / np.max(img))
        psf = np.asarray(psf / np.linalg.norm(psf))
    
    # note that images can't have odd dimensions. otherwise there will be a mismatch error raised
    if not gray_image:
        channels = 3
        psf = psf[:,:,:3] #in case the recorded data has 4 dimensions
        img = img[:,:,:3] 
    else:
        channels = 1
        if psf.ndim == 2:
            psf = np.expand_dims(psf, axis=2)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
    # create mask for operations
    mask = np.asarray(np.ones((psf.shape[0], psf.shape[1], channels)))
    
    
    return psf, img, mask

def preplot(image):
    image = image.squeeze()
    out_image = np.clip(image/np.max(image), 0,1) #removed np.flipud
    return out_image[:,:,...] # changed last index slicing from : to ...

def contrast_stretch(img, factor):
    if factor is not 0:
        return np.clip(img, 0, np.max(img)*factor)
    return img