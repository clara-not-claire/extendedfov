import scipy.io
import numpy as np
import skimage
import skimage.transform
import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def load_hyperspectral(simulation = False, 
                      img_path = '/media/midoridata/Kristina/spectral/viavi_camera_hamamatsu/july_2020/images/thor_alone2.png'):
    
    #loaded_mat = scipy.io.loadmat('/media/midoridata/Kristina/spectral/viavi_camera_hamamatsu/march_12_2020_FINAL.mat')
    #loaded_mat_psf = scipy.io.loadmat('/media/midoridata/Kristina/spectral/viavi_camera_hamamatsu/birthday_psf3.mat')
    
    loaded_mat = scipy.io.loadmat('/home/kristina/SpectralDiffuserCam/SpectralDiffuserCam/Python/SampleData/calibration.mat')
    

    # Pre-process mask and PSF
    mask1 = np.asarray(loaded_mat['mask'], dtype = np.float32)
    psf1 = np.asarray(loaded_mat['psf'], dtype = np.float32)

    c1 = 100; c2 = 420; c3 = 80; c4 = 540
    mask = mask1[c1:c2, c3:c4, :]
    psf = psf1[c1:c2, c3:c4]

    psf = psf/np.linalg.norm(psf)

    mask_sum = np.sum(mask, 2)
    ind = np.unravel_index((np.argmax(mask_sum, axis = None)), mask_sum.shape)
    mask[ind[0]-2:ind[0]+2, ind[1]-2:ind[1]+2, :] = 0
    
    # Load in image
    img = plt.imread(img_path)
    
    im = np.asarray(img[c1:c2, c3:c4]) #.astype(np.float32)
    im = im/np.max(im)
    im[ind[0]-2:ind[0]+2, ind[1]-2:ind[1]+2] = 0
    
    return im, mask, psf

def load_rolling_shutter(simulation = False, ):
    ds = 8
    path_diffuser = 'data/rolling_shutter/psf_averaged_2018-12-5.tif'
    psf_diffuser = plt.imread(path_diffuser)[:,:,1]
    psf_diffuser_ds = skimage.transform.resize(psf_diffuser, (psf_diffuser.shape[0]//ds, psf_diffuser.shape[1]//ds),
                                              anti_aliasing = True)
    psf_diffuser_ds = psf_diffuser_ds.astype('float32')

    path_img = 'data/rolling_shutter/tennis_bounce_00182.tif'
    path_img = 'data/rolling_shutter/dart_apple_220_1320us.tif'
    img = plt.imread(path_img)
    img_ds = skimage.transform.resize(img, (img.shape[0]//ds, img.shape[1]//ds),
                                              anti_aliasing = True)

    img_ds = img_ds.astype('float32')
    img_ds = img_ds/np.max(img_ds)

    if ds == 4:
        path_shutter = 'data/rolling_shutter/shutter_ds.mat'
        shutter = scipy.io.loadmat(path_shutter)['shutter_indicator']
    elif ds == 8:
        path_shutter = 'data/rolling_shutter/shutter_ds.mat'
        shutter = scipy.io.loadmat(path_shutter)['shutter_indicator']
    else:
        path_shutter = 'data/rolling_shutter/shutter_ds.mat'
        
    
    return img_ds, shutter, psf_diffuser_ds

def load_2D(simulation = False, img_index = 4):
    
    
    
    file_path_diffuser = '/home/kristina/2019_LenslessLearning_GitRepo/sample images/diffuser/'
    file_path_lensed = '/home/kristina/2019_LenslessLearning_GitRepo/sample images/lensed/'

    files = glob.glob(file_path_diffuser + '/*.npy')


    image_np = np.load(file_path_diffuser + files[img_index].split('/')[-1]).astype('float32')
    label_np = np.load(file_path_lensed + files[img_index].split('/')[-1]).astype('float32')
    
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    label_np = cv2.cvtColor(label_np, cv2.COLOR_BGR2RGB)
    
    path_diffuser = '/home/kristina/2019_LenslessLearning_GitRepo/sample images/psf.tiff'

    my_psf = np.array(Image.open(path_diffuser))
        
    psf_bg = np.mean(my_psf[0 : 15, 0 : 15])             #102
    psf_down = my_psf - psf_bg
    
    psf_down = psf_down/np.linalg.norm(psf_down)
    
    ds = 4   # Amount of down-sampling.  Must be set to 4 to use dataset images 

    psf_diffuser = np.sum(psf_down,2)


    h = skimage.transform.resize(psf_diffuser, 
                                 (psf_diffuser.shape[0]//ds,psf_diffuser.shape[1]//ds), 
                                 mode='constant', anti_aliasing=True)

    return image_np/np.max(image_np), label_np, h


def load_3D(simulation = False, ds = 3):
    psf_mat2 = scipy.io.loadmat('data/3D/pco_dense_zstack.mat')
    psf_3D2 = psf_mat2['zstack'].astype('float32')
    psf_3D2 = psf_3D2/np.linalg.norm(psf_3D2)

    psf_3D = cv2.resize(psf_3D2, (psf_3D2.shape[1]//ds, psf_3D2.shape[0]//ds))
    
    img_path = 'data/3D/fern2.png'
    img = plt.imread(img_path)

    img = img.astype('float32')


    img = cv2.resize(img, (psf_3D.shape[1], psf_3D.shape[0]))
    img = img/np.max(img)
    return img, psf_3D