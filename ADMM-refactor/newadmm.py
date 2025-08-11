import numpy as np
import numpy.fft as fft
from PIL import Image
import matplotlib.pyplot as plt
import yaml
import skimage.io as skio
from skimage.transform import rescale, resize

def normalize(x, type='img'):
    """normalize image to be between 0-1"""
    if type == 'img':
        return np.asarray(x / np.max(x))
    elif type == 'psf':
        return np.asarray(x / np.linalg.norm(x))
    else:
        return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
    
def normalize_per_channel(x, type='img'):
    # from fista code
    if type == 'img':
        return np.asarray(x / np.max(x, axis=(0,1)))
    elif type == 'psf':
        return np.asarray(x / np.linalg.norm(x, axis=(0,1)))
    else:
        min_c = np.min(x, axis=(0,1), keepdims=True)
        max_c = np.max(x, axis=(0,1), keepdims=True)
        return (x - min_c) / (max_c - min_c + 1e-8)
    

# def resize(img, factor):
#     num = int(-np.log2(factor))
#     for i in range(num):
#         img = 0.25*(img[::2,::2,...]+img[1::2,::2,...]+img[::2,1::2,...]+img[1::2,1::2,...])
#     return img

def loadData(show_im=True):
    psf = skio.imread(psfname)
    psf = np.array(psf, dtype='float32')
    data = skio.imread(imgname)
    data = np.array(data, dtype='float32')

    if len(data.shape) == 2:
        # pad to 3 channels
        data =  np.repeat(data[..., None], 3, axis=2)
    if len(psf.shape) == 2:
        # pad to 3 channels
        psf =  np.repeat(psf[..., None], 3, axis=2)
    # remove alpha channel
    if data.shape[-1] > 3:
        data = data[:, :, 0:3]
 
    """In the picamera, there is a non-trivial background 
    (even in the dark) that must be subtracted"""

    # TODO: adjust for our data
    # bg = skio.imread('/Users/clarahung/repos/lensless-dataset/rml_darkframe.tiff')
    
    # per-channel background subtraction 
    bg = np.mean(data[5:100,5:100, :], axis=(0, 1))
    psf -= bg
    data -= bg
    
    
    psf = rescale(psf, f, channel_axis=2, anti_aliasing=True)
    data = rescale(data, f, channel_axis=2, anti_aliasing=True)
    
    """Now we normalize the images so they have the same total power. Technically not a
    necessary step, but the optimal hyperparameters are a function of the total power in 
    the PSF (among other things), so it makes sense to standardize it"""
    # this was for grayscale images
    # psf /= np.linalg.norm(psf.ravel())
    # data /= np.linalg.norm(data.ravel())

    psf = normalize_per_channel(psf, type='x')
    data = normalize_per_channel(data, type='x')
    # psf = normalize(psf, type='psf')
    # data = normalize(data, type='img')


    # for c, name in enumerate(['R', 'G', 'B']):
    #     plt.imshow(psf[:, :, c])
    #     plt.title(name)
    #     plt.colorbar()  # Add color bar
    #     plt.show()
    
    if show_im:
        fig1 = plt.figure()
        plt.imshow(psf)
        plt.title('PSF')
#         display.display(fig1)
        fig2 = plt.figure()
        plt.imshow(data)
        plt.title('Raw data')
#         display.display(fig2)
    return psf, data

def U_update(eta, image_est, tau):
    """
    Soft-threshold across the gradient and colors.
    """
    return VectorSoftThresh(Psi(image_est) + eta/mu2, tau/mu2, axis=(2, 3)) # change to just 2?


def SoftThresh(x, tau):
    # numpy automatically applies functions to each element of the array
    return np.sign(x)*np.maximum(0, np.abs(x) - tau)

def VectorSoftThresh(x, tau, axis=-1):
    """
    Vector soft thresholding.
    For RGB: x.shape = (H, W, 3), axis=-1
    For gradients: x.shape = (H, W, 2), axis=-1
    """
    norm = np.sqrt(np.sum(x**2, axis=axis, keepdims=True))
    # np.linalg.norm(x, axis=axis, keepdims=True)
    scale = np.maximum(0, norm - tau) / (norm + 1e-8)  # Avoid divide-by-zero
    return scale * x

def Psi(v):
    """
    Computes the backwards finite differences and stacks them together.
    Input: (H, W, 3)
    Output: (H, W, 2, 3)
    """
    dx = np.roll(v, 1, axis=1) - v
    dy = np.roll(v,1,axis=0) - v
    return np.stack((dy, dx), axis=2)


def X_update(xi, image_est, H_fft, sensor_reading, X_divmat):
    """
    Computes updated estimate of our blurry image in the full size.
    Out: (H, W, 3)
    """
    return X_divmat * (xi + mu1*M(image_est, H_fft) + CT(sensor_reading))


def M(vk, H_fft):
    """
    This function applies the forward model to vk in the Fourier domain.
    input: (H, W, 3)
    output: (H, W, 3)
    """
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(vk, axes=(0,1)), axes=(0,1))*H_fft, axes=(0,1)), axes=(0,1)))


def C(M):
    """Crops to sensor size."""
    # Image stored as matrix (row-column rather than x-y)
    top = (full_size[0] - sensor_size[0])//2
    bottom = (full_size[0] + sensor_size[0])//2
    left = (full_size[1] - sensor_size[1])//2
    right = (full_size[1] + sensor_size[1])//2
    return M[top:bottom,left:right, :]

def CT(b):
    """
    This function zero-pads b to the full size.
    """
    v_pad = (full_size[0] - sensor_size[0])//2
    h_pad = (full_size[1] - sensor_size[1])//2
    return np.pad(b, ((v_pad, v_pad), (h_pad, h_pad), (0, 0)), 'constant',constant_values=(0,0))


def precompute_X_divmat(): 
    """Only call this function once! 
    Store it in a variable and only use that variable 
    during every update step.
    
    Computes a constant mask of 1/(1 + mu1)
    Output: (H, W, 3)
    """
    return 1./(CT(np.ones(sensor_size)) + mu1 + 1e-8)


def W_update(rho, image_est):
    """
    Performs non-negativity projection.
    """
    return np.maximum(rho/mu3 + image_est, 0)

def r_calc(w, rho, u, eta, x, xi, H_fft):
    """
    Non-negativity + TV reg + data fidelity term
    """
    return (mu3*w - rho)+PsiT(mu2*u - eta) + MT(mu1*x - xi, H_fft)

def V_update(w, rho, u, eta, x, xi, H_fft, R_divmat):
    """Computes the update estimate of the clean image.
    """
    freq_space_result = R_divmat*fft.fft2(fft.ifftshift(r_calc(w, rho, u, eta, x, xi, H_fft), axes=(0,1)), axes=(0,1))
    return np.real(fft.fftshift(fft.ifft2(freq_space_result, axes=(0,1)), axes=(0,1)))

def PsiT(U):
    """
    This computes the forward finite differences across the x, y axes of the image, then sums them together.
    """
    diff1 = np.roll(U[...,0, :],-1,axis=0) - U[...,0, :]
    diff2 = np.roll(U[...,1, :],-1,axis=1) - U[...,1, :]
    return diff1 + diff2

def MT(x, H_fft):
    """
    Computes the adjoint / transpose of M.
    # """
    # print("C1: ", "min: ", np.min(H_fft[:, :, 0]), "max: ", np.max(H_fft[:, :, 0]))
    # print("C2: ", "min: ", np.min(H_fft[:, :, 1]), "max: ", np.max(H_fft[:, :, 1]))
    # print("C3: ", "min: ", np.min(H_fft[:, :, 2]), "max: ", np.max(H_fft[:, :, 2]))

    # mag = np.abs(H_fft)
    # print(np.mean(mag[:,:,0]), np.mean(mag[:,:,1]), np.mean(mag[:,:,2]))
    x_zeroed = fft.ifftshift(x)
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(x_zeroed, axes=(0,1)) * np.conj(H_fft), axes=(0,1)), axes=(0,1)))

def precompute_PsiTPsi():
    """
    Pre-computes the laplacian operator in frequency domain.
    Should be a spatial operator, so output: (H, W) ? 
    Then broadcast cross channels when applied
    """
    PsiTPsi = np.zeros(full_size[:2])
    PsiTPsi[0,0] = 4
    PsiTPsi[0,1] = PsiTPsi[1,0] = PsiTPsi[0,-1] = PsiTPsi[-1,0] = -1
    PsiTPsi = fft.fft2(PsiTPsi, axes=(0,1))
    return PsiTPsi[..., None] # broadcast to (H, W, 1)


def precompute_R_divmat(H_fft, PsiTPsi): 
    """Only call this function once! 
    Store it in a variable and only use that variable 
    during every update step
    
    H_fft: (H, W, 3)
    PTP: (H, W)
    
    """
    # MTM_component = mu1*(np.abs(np.conj(H_fft)*H_fft))
    MTM_component = mu1 * (np.abs(H_fft) ** 2)
    PsiTPsi_component = mu2 * np.abs(PsiTPsi) * np.ones((1, 1, H_fft.shape[2]), dtype=np.float32)
    # PsiTPsi_component = mu2*np.abs(PsiTPsi)
    id_component = mu3
    """This matrix is a mask in frequency space. So we will only use
    it on images that have already been transformed via an fft"""
    return 1./(MTM_component + PsiTPsi_component + id_component + 1e-8)

def xi_update(xi, V, H_fft, X):
    """
    Updates the dual var associated with xi.
    """
    return xi + mu1*(M(V,H_fft) - X)

def eta_update(eta, V, U):
    """
    Updates the dual var associated with eta.
    """
    return eta + mu2*(Psi(V) - U)

def rho_update(rho, V, W):
    """
    Updates the dual var associated with rho.
    """
    return rho + mu3*(V - W)


def init_Matrices(H_fft):
    X = np.zeros(full_size) 
    U = np.zeros((full_size[0], full_size[1], 2, full_size[-1])) # difference
    V = np.zeros(full_size)
    W = np.zeros(full_size)

    xi = np.zeros_like(M(V,H_fft)) # (H, W, 3)
    eta = np.zeros_like(Psi(V)) # (H, W, 2, 3)
    rho = np.zeros_like(W) # (H, W, 3)
    return X,U,V,W,xi,eta,rho


def precompute_H_fft(psf):
    """
    This function first zero-pads the PSF to the full convolution output size (2x sensor), then performs the Fourier Transform.
    Input: (H, W, 3)
    Output: (H, W, 3)
    """
    return fft.fft2(fft.ifftshift(CT(psf), axes=(0,1)), axes=(0,1))

def ADMM_Step(X,U,V,W,xi,eta,rho, precomputed):
    H_fft, data, X_divmat, R_divmat = precomputed
    U = U_update(eta, V, tau)
    X = X_update(xi, V, H_fft, data, X_divmat)
    V = V_update(W, rho, U, eta, X, xi, H_fft, R_divmat)
    W = W_update(rho, V)
    xi = xi_update(xi, V, H_fft, X)
    eta = eta_update(eta, V, U)
    rho = rho_update(rho, V, W)
    
    return X,U,V,W,xi,eta,rho


def runADMM(psf, data):
    H_fft = precompute_H_fft(psf) # (H, W, 3)
    X,U,V,W,xi,eta,rho = init_Matrices(H_fft)
    X_divmat = precompute_X_divmat()
    PsiTPsi = precompute_PsiTPsi() # (H, W, 1)
    R_divmat = precompute_R_divmat(H_fft, PsiTPsi)
    
    for i in range(iters):
        X,U,V,W,xi,eta,rho = ADMM_Step(X,U,V,W,xi,eta,rho, [H_fft, data, X_divmat, R_divmat])

        # debugging
        image = C(V)
        print("Iteration: ", i)

        if i % disp_pic == 0:
            image = C(V)
            image[image<0] = 0 # clipping
            f = plt.figure(1)
            # plt.imshow(image)
            # plt.imshow(np.clip(image / np.max(image), 0, 1))
            plt.imshow(np.clip(normalize_per_channel(image, type='x'),0,1))
            plt.title('Reconstruction after iteration {}'.format(i))
            plt.show()
    return image



if __name__ == "__main__":
    ### Reading in params from config file (don't mess with parameter names!)
    params = yaml.load(open("/Users/clarahung/repos/extendedfov/ADMM-refactor/admm_config.yml"), Loader=yaml.SafeLoader)
    for k,v in params.items():
        exec(k + "=v")

    ### Loading images and initializing the required arrays
    psf, data = loadData(True)
    sensor_size = np.array(psf.shape)
    full_size = (2*sensor_size[0], 2*sensor_size[1], sensor_size[2])

    ### Running the algorithm
    final_im = runADMM(psf, data)
    # plt.imshow(normalize_per_channel(final_im, type='img'))
    plt.imshow(np.clip(final_im / np.max(final_im), 0, 1))
    plt.title('Final reconstructed image after {} iterations'.format(iters))
    plt.show()
    saveim = input('Save final image? (y/n) ')
    if saveim == 'y':
        filename = input('Name of file: ')
        # plt.imshow(normalize_per_channel(final_im, type='img'))
        plt.imshow(final_im)
        plt.axis('off')
        plt.savefig(filename+'.png', bbox_inches='tight')

