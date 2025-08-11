import numpy as np
import numpy.fft as fft
from PIL import Image
import matplotlib.pyplot as plt
import yaml
import skimage.io as skio

def normalize_per_channel(img):
    """
    Normalize each channel of an image to [0, 1].
    Keeps dtype float32 until saving.
    """
    img = img.astype(np.float32)
    if img.ndim == 3:  # multi-channel
        for c in range(img.shape[2]):
            ch_min = img[..., c].min()
            ch_max = img[..., c].max()
            if ch_max > ch_min:
                img[..., c] = (img[..., c] - ch_min) / (ch_max - ch_min)
            else:
                img[..., c] = 0
    else:
        ch_min = img.min()
        ch_max = img.max()
        if ch_max > ch_min:
            img = (img - ch_min) / (ch_max - ch_min)
        else:
            img[:] = 0
    return img

def _l2_norm_per_channel(x):
    """Return L2 norm per channel: shape = (C,)"""
    return np.sqrt(np.sum(x.astype(np.float64)**2, axis=(0,1)))

def loadData(show_im=True):
    psf = skio.imread(psfname).astype('float32')
    data = skio.imread(imgname).astype('float32')

    # pad to 3 channels if needed
    if data.ndim == 2:
        data = np.repeat(data[..., None], 3, axis=2)
    if psf.ndim == 2:
        psf = np.repeat(psf[..., None], 3, axis=2)
    if data.shape[-1] > 3:
        data = data[..., :3]
    if psf.shape[-1] > 3:
        psf = psf[..., :3]

    # ---------- background subtraction (per-channel) ----------
    bg_data = np.mean(data[:100, :100, :], axis=(0,1))
    bg_psf  = np.mean(psf[:10, :10, :], axis=(0,1))

    data = np.clip(data - bg_data[None, None, :], a_min=0.0, a_max=None)
    psf  = np.clip(psf  - bg_psf[None, None, :], a_min=0.0, a_max=None)

    # ---------- resize ----------
    def resize(img, factor):
        num = int(-np.log2(factor))
        for _ in range(num):
            img = 0.25*(img[::2,::2,...] + img[1::2,::2,...] + img[::2,1::2,...] + img[1::2,1::2,...])
        return img

    psf = resize(psf, f)
    data = resize(data, f)

    # ---------- align PSF and data energy per-channel ----------
    data_norms = _l2_norm_per_channel(data) + 1e-12
    psf_norms  = _l2_norm_per_channel(psf)  + 1e-12
    scale = data_norms / psf_norms
    psf = psf * scale[None, None, :]

    # ---------- global stability scaling ----------
    common_max = max(np.max(np.abs(psf)), np.max(np.abs(data)), 1.0)
    psf  = psf  / common_max
    data = data / common_max

    # ---------- final per-channel normalization for algorithm inputs ----------
    psf  = normalize_per_channel(psf)
    data = normalize_per_channel(data)

    if show_im:
        plt.figure(); plt.imshow(psf); plt.title('PSF (processed)')
        plt.figure(); plt.imshow(data); plt.title('Raw data (processed)')
        plt.show()

    return psf.astype('float32'), data.astype('float32')

def U_update(eta, image_est, tau):
    return VectorSoftThresh(Psi(image_est) + eta/mu2, tau/mu2, axis=(2,3))

def SoftThresh(x, tau):
    return np.sign(x)*np.maximum(0, np.abs(x) - tau)

def VectorSoftThresh(x, tau, axis=-1):
    norm = np.sqrt(np.sum(x**2, axis=axis, keepdims=True))
    scale = np.maximum(0, norm - tau) / (norm + 1e-8)
    return scale * x

def Psi(v):
    dx = np.roll(v, 1, axis=1) - v
    dy = np.roll(v, 1, axis=0) - v
    return np.stack((dy, dx), axis=2)

def X_update(xi, image_est, H_fft, sensor_reading, X_divmat):
    return X_divmat * (xi + mu1*M(image_est, H_fft) + CT(sensor_reading))

def M(vk, H_fft):
    mag = np.abs(H_fft)
    print(np.mean(mag[:,:,0]), np.mean(mag[:,:,1]), np.mean(mag[:,:,2]))
    
    return np.real(fft.fftshift(
        fft.ifft2(fft.fft2(fft.ifftshift(vk, axes=(0,1)), axes=(0,1))*H_fft, axes=(0,1)),
        axes=(0,1)
    ))

def C(M):
    top = (full_size[0] - sensor_size[0])//2
    bottom = (full_size[0] + sensor_size[0])//2
    left = (full_size[1] - sensor_size[1])//2
    right = (full_size[1] + sensor_size[1])//2
    return M[top:bottom,left:right, :]

def CT(b):
    v_pad = (full_size[0] - sensor_size[0])//2
    h_pad = (full_size[1] - sensor_size[1])//2
    return np.pad(b, ((v_pad, v_pad), (h_pad, h_pad), (0, 0)), 'constant',constant_values=(0,0))

def precompute_X_divmat(): 
    return 1./(CT(np.ones(sensor_size)) + mu1 + 1e-8)

def W_update(rho, image_est):
    return np.maximum(rho/mu3 + image_est, 0)

def r_calc(w, rho, u, eta, x, xi, H_fft):
    return (mu3*w - rho)+PsiT(mu2*u - eta) + MT(mu1*x - xi, H_fft)

def V_update(w, rho, u, eta, x, xi, H_fft, R_divmat):
    freq_space_result = R_divmat*fft.fft2(fft.ifftshift(r_calc(w, rho, u, eta, x, xi, H_fft), axes=(0,1)), axes=(0,1))
    return np.real(fft.fftshift(fft.ifft2(freq_space_result, axes=(0,1)), axes=(0,1)))

def PsiT(U):
    diff1 = np.roll(U[...,0, :],-1,axis=0) - U[...,0, :]
    diff2 = np.roll(U[...,1, :],-1,axis=1) - U[...,1, :]
    return diff1 + diff2

def MT(x, H_fft):
    x_zeroed = fft.ifftshift(x)
    return np.real(fft.fftshift(
        fft.ifft2(fft.fft2(x_zeroed, axes=(0,1)) * np.conj(H_fft), axes=(0,1)),
        axes=(0,1)
    ))

def precompute_PsiTPsi():
    PsiTPsi = np.zeros(full_size[:2])
    PsiTPsi[0,0] = 4
    PsiTPsi[0,1] = PsiTPsi[1,0] = PsiTPsi[0,-1] = PsiTPsi[-1,0] = -1
    PsiTPsi = fft.fft2(PsiTPsi, axes=(0,1))
    return PsiTPsi[..., None]

def precompute_R_divmat(H_fft, PsiTPsi): 
    MTM_component = mu1 * (np.abs(H_fft) ** 2)
    PsiTPsi_component = mu2 * np.abs(PsiTPsi) * np.ones((1, 1, H_fft.shape[2]), dtype=np.float32)
    id_component = mu3
    return 1./(MTM_component + PsiTPsi_component + id_component + 1e-8)

def xi_update(xi, V, H_fft, X):
    return xi + mu1*(M(V,H_fft) - X)

def eta_update(eta, V, U):
    return eta + mu2*(Psi(V) - U)

def rho_update(rho, V, W):
    return rho + mu3*(V - W)

def init_Matrices(H_fft):
    X = np.zeros(full_size) 
    U = np.zeros((full_size[0], full_size[1], 2, full_size[-1]))
    V = np.zeros(full_size)
    W = np.zeros(full_size)
    xi = np.zeros_like(M(V,H_fft))
    eta = np.zeros_like(Psi(V))
    rho = np.zeros_like(W)
    return X,U,V,W,xi,eta,rho

def precompute_H_fft(psf):
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
    H_fft = precompute_H_fft(psf)
    X,U,V,W,xi,eta,rho = init_Matrices(H_fft)
    X_divmat = precompute_X_divmat()
    PsiTPsi = precompute_PsiTPsi()
    R_divmat = precompute_R_divmat(H_fft, PsiTPsi)
    
    for i in range(iters):
        X,U,V,W,xi,eta,rho = ADMM_Step(X,U,V,W,xi,eta,rho, [H_fft, data, X_divmat, R_divmat])

        image = C(V)
        print("Iteration:", i)
        for ch, name in enumerate(['R','G','B']):
            print(f"C{name}:", "min:", np.min(image[:,:,ch]), "max:", np.max(image[:,:,ch]))

        if i % disp_pic == 0:
            image = np.clip(C(V), 0, None)
            plt.figure(1)
            plt.imshow(normalize_per_channel(image))
            plt.title(f'Reconstruction after iteration {i}')
            plt.show()
    return image

if __name__ == "__main__":
    params = yaml.load(open("/Users/clarahung/repos/extendedfov/ADMM-refactor/admm_config.yml"), Loader=yaml.SafeLoader)
    for k,v in params.items():
        exec(k + "=v")

    psf, data = loadData(True)
    sensor_size = np.array(psf.shape)
    full_size = (2*sensor_size[0], 2*sensor_size[1], sensor_size[2])

    final_im = runADMM(psf, data)
    plt.imshow(normalize_per_channel(final_im))
    plt.title(f'Final reconstructed image after {iters} iterations')
    plt.show()

    saveim = input('Save final image? (y/n) ')
    if saveim == 'y':
        filename = input('Name of file: ')
        out_im = normalize_per_channel(final_im)
        plt.imshow(out_im)
        plt.axis('off')
        plt.savefig(filename+'.png', bbox_inches='tight')
