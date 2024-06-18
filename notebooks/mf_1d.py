import numpy as np
from scipy import signal


def smooth_vector(vec, sig):

    n = len(vec)
    x = np.arange(n)
    x[n//2:] = x[n//2:] - n
    
    kernel=np.exp(-0.5*x**2/sig**2) #make a Gaussian kernel
    kernel=kernel/kernel.sum()

    vecft=np.fft.rfft(vec)
    kernelft=np.fft.rfft(kernel)
    vec_smooth=np.fft.irfft(vecft*kernelft) #convolve the data with the kernel
    
    return vec_smooth

def get_window(image):

    x = np.linspace(-np.pi/2,np.pi/2,len(image))
    win = signal.windows.tukey(len(x))

    return win

def get_pspec(image, sigma):

    win = get_window(image)

    ps = np.abs(np.fft.fft(image * win))**2
    ps_smooth = smooth_vector(ps, sigma)

    return ps_smooth

def matched_filter1d(image, beam, sigma_smooth):

    win = get_window(image)
    Ninv = 1/get_pspec(image, sigma_smooth)
    
    beam_ft = np.fft.fft(beam*win)
    beam_ft_white = beam_ft * np.sqrt(Ninv)

    data_ft = np.fft.fft(image*win)
    data_ft_white = data_ft * np.sqrt(Ninv)

    rhs = np.real(np.fft.ifft(data_ft_white * np.conj(beam_ft_white)))
    lhs = np.real(beam_ft_white.T @ beam_ft_white)

    mf = np.fft.fftshift(rhs / lhs)

    return mf