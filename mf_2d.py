import numpy as np


def pspec(image, pix_size, return_k=False):

    npix = image.size

    image_fft = np.fft.fftshift((np.fft.fft2(image, norm=None))) / npix
    pspec_ = np.abs(image_fft)**2

    k, Pk = spherical_bin(pspec_, pix_size, return_k=return_k)

    return k, Pk


def spherical_bin(image, pix_size, return_k=False):

    '''pix_size is in arcminutes'''

    npix_x, npix_y = image.shape[0], image.shape[1]

    xx, yy = np.indices((image.shape))
    r = np.sqrt((xx - npix_x//2)**2 + (yy - npix_y//2)**2)

    if return_k:
        k = 360/(pix_size/60.) * np.sqrt(((xx-npix_x//2)/npix_x)**2 + ((yy-npix_y//2)/npix_y)**2)
    else:
        k = np.copy(r)

    r_int = r.astype(int)

    weight = np.bincount(r_int.ravel())
    ks = np.bincount(r_int.ravel(), k.ravel()) / weight
    P_real = np.bincount(r_int.ravel(), np.real(image.ravel())) / weight
    P_imag = np.bincount(r_int.ravel(), np.imag(image.ravel())) / weight
    Pk = P_real + P_imag*1j
    
    return ks, Pk


def make_map_filter(image, x, profile):

    npix_x, npix_y = image.shape[0], image.shape[1]

    xx, yy = np.indices(image.shape)

    r = np.sqrt((xx - npix_x//2)**2 + (yy-npix_y//2)**2)
    r = r.ravel()

    filter_image = np.interp(r, x, profile)
    filter_image = filter_image.reshape(npix_x, npix_y)

    return filter_image


def matched_filter2d(image, beam, pix_size):

    npix = image.size

    beam_ft = np.fft.fftshift(np.fft.fft2(beam)) / npix
    image_ft = np.fft.fftshift(np.fft.fft2(image)) / npix

    k, Pk = pspec(image, pix_size)

    pspec_2d = make_map_filter(image, k, Pk)

    mf_ft = (beam_ft/pspec_2d) / np.sum(beam_ft**2/pspec_2d)

    mf = np.fft.ifftshift(
                        np.real(
                            np.fft.ifft2(
                                np.fft.ifftshift(mf_ft*image_ft)))) * npix
    noise = np.sqrt(np.real(np.sum(mf_ft**2 * pspec_2d)))

    return mf, noise