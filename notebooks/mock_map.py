import numpy as np
import healpy as hp

def create_1d_input_map(npix, nsources, source_location=None, source_brightness=None, seed=0):
    np.random.seed(seed)
    map_1d = np.zeros(npix)

    if source_location is None:
        source_location = int(np.uniform(low=0,
                                     high=npix,
                                     size=nsources))

    if source_brightness is None:
        source_brightness = np.random.uniform(size=nsources)

    map_1d[source_location] = source_brightness

    return map_1d

def perform_observation(map, telescope_beam=None):
    '''
    Performs a mock observation of a map by convolving with a chosen beam
    
    Inputs:
    -------
    map: 1D array representing the input map
    telescope_beam: string labelling the type of beam.
                    default is None and using hp.smoothing which is Gaussian symmetric.
                    Options are "airy" "gaussian". 
                    
    Outputs:
    --------
    map_observed: 1D array representing the observed map'''
    
    npix = map.size

    if telescope_beam is None:
        map_observed = hp.smoothing(map)
    
    else:
        beam_object = BEAMS[telescope_beam]
        beam = beam_object.get_beam(npix)
        map_observed = np.fft.fftshift(np.fft.ifft(np.fft.fft(beam) * np.fft.fft(map)))

    return np.real(map_observed)

def add_noise(map, noise_std, seed=0):

    np.random.seed(seed)
    noisy_map = map + np.random.normal(loc=0, scale=noise_std, size=map.size)

    return noisy_map




class GaussianBeam():

    def __init__(self, npix=10, std=10):
        
        self.std = std
        self.npix = npix

    def get_beam(self, npix):

        x = np.linspace(-(npix//2), npix//2, npix)
        beam = np.exp(-x**2 / (2*self.std**2))

        return beam
    

BEAMS = {'gaussian': GaussianBeam()}