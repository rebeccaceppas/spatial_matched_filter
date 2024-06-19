import numpy as np
import healpy as hp


class InputMap():

    def __init__(self, ra_max, ra_min, dec_max, dec_min, nsources=1, resolution=1):
        
        self.nsources = nsources
        self.ra_max = ra_max
        self.ra_min = ra_min
        self.dec_max = dec_max
        self.dec_min = dec_min
        self.resolution = resolution  # resolution in astropy units

    @property
    def ra(self):
        '''The RA bins of the 2D map'''
        return np.arange(self.ra_min, self.ra_max, self.resolution)
    
    @property
    def dec(self):
        '''The Dec bins of the 2D map'''
        return np.arange(self.dec_min, self.dec_max, self.resolution)



    def get_2d_map(self, locations=None, brightness=None, seed=1):
        '''locations are a list tuples (ra, dec) for each source'''

        nra = self.ra.size
        ndec = self.dec.size
        pix_x = np.arange(nra)
        pix_y = np.arange(ndec)

        map_2d = np.zeros((ndec, nra))

        if locations is None:
            xx = np.array(np.random.uniform(low=pix_x.min(), 
                                       high=pix_x.max(), 
                                       size=self.nsources), dtype=int)
            yy = np.array(np.random.uniform(low=pix_y.min(),
                                        high=pix_y.max(),
                                          size=self.nsources), dtype=int)

        else:
            ras = np.array(locations)[:,0]
            decs = np.array(locations)[:,1]

            xx = np.array(np.interp(ras, self.ra, pix_x), dtype=int)
            yy = np.array(np.interp(decs, self.dec, pix_y), dtype=int)

        if brightness is None:
            brightness = np.random.uniform(size=self.nsources)

        map_2d[yy, xx] = brightness

        return map_2d
    
    def get_2d_beam(self, beam_function, *args):

        npix_x = self.ra.size
        npix_y = self.dec.size

        x = np.linspace(-(npix_x//2), npix_x//2, npix_x)
        y = np.linspace(-(npix_y//2), npix_y//2, npix_y)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)

        beam = beam_function(r, *args)

        return beam/np.max(beam)

    def get_1d_map(self, nside, map_2d):

        npix = hp.nside2npix(nside)
        map_1d = np.zeros(npix)

        for i in range(self.ra.size):
            for j in range(self.dec.size):
                pix = hp.ang2pix(nside, self.ra[i], self.dec[j], lonlat=True)
                map_1d[pix] += map_2d[j, i]
        
        return map_1d
    


    def observe(self, map_2d, telescope_beam, noise_std=1e-2, add_noise=True):
        '''adds instrumental effects
        
        telescope beam is an array of the beam of the same shape as map_2d'''

        map_observed = np.fft.fftshift(
            np.fft.ifft2(
                np.fft.fft2(telescope_beam) * np.fft.fft2(map_2d)
            ))
        
        if add_noise:
            noise = np.random.normal(scale=noise_std, size=map_2d.size)
            map_observed += noise.reshape(map_2d.shape)

        return np.real(map_observed)



class GaussianBeam():

    def __init__(self, npix_x, npix_y, std):
        
        self.npix_x = npix_x
        self.npix_y = npix_y
        self.std = std

    def get_2d_beam(self):

        x = np.linspace(-(self.npix_x//2), self.npix_x//2, self.npix_x)
        y = np.linspace(-(self.npix_y//2), self.npix_y//2, self.npix_y)

        xx, yy = np.meshgrid(y, x)

        r = np.sqrt(xx**2 + yy**2)

        beam = np.exp(-r**2 / (2*self.std**2))

        self.beam2d = beam/np.max(beam)

        return beam/np.max(beam)
    
    def get_1d_beam(self, nside, beam2d):

        npix = hp.nside2npix(nside)
        beam_1d = np.zeros(npix)

        for i in range(self.ra.size):
            for j in range(self.dec.size):
                pix = hp.ang2pix(nside, self.ra[i], self.dec[j], lonlat=True)
                map_1d[pix] = map_2d[j, i]
        
        return map_1d


    

def gaussian_beam(r, std):

    beam = np.exp(-r**2 / (2*std**2))

    return beam / np.max(beam)







## old functions

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
