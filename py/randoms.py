'''

module for generating randoms 

Authors: 
    ChangHoon Hahn 
    Michael Wilson

'''
import numpy as np 
import h5py 
import warnings

import nbodykit.lab as    *
from astropy.io import fits

from desimodel.io  import load_pixweight
from desimodel     import footprint
from astropy.table import Table
from astropy.io    import fits


class Randoms(object): 
    ''' add description of the object here. 
    '''
    def __init__(self): 
        pass 

    def make(self, Nr, zs=None, dndz=None, seed=None, file=None): 
        ''' code for constructing random catalogs. 
        '''
        if zs is None and dndz is None: 
            msg = "No redshifts or dn/dz provided \n code will not return redshifts"
            raise warnings.warn(msg) 

        if zs and dndz: 
            raise ValueError("specify either zs or dndz, not both") 

        # generate randoms in a big box that covers all of 
        # the DESI footprint. This is a very brute froce 
        # way of doing things. Would require little effort 
        # to improve
        rr = np.random.RandomState(seed) # random seed

        ra, dec = rr.uniform(size=(4.*Nr,2)) 
    
        ra *= 360.
        dec *= 90.
        dec += -25
        w = self.trim_to_footprint(ra, dec)

        # trim random galaxies outside footprint and only keep N_r of them 
        keep = (w > 0.) 
        assert np.sum(keep) > Nr 
        
        ra = (ra[keep])[:Nr]
        dec = (dec[keep])[:Nr]
        w = (w[keep])[:Nr]

        if zs is None and dndz is None: 
            # no redshifts needed output now 
            if file is not  None: # save to file 
                self.write2file(file, np.vstack([ra, dec, w]).T, 
                        cols=['ra', 'dec', 'weight'])
            return np.vstack([ra, dec, w]).T
    
        # generate redshifts for galaxies 
        if zs is not None: 
            # sample redshift from the provided redshifts
            # this automatically reproduces the dn/dz of the catalog 
            z = rr.choice(zs, size=Nr) 
        else: 
            raise NotImplementedError

        if file is not None: 
            self.write2file(file, np.vstack([ra, dec, z, w]).T, 
                    cols=['ra', 'dec', 'z', 'weight'])
        return np.vstack([ra, dec, z, w]).T          

    def trim_to_footprint(self, ra, dec):
        ''' Given RA and Dec values, return a np.ndarray of which pixels are in the 
        DESI footprint. In principle `desimodel.footprint` should return appropriate 
        systematic weight given ra and dec. I dont' think this is the case at the 
        moment. 
        '''
        assert len(ra) == len(dec) 

        pixweight = load_pixweight(256)
        healpix   = footprint.radec2pix(256, ra, dec) # ra dec of the targets
        weights   = pixweight[healpix] # weights=1 in footprint
        return weights

    def write2file(self, name, data, cols=None): 
        ''' write data to specified file 
        '''
        assert len(cols) == data.shape[1] 

        # check file type 
        ftype = name.strip('.')[-1].lower()
        if ftype not in ['fits', 'hdf5']: 
            raise NotImplementedError("Only fits and hdf5 files supported at the moment") 

        if ftype == 'fits': # fits file 


        elif ftype == 'hdf5': 

        return None 
