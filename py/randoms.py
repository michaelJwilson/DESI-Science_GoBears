'''

module for generating randoms catalogs used in cosmological analyses 
(e.g. measuring the two-point correlation function) 

Authors: 
    ChangHoon Hahn -- changhoonhahn@lbl.gov
    Sukdeep Singh 
    Martin White 

'''
import numpy as np 
import h5py 
import warnings

from astropy.io import fits
from astropy.table import Table

from desimodel.io  import load_pixweight
from desimodel     import footprint


class Randoms(object): 
    ''' class object for constructing random catalogs used in 
    cosmological analyses (e.g. measuring the two-point correlation 
    function). 
    '''
    def __init__(self): 
        pass 

    def make(self, Nr, zs=None, dndz=None, seed=None, file=None): 
        ''' construct random catalogs with `Nr` randoms with the DESI 
        footprint. First (RA, Dec) values are uniformly sampled from a 
        box that conservatively covers the DESI footprint. The (RA, Dec) 
        values are then checked to ensure they are within the DESI 
        footprint. If redshift values are desired (i.e. kwargs zs or dndz
        is specified) redshifts are assigned to the random points. 

        parameters
        ----------
        Nr : int
            Number of random galaxies in the catalog

        zs : (optional) ndarray
            if specified the random (RA, Dec) values within the DESI
            footprint are assigned redshifts randomly sampled from this 
            array. 

        dndz : (optional) ndarray
            instead of zs, dndz can be provided and the random (RA, Dec) 
            values within the DESI footprint are assigned redshifts randomly 
            to reproduce the dndz.   

        seed : (optional) int
            random seed

        file : (optional) str
            if specified, the random catalog will be saved to file. At the
            moment only .fits and .hdf5 can be specified.   
        '''
        if zs is None and dndz is None: 
            msg = "No redshifts or dn/dz provided \n code will not return redshifts"
            warnings.warn(msg) 
        if zs and dndz: 
            raise ValueError("specify either zs or dndz, not both") 

        # generate randoms in a big box that covers all of 
        # the DESI footprint. This is a very brute froce 
        # way of doing things. Would require little effort 
        # to improve
        rr = np.random.RandomState(seed) # random seed

        ra, dec = rr.uniform(size=(2, 4*Nr)) 
    
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
            # save to astropy table 
            tb = Table([ra, dec, w], names=('ra', 'dec', 'weight'))  

            # no redshifts needed output now 
            if file is not None: # save to file 
                self._write2file(file, tb)
            return tb
    
        # generate redshifts for galaxies 
        if zs is not None: 
            # sample redshift from the provided redshifts
            # this automatically reproduces the dn/dz of the catalog 
            z = rr.choice(zs, size=Nr) 
        else: 
            raise NotImplementedError

        if file is not None: 
            self._write2file(file, np.vstack([ra, dec, z, w]).T, 
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

    def _write2file(self, name, tbl): 
        ''' write data to specified file 
        '''
        # check file type 
        ftype = name.split('.')[-1].lower()
        if ftype not in ['fits', 'hdf5']: 
            msg = ''.join([ftype, ' not supported at the moment', '\n', 'Only fits and hdf5 files supported at the moment']) 
            raise NotImplementedError(msg) 

        if ftype == 'fits': # fits file 
            # header specifies the columns; this can easily 
            # be expanded to fit more things 
            tbl.meta['COMMENTS'] = 'random catalog generated from randoms.py' # some comment here 
            tbl.write(name, format='fits') 

        elif ftype == 'hdf5':
            f = h5py.File(name, 'w')
            # set metadata 
            f.attrs['COMMENTS'] = 'random catalog generated from randoms.py' # some comment here 
            # save each column 
            for i, col in enumerate(tbl.colnames): 
                f.create_dataset(col, data=tbl.columns[i])
            f.close() 
        return None 


if __name__=="__main__": 
    rand = Randoms() 
    #rand.make(10000, seed=1, file='test.fits') 
    rand.make(10000, seed=1, file='test.hdf5') 
