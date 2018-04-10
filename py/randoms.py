'''

module for generating randoms 

Authors: 
    ChangHoon Hahn 
    Michael Wilson

'''
import  matplotlib;             matplotlib.use('PDF')
import  math
import numpy as np 
import pylabl as pl 
import warnings

import nbodykit.lab as    *
from astropy.io import fits

from desimodel.io  import load_pixweight
from desimodel     import footprint
from astropy.table import Table
from astropy.io    import fits


class Randoms(object): 

    def __init__(self): 
        '''
        '''
        pass 

    def make(self, Nr, zs=None, dndz=None, seed=None, file=None): 
        ''' code for making the random catalogs  

        '''
        if zs is None and dndz is None: 
            msg = "No redshifts or dn/dz provided \n code will not return redshifts"
            raise warnings.warn(msg) 

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

        # trim random galaxies outside footprint and only keep N_r 
        # galaxies
        keep = (w > 0.) 
        assert np.sum(keep) > Nr 
        
        ra = ra[keep][:Nr]
        dec = dec[keep][:Nr]
        w = w[keep][:Nr]

        if zs is None and dndz is None: 
            if file is None: 
                # no redshifts needed output now 
                return np.vstack([ra, dec, w]).T
            else:
                # save to file 
                self.write2file(file, np.vstack([ra, dec, w]).T, cols=['ra', 'dec', 'weight'])
                return None 
    
        # sample redshift 
        if zs is not None: 
            z = rr.choice(zs, size=Nr) 
        else: 
    
        if file is None: 
            return np.vstack([ra, dec, z, w]).T          
        else: 
            self.write2file(file, np.vstack([ra, dec, z, w]).T, 
                    cols=['ra', 'dec', 'z', 'weight'])
            return None 

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


        return None 

'''
    #cosmo              = cosmology.Cosmology(h=0.7).match(Omega0_m=0.31)

    #ftiles             = fits.open('./desi-tiles.fits')
    #ftiles             = ftiles[1]


    tiles              = {}
    tiles['META']      = {}

    ## tile radius [degree].                                                                                            
    tiles['META']['DRADIUS'] = 1.605
    tiles['META']['RRADIUS'] = np.radians(tiles['META']['DRADIUS'])

    for tile in ftiles.data[::1]:
      if tile[4] == 1: ## In DESI;  Tiled whole 4PI steradians incase of footprint changes.    
        id                = tile[0]

        tiles[id]         = {}

        ## RA, DEC in DEG
        tiles[id]['RA']   = tile[1] 
        tiles[id]['DEC']  = tile[2]

        ## print id, tiles[id]['RA'], tiles[id]['DEC']
        
        tiles[id]['THETA'] = np.pi/2. - np.radians(tiles[id]['DEC'])

        tiles[id]['SINT']  = np.sin(tiles[id]['THETA'])
        tiles[id]['COST']  = np.cos(tiles[id]['THETA'])

        pl.plot(tiles[id]['RA'], tiles[id]['DEC'], 'k.')

    ## RAND GENERATION
    NRANDPERTILE       = 1
     
    rands              = {}

    for id in tiles:
        if id is not 'META':
            print id, tiles[id]['COST']
        
            radraws              =  np.random.uniform(low =  0.0, high = 1.0, size = NRANDPERTILE)
            cosdraws             =  np.random.uniform(low = -0.5, high = 0.5, size = NRANDPERTILE)

            rands[id]            = {}
            
            rands[id]['RAS']     =  2. * np.pi * radraws
            rands[id]['RAS']     =  np.degrees(rands[id]['RAS'])

            rands[id]['COSTS']   =  tiles[id]['COST'] ## + tiles[id]['SINT'] * tiles['METAx']['RRADIUS'] * cosdraws

            rands[id]['THETAS']  =  np.arccos(rands[id]['COSTS'])

            rands[id]['DECS']    =  np.pi/2. - np.arccos(rands[id]['THETAS'])
            rands[id]['DECS']    =  np.degrees(rands[id]['DECS'])

            pl.plot(rands[id]['RAS'], rands[id]['DECS'], 'k.')
    edges              = np.logspace(0.0, 3.0, num=100)

    tiles['Position']  = transform.SkyToCartesian(tiles[10]['RA'], tilse[10]['DEC'], np.ones_like(tiles[10]['RA']), cosmo=cosmo)

    ## http://nbodykit.readthedocs.io/en/latest/api/_autosummary/nbodykit.algorithms.paircount_tpcf.tpcf.html
    ## nbodykit.algorithms.paircount_tpcf.tpcf.SurveyData2PCF: 
    ##
    ## Default:  'RA', 'DEC', 'Redshift'   
    ## print SurveyData2PCF('2d', data, randoms, edges, cosmo=cosmo, Nmu=20)
         
    pl.xlim(360.0,  0.0)
    pl.ylim(-20.0, 90.0)

    pl.savefig('footprint.pdf')
'''   
