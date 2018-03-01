from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
from matplotlib import *
from pylab import *
import sisl as si
import numpy as np
from netCDF4 import Dataset
import time, sys
import scipy.linalg as sli
import scipy as sp
from scipy import mgrid
import os,sys
from itertools import groupby
from tbtncTools import sc_xyz_shift, makeTB_FrameOutside, CAP, read_fullTSHS, get_R_hop, Delta, get_dft_param, plot_couplings_dft2tb, makeTB
#from PIL import Image
from libINOUT import makeTB_FrameOutside, in2out_frame_PBCoff, out2in_frame


####### INPUT FILES
dir = '../'
TSHS = read_fullTSHS(dir+'GR.TSHS', dir+'STRUCT.fdf')
TBTSE = si.get_sile(dir+'siesta.TBT.SE.nc')
elec = si.get_sile('/zhome/6d/5/111711/DTU-SCRATCH/dft2tb/inout_tip/ELEC_GR_gate0/STRUCT.fdf').read_geometry(True)

dir0 = '/zhome/6d/5/111711/DTU-SCRATCH/dft2tb/inout_tip/pristine_0/'
TSHS_0 = read_fullTSHS(dir0+'GR.TSHS', dir0+'STRUCT.fdf')
TBTSE_0 = si.get_sile(dir0+'siesta.TBT.SE.nc')

###### Check whether Dirac point is contained in KP
# from tbtncTools import check_Dirac, get_Dirac
# mp = [...] # as in SIESTA
# E_Dirac = get_Dirac(TSHS, mp) 

###### ZOOM OUT
print('\n Creating files for ZOOM OUT\n')
# Define frame
z_graphene = TSHS.xyz[1, 2]
a_Delta, a_int, Delta, area_int = Delta(TSHS, shape='Cuboid', 
    z_graphene=z_graphene, ext_offset=[0., 1.1*elec.cell[1,1], 0])

# Create a large TB model parametrized from the DFT-pristine reference calculation
ww, ll = 5*TSHS.cell[0,0], 5*TSHS.cell[1,1]   # in Angstrom
print('Size of large host geometry: {} x {} Ang\n'.format(ww, ll))

# Coordinates of tip apex in TSHS
i_tip = -1
xyz_tip = TSHS.xyz[i_tip].copy()
xyz_tip[2] = z_graphene
print('Tip (x, y, z=z_graphene) coordinates in TSHS are:\n\t{}'.format(xyz_tip))
# the TB files will be saved later, after rearranging the atoms
HS_dev = makeTB_FrameOutside(TSHS, TBTSE, xyz_tip, TSHS_0, pzidx=2, nn='all', 
    WW=ww, LL=ll, elec=elec, save=False, return_bands=False, z_graphene=z_graphene)


# Energy and eta
Ens = [0.64]
eta = 0.00001   # In eV. In principle it oculd be different for ZOOM IN and ZOOM OUT
# Create SE file to be read in tbtrans
in2out_frame_PBCoff(TSHS, TSHS_0, a_Delta, eta, Ens, TBTSE,
    HS_dev, pzidx=2, pos_dSE=None, area_int=area_int, 
    area_Delta=Delta, TBTSE=TBTSE, useCAP='left+right+top+bottom')      


# ###### ZOOM IN
# print('\n Creating files for ZOOM IN\n')
# # Define frame
# # dx and dy are the distances from the cell boundaries 
# # within which we should consider outmost atoms
# # Remember to exclude the electrode along the transport axis
# dx, dy = 0.15*TSHS_elec.cell[0,0], 2*TSHS_elec.cell[1,1]
# a_inner = frame(TSHS_0, dx, dy, TBTSE_0)[0]
# # Energy and eta
# Ens = [0.792]
# eta = 0.001   # In eV. In principle it oculd be different for ZOOM IN and ZOOM OUT
# # Index of francobollo's origin in the large TB geometry
# ixyz_0_1 = TSHS_elec.na
# # Create SE file to be read in tbtrans
# out2in_frame(TSHS_0, a_inner, eta, Ens, TBTSE_0,
#     TSHS, pos_dSE=ixyz_0_1, TSHS_elec=TSHS_elec, 
#     dx=dx, dy=dy, TBTSE=TBTSE_0)      
