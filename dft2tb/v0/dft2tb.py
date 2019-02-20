import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pylab import *
import sisl as si
import numpy as np

# Input tshs and nc files
tshs = si.get_sile('../SZP_Hpass_tip/sc_atop/gatedminus_morek/RUN.fdf').read_hamiltonian()
SEnc = si.get_sile('../SZP_Hpass_tip/sc_atop/gatedminus_morek/siesta.TBT.SE.nc')
tshs_0 = si.get_sile('../SZP_Hpass_tip/ELEC_MESH/gatedminus/RUN.fdf').read_hamiltonian()

# Frame
C_list = (tshs.atoms.Z == 6).nonzero()[0]
from tbtncTools import Delta
a_Delta, a_int, Delta, area_int = Delta(tshs, shape='Cuboid', z_graphene=tshs.xyz[0, 2], 
    thickness=10., ext_offset=tshs_0.cell[1,:].copy(), zaxis=2, atoms=C_list)

# Electrode in large scale TB
C_list_e = (tshs_0.atoms.Z == 6).nonzero()[0]
He = tshs_0.sub(C_list_e); He.reduce()
He = He.sub(He.atoms[0], orb_index=[2])
print(He)
#He = He.repeat(2,0)
He.write('He.nc')
He.geom.write('He.xyz')

# Large scale TB
H0 = He.repeat(10,0).tile(79,1)
#H0 = tshs.sub(C_list); H0.reduce()
#H0 = H0.sub(H0.atoms[0], orb_index=[2])
#H0 = H0.repeat(3,0).tile(3,1)
#H0.write('HS_DEV.nc')
H0.geom.write('HS_DEV_tmp.xyz')

# Energy contour
ne = 150
Ens = np.linspace(0., 1.6, ne)
#tbl = si.io.table.tableSile('contour.IN', 'w')
#tbl.write_data(Ens, np.zeros(ne), np.ones(ne), fmt='.8f')

# Create SE file to be read in tbtrans
eta = 0.    # use always zero! Remember to set it in electrode block
from libINOUT import in2out_frame_PBCoff
in2out_frame_PBCoff(tshs, tshs_0, a_Delta, eta, Ens, SEnc,
    H0, pzidx=2, pos_dSE=H0.center(what='xyz'), area_int=area_int,
    area_Delta=Delta, TBTSE=SEnc, useCAP='left+right')      

#    H0, pzidx=2, pos_dSE=H0.xyz[9627]+[-.5,-1.,0.], area_int=area_int,

# pos_dSE=H0.center(what='xyz')
# pos_dSE = H0.xyz[4588]+[1.5,-1.,0.]
