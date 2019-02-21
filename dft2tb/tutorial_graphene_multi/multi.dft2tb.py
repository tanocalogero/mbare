import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pylab import *
import sisl as si
import numpy as np

# Input tshs and nc files
tshs_0 = si.get_sile('/work3/s173078/graphene_molecule_tip/ELECS/ELEC_GR_gateminus/RUN.fdf').read_hamiltonian()
print(tshs_0)

# Electrode in large scale TB
He = tshs_0.sub_orbital(tshs_0.atoms[0], orbital=[2])
print(He)
He.write('He.nc')

# Large scale TB
H0 = He.repeat(16,0).tile(72,1)
H0.geom.write('HS_DEV_tmp.xyz')
print(H0)


from lib_dft2tb import makearea

# Frame tip
tshs = si.get_sile('/zhome/6d/5/111711/DTU-SCRATCH/dft2tb/inout_tip/hpc/z1.8/gateminus/RUN.fdf').read_hamiltonian()
print(tshs)
z_graphene = tshs.xyz[0, 2]
alist = (tshs.xyz[:,2] < z_graphene+1).nonzero()[0]
a_Delta, a_int, Delta, area_ext, area_int = makearea(tshs, 
	shape='Cuboid', 
	z_area=z_graphene, 
    thickness=6., 
    ext_offset=tshs_0.cell[1,:].copy(), 
    zaxis=2, 
    atoms=alist)
frame_tip = (a_Delta, Delta, area_ext, area_int)

# Frame epoxy
tshs_epoxy = si.get_sile('/work3/s173078/graphene_epoxy/DFTTBT/RUN.fdf').read_hamiltonian()
print(tshs_epoxy)
z_graphene = tshs_epoxy.xyz[0, 2]
alist = (tshs_epoxy.xyz[:,2] < z_graphene+0.4).nonzero()[0]
a_Delta_epoxy, a_int_epoxy, Delta_epoxy, area_ext_epoxy, area_int_epoxy = makearea(tshs_epoxy, 
	shape='Cuboid', 
	z_area=z_graphene, 
    thickness=6., 
    ext_offset=tshs_0.cell[1,:].copy(), 
    zaxis=2, 
    atoms=alist)
frame_epoxy = (a_Delta_epoxy, Delta_epoxy, area_ext_epoxy, area_int_epoxy)



# Span along x, providing distance from source x coord. in command line
import sys
# Number of times drain should be shifted (x-xsource). Integer Multiple of 2pores cell vector
x = float(sys.argv[1])		


# In order to use 2 sigmas in the TBmodel we need to 
# rearrange the coresponding atoms consecutively at the end of the atoms list
xyz_tsource = H0.center(what='xyz') - 0.25*H0.cell[1,:]
xyz_tdrain = H0.center(what='xyz') + (-3+x)*tshs_0.cell[0,:]
xyz_epoxy1 = H0.center(what='xyz') + 0.25*H0.cell[1,:]

from lib_dft2tb import construct_modular
Hfinal, l_al, l_buf = construct_modular(H0=H0, 
	TSHS=[tshs, tshs, tshs_epoxy], 
	modules=[frame_tip, frame_tip, frame_epoxy], 
	positions=[xyz_tsource, xyz_tdrain, xyz_epoxy1])



# Write final model
Hfinal.geom.write('HS_DEV.xyz')
Hfinal.geom.write('HS_DEV.fdf')
Hfinal.write('HS_DEV.nc')



# Create dH | CAP
from lib_dft2tb import CAP
dH_CAP = CAP(Hfinal.geom, 'left+right', dz_CAP=50, write_xyz=True)
dH_CAP_sile = si.get_sile('CAP.delta.nc', 'w')
dH_CAP_sile.write_delta(dH_CAP)
