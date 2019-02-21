import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pylab import *
import sisl as si
import numpy as np
 
# Nomenclature is the same as arXiv:1812.08054v1.

""" INPUT DFT FILES
Consider a DFT device with tip electrode and 2 NPG electrodes
1) run siesta for electrodes --> read Hamiltonian of NPG electrode (*.TSHS)
2) run siesta for device --> read Hamiltonian (*.TSHS)
3) run tbtrans using: 
 	TBT.CDF.SelfEnergy.Save 
 	TBT.CDF.SelfEnergy.Only F
--> read self energy file (*.TBT.SE.nc) 
"""
dir = '/zhome/6d/5/111711/DTU-SCRATCH/SanSeb/BerniNanomesh/June2018/SZP_Hpass_tip/'
tshs = si.get_sile(dir+'sc_atop/gatedminus_morek/RUN.fdf').read_hamiltonian()
SEnc = si.get_sile(dir + 'sc_atop/gatedminus_morek/siesta.TBT.SE.nc')
tshs_0 = si.get_sile(dir + 'ELEC_MESH/gatedminus/RUN.fdf').read_hamiltonian()


""" CONSTRUCT LARGE TB MODEL 
using parameters from the projection of the unperturbed tshs, i.e. the NPG electrode 
"""
# ELECTRODE (the same width as tshs. We will use Bloch theorem in tbtrans for the large TB)
# Sub the Hamiltonian on C atoms in NPG electrode
C_list_e = (tshs_0.atoms.Z == 6).nonzero()[0]  # nonzero returns a tuple. We don't want that
He = tshs_0.sub(C_list_e); He.reduce()  # sub does not eliminate empty species. We don't want that
# Sub this on the pz orbitals only
He = He.sub_orbital(He.atoms[0], orbital=[2])
# Save nc for tbtrans and xyz to check
He.write('He.nc')
He.geom.write('He.xyz')
# DEVICE Hext (tiled Hamiltonian!)
Hext = He.repeat(8,0).tile(39,1)
Hext.geom.write('Hext_tmp.xyz')

tshs.geom.write('tshs_tmp.xyz')


""" NOW LET'S CALCULATE THE SELF-ENERGY SigmaDFT2TB TO CONNECT THEM!
1) Define an area of tshs geometry where the connecting self-energy will be projected
This is the area for H11
"""
from lib_dft2tb import makearea
# We just want pz orbitals of C atoms (no H)
C_list = (tshs.atoms.Z == 6).nonzero()[0]
a_R1, a_R2, R1, area_ext, area_R2 = makearea(tshs, 
	shape='Cuboid', 
	z_area=tshs.xyz[0, 2], 
	thickness=10., 
	ext_offset=tshs_0.cell[1,:].copy(), 
	zaxis=2, 
#	center=whatever,
	atoms=C_list)
# We will then map this area into Hext. This is done inside 'in2out_frame_PBCoff'


"""
2) Calculate Sigma
"""
# Energy contour
ne = 1
Ens = np.array([0.7])
#Ens = np.linspace(0.2, 0.8, ne)
eta = 0.    # use always zero! Remember to set it in electrode block for tbtrans

# Compute SigmaDFT2TB and store it in a *.TBTGF file 
# We will use this file as an electrode in tbtrans
from lib_dft2tb import in2out_frame_PBCoff
in2out_frame_PBCoff(TSHS=tshs,
	a_R1=a_R1, 
	eta_value=eta, 
	energies=Ens, 
	TBT=SEnc, 
	HS_host=Hext, 
	orb_idx=2, 
	pos_dSE=Hext.center(what='xyz')+0.4*Hext.cell[1,:]-[0,5.31,0], 
	area_R2=area_R2, 
    area_for_buffer=area_ext, 
    area_R1=R1, 
    TBTSE=SEnc, 
    useCAP='left+right')      
