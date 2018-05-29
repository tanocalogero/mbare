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
from tbtncTools import dagger, list2range_TBTblock
#from PIL import Image

####  Misc. defs.

def couplingMat(M, iosel1, iosel2, format='array'):
    iosel1.shape = (-1, 1)
    iosel2.shape = (1, -1)
    Mp = M[iosel1, iosel2]
    if format == 'csr':
        return sp.sparse.csr_matrix(Mp)
    elif format == 'array':
        return Mp

def pruneMat(M, iosel, format='array'):
    """
    M:      sparse matrix in csr format
    iosel:  list of orbitals on which M has to be pruned
    """    
    n_s = M.shape[1] // M.shape[0]
    iosel.shape = (-1, 1)
    if n_s != 1:
        iosel2 = np.arange(n_s) * M.shape[0]
        a = np.repeat(iosel.ravel(), n_s)
        iosel2 = (a.reshape(-1, n_s) + iosel2.reshape(1, -1)).ravel()
        iosel2 = np.sort(iosel2).reshape(1, -1)
    else:
        iosel2 = iosel.reshape(1, -1)
    Mp = M[iosel, iosel2]
    if format == 'csr':
        return sp.sparse.csr_matrix(Mp)
    elif format == 'array':
        return Mp

def pruneMat_range(M, RMAX):
    """
    M:      Hamiltonian object
    RMAX:   Maximum interaction range to be retained
    """
    Mcsr1, Mcsr2 = M.tocsr(0), M.tocsr(M.S_idx)
    RIJ = M.rij(what='orbital').tocsr(0)
    mask = RIJ > RMAX
    Mcsr1[mask] = 0
    Mcsr1.eliminate_zeros()
    Mcsr2[mask] = 0
    Mcsr2.eliminate_zeros()
    return Mcsr1, Mcsr2

def rearrange(HS, list, where='end'):
    if 'end' in where:
        # Remove selection from the current TB geom and append it at the end of its atoms list
        # The selected atoms have new indices from -len(a_dSE_host) to -1
        full = np.arange(HS.na)
        full = np.delete(full, list)
        full = np.concatenate((full, list))
        # New HS
        HS_new = HS.sub(full) 
        # List indices in new HS
        list_new = np.arange(HS.na-len(list), HS.na)
    return list_new, HS_new

def map_xyz(A, B, area_Delta, center_B=None, pos_B=None, area_int_for_buffer=None, 
    tol=None):
    ### FRAME
    print('Mapping from DELTA in DFT geometry to THETA in host geometry')
    # Recover atoms in Delta region of model A
    a_Delta = area_Delta.within_index(A.xyz)
    # Find the set Theta of unique corresponding atoms in model B 
    area_Theta = area_Delta.copy()

    if pos_B is not None:
        vector = pos_B
        B_translated = B.translate(-vector) 
    else: 
        if center_B is None: 
            center_B = B.center(what='xyz')
        #shift = [sc_xyz_shift(B, 0), 2*sc_xyz_shift(B, 1), 0.]
        shift = [0.,0.,0.]
        vector = center_B -shift -area_Theta.center
        B_translated = B.translate(-vector) 
    
    a_Theta = area_Theta.within_index(B_translated.xyz)

    # shift DFT and TB xyz to origo; sort DFT and TB lists in the same way; compare
    SE_xyz_inTSHS = A.xyz[a_Delta, :]
    SE_xyz_inTB = B.xyz[a_Theta, :]
    v1, v2 = np.amin(SE_xyz_inTSHS, axis=0), np.amin(SE_xyz_inTB, axis=0)
    a = SE_xyz_inTB - v2[None,:]
    b = SE_xyz_inTSHS - v1[None,:]
    
    if tol is None:
        tol = [0.01, 0.01, 0.01]
    print('Tolerance along x,y,z is set to {}'.format(tol))
    

    # Check if a_Delta is included in a_Theta and if yes, try to solve the problem
    # Following `https://stackoverflow.com/questions/33513204/finding-intersection-of-two-matrices-in-python-within-a-tolerance`
    # Get absolute differences between a and b keeping their columns aligned
    diffs = np.abs(np.asarray(a[:,None]) - np.asarray(b))
    # Compare each row with the triplet from `tol`.
    # Get mask of all matching rows and finally get the matching indices
    x1,x2 = np.nonzero((diffs < tol).all(2))

    # CHECK
    if len(x1) == len(a_Delta):
        if len(a_Theta) != len(a_Delta):
            print('\nWARNING: len(a_Delta) = {} is not equal to len(a_Theta) = {}'.format(len(a_Delta), len(a_Theta)))
            print('But since a_Delta is entirely contained in a_Theta, I will fix it by removing the extra atoms')
        print('\n OK! The coordinates of the mapped atoms in the two geometries match \
within the desired tolerance! ;)')
        a_Theta, a_Delta = a_Theta[x1], a_Delta[x2]
        a, b = a[x1], b[x2]
    elif len(x1) < len(a_Delta):
        print('\nWARNING: len(a_Delta) = {} is not equal to len(a_Theta) = {}'.format(len(a_Delta), len(a_Theta)))
        print('\n STOOOOOP: not all elements of a_Delta are in a_Theta')
        print('   Check `a_Theta_not_matching.xyz` vs `a_Delta.xyz` and \
try to change `pos_B` or increase the tolerance')
        v = B.geom.copy(); v.atom[a_Theta] = si.Atom(8, R=[1.44]); v.write('a_Theta_not_matching.xyz')
        exit(1)

    # Further CHECK, just to be sure
    if not np.allclose(a, b, rtol=np.amax(tol), atol=np.amax(tol)):
        print('\n STOOOOOP: The coordinates of the mapped atoms in the two geometries don\'t match \
within the tolerance!!!!')
        print('   Check `a_Theta_not_matching.xyz` vs `a_Delta.xyz` and \
try to change `pos_B` or increase the tolerance')
        v = B.geom.copy(); v.atom[a_Theta] = si.Atom(8, R=[1.44]); v.write('a_Theta_not_matching.xyz')
        exit(1)

    print(' Max deviation (Ang) =', np.amax(b-a))

    # WARNING: we are about to rearrange the atoms in the host geometry!!!
    a_Theta_rearranged, new_B = rearrange(B, a_Theta, where='end')
    print("\nSelected atoms mapped into host geometry, after rearrangement\n\
at the end of the coordinates list (1-based): {}\n{}".format(len(a_Theta_rearranged), list2range_TBTblock(a_Theta_rearranged)))

    # Find and print buffer atoms
    if area_int_for_buffer is not None:
        ### FRAME
        # NB: that cuboids are always independent from the sorting in the host geometry
        area_int_B = area_int_for_buffer.copy()
        if center_B is not None:
            area_int_B.set_center(center_B-shift)
        buffer = area_int_B.within_index(new_B.xyz)
        # Write buffer atoms fdf block
        print("\nbuffer atoms after rearranging (1-based): {}\n{}".format(len(buffer), list2range_TBTblock(buffer)))
        v = new_B.geom.copy(); v.atom[buffer] = si.Atom(8, R=[1.44]); v.write('buffer.xyz')
        with open('block_buffer.fdf', 'w') as fb:
            fb.write("%block TBT.Atoms.Buffer\n")
            fb.write(list2range_TBTblock(buffer))    
            fb.write("\n%endblock\n")

    return a_Theta_rearranged, new_B


def in2out_frame_PBCoff(TSHS, TSHS_0, a_inner, eta_value, energies, TBT, 
    HS_host, pzidx=None, pos_dSE=None, area_Delta=None, area_int=None, TBTSE=None,
    useCAP=None, spin=0, tol=None):
    """
    TSHS:                   TSHS from perturbed DFT system
    TSHS_0:                 TSHS from reference unperturbed DFT system
    a_inner:                idx atoms in sub-region A of perturbed DFT system (e.g. frame)
                            \Sigma will live on these atoms
    eta_value:              imaginary part in Green's function
    energies:               energy in eV for which \Sigma should be computed (closest E in TBT will be used )
    TBT:                    *.TBT.nc (or *.TBT.SE.nc) from a TBtrans calc. where TBT.HS = TSHS 
    HS_host:                host (H, S) model (e.g. a large TB model of unperturbed system). 
                            Coordinates of atoms "a_inner" in TSHS will be mapped into this new model.
                            Atomic order will be adjusted so that mapped atoms will be consecutive and at the end of the list   
    pzidx (=None):          idx of orbital per atom to be extracted from TSHS, in case HS_host has a reduced basis size
    pos_dSE (=0):           center of region where \Sigma atoms should be placed in HS_host 
    area_Delta (=None):     si.shape.Cuboid object used to select "a_Delta" atoms in TSHS
    area_int (=None):       si.shape.Cuboid object used to select "a_outer" atoms in TSHS
    TBTSE (=None):          *TBT.SE.nc file of self-energy enclosed by the atoms "a_inner" in TSHS (e.g., tip) 
    useCAP (=None):         use 'left+right+top+bottom' to set complex absorbing potential in all in-plane directions
    
    Important output files:
    "HS_DEV.nc":        HS file for TBtrans (to be used with "TBT.HS" flag)
                        this Hamiltonian is identical to HS_host, but it has no PBC 
                        and \Sigma projected atoms are moved to the end of the atom list   
    "SE_i.delta.nc":    \Delta \Sigma file for TBtrans (to be used as "TBT.dSE" flag)
                        it will contain \Sigma from k-averaged Green's function from TSHS,
                        projected on the atoms "a_inner" equivalent atoms of HS_host
    "SE_i.TBTGF":       Green's function file for usage as electrode in TBtrans 
                        (to be used with "GF" flag in the electrode block for \Sigma)
                        it will contain S^{noPBC}*e - H^{noPBC} - \Sigma from TSHS k-averaged Green's function,
                        projected on the atoms "a_inner" equivalent atoms of HS_host    
    "HS_SE_i.nc":       electrode HS file for usage of TBTGF as electrode in TBtrans
                        (to be used with "HS" flag in the electrode block for \Sigma)
    """

    dR = 0.005

    # a_dev from *TBT.nc and TBT.SE.nc is not sorted correctly in older versions of tbtrans!!! 
    a_dev = np.sort(TBT.a_dev)
    a_inner = np.sort(a_inner)
    
    # Find indices of atoms and orbitals in device region 
    o_dev = TSHS.a2o(a_dev, all=True)

    # Check it's carbon atoms in inner
    #for ia in a_inner:
    #    if TSHS.atom[ia].Z != 6:
    #        print('\nERROR: please select C atoms in inner region \n')
    #        exit(1)

    # Define inner region INSIDE the device region
    if pzidx is not None:
        # Selected 'pzidx' orbitals inside inner region   
        print('WARNING: you are selecting only orbital index \'{}\' in inner region'.format(pzidx))
        o_inner = TSHS.a2o(a_inner) + pzidx  # these are pz indices in the full L+D+R geometry 
    else:
        # ALL orbitals inside inner region
        o_inner = TSHS.a2o(a_inner, all=True)  # these are ALL orbitals indices in the full L+D+R geometry
    # Same atoms but w.r.t. device region 
    o_inner = np.in1d(o_dev, o_inner).nonzero()[0]

    # We will consider ALL orbitals of ALL atoms enclosed by the frame
    vv = TSHS.geom.sub(a_dev)
    a_outer_tmp = area_int.within_index(vv.xyz)
    o_outer = vv.a2o(a_outer_tmp, all=True)  # these are ALL orbitals indices in the outer region

    # Check
    v = TSHS.geom.copy()
    v.atom[v.o2a(o_dev, uniq=True)] = si.Atom(8, R=[1.44])
    v.write('o_dev.xyz')
    # Check
    vv = TSHS.geom.sub(a_dev)
    vv.atom[vv.o2a(o_inner, uniq=True)] = si.Atom(8, R=[1.44])
    vv.write('o_inner.xyz')
    # Check
    vv = TSHS.geom.sub(a_dev)
    vv.atom[vv.o2a(o_outer, uniq=True)] = si.Atom(8, R=[1.44])
    vv.write('o_outer.xyz')

    # Map a_inner into host geometry (which includes electrodes!)
    # WARNING: we will now rearrange the atoms in the host geometry
    # putting the mapped ones at the end of the coordinates list
    a_dSE_host, new_HS_host = map_xyz(A=TSHS, B=HS_host, center_B=pos_dSE, 
        area_Delta=area_Delta, area_int_for_buffer=area_int, tol=tol)
    v = new_HS_host.geom.copy(); v.atom[a_dSE_host] = si.Atom(8, R=[1.44]); v.write('a_dSE_host.xyz')
    # Write final host model
    new_HS_host.geom.write('HS_DEV.xyz')
    new_HS_host.geom.write('HS_DEV.fdf')
    new_HS_host.write('HS_DEV.nc')
    
    if useCAP:
        # Create dH | CAP
        dH_CAP = CAP(new_HS_host.geom, useCAP, dz_CAP=30, write_xyz=True)
        dH_CAP_sile = si.get_sile('CAP.delta.nc', 'w')
        dH_CAP_sile.write_delta(dH_CAP)

    # Energy grid
    if isinstance(energies[0], int):
        Eindices = list(energies)
    else:
        Eindices = [TBT.Eindex(en) for en in energies]
    E = TBT.E[Eindices] + 1j*eta_value
    
    # Remove periodic boundary conditions
    print('Removing periodic boundary conditions')
    TSHS_n = TSHS.copy()
    TSHS_n.set_nsc([1]*3)
    
    ##### Setup dSE
    print('Initializing dSE file...')
    o_dSE_host = new_HS_host.a2o(a_dSE_host, all=True).reshape(-1, 1)  # this has to be wrt L+D+R host geometry
    dSE = si.get_sile('SE_i.delta.nc', 'w')
    
    ##### Setup TBTGF
    print('Initializing TBTGF files...')
    # This is needed already here because of TBTGF (only @ Gamma!)
    if TSHS_n.spin.is_polarized:
        H_tbtgf = TSHS_n.Hk(dtype=np.float64, spin=spin)
    else:
        H_tbtgf = TSHS_n.Hk(dtype=np.float64)
    S_tbtgf = TSHS_n.Sk(dtype=np.float64)
    print(' Hk and Sk: DONE')
    # Prune to dev region
    H_tbtgf_d = pruneMat(H_tbtgf, o_dev)
    S_tbtgf_d = pruneMat(S_tbtgf, o_dev)
    # Prune matrices from device region to inner region
    H_tbtgf_i = pruneMat(H_tbtgf_d, o_inner)
    S_tbtgf_i = pruneMat(S_tbtgf_d, o_inner)
    # Create a geometry (one orb per atom, if DFT2TB) containing only inner atoms
    r = get_dft_param(TSHS_0, 0, pzidx, pzidx, unique=True, onlynnz=True)[0]
    g = si.geom.graphene(r[1], orthogonal=True)
    geom_dev = TSHS_n.geom.sub(a_dev)
    geom_dev = geom_dev.translate(-geom_dev.xyz[0, :])
    geom_dev.set_sc(g.sc.fit(geom_dev))
    geom_inner = geom_dev.sub(geom_dev.o2a(o_inner, uniq=True))
    geom_inner = geom_inner.translate(-geom_inner.xyz[0, :])
    geom_inner.set_sc(g.sc.fit(geom_inner))
    if pzidx is not None:
        geom_inner.atom[:] = si.Atom(1, R=r[-1]+dR); geom_inner.reduce()  # Necessary when going from DFT to TB
    # Construct the TBTGF form
    Semi = si.Hamiltonian.fromsp(geom_inner, H_tbtgf_i, S_tbtgf_i)
    # It is vital that you also write an electrode Hamiltonian,
    # i.e. the Hamiltonian object passed as "Semi", has to be written:
    Semi.write('HS_SE_i.nc')
    # Brillouin zone. In this case we will have a Gamma-only TBTGF!!!!!
    kpts, wkpts = np.array([[0.,0.,0.]]), np.array([1.0])
    BZ = si.BrillouinZone(TSHS_n); BZ._k = kpts; BZ._wk = wkpts
    # Now try and generate a TBTGF file
    GF = si.io.TBTGFSileTBtrans('SE_i.TBTGF')
    GF.write_header(E, BZ, Semi) # Semi HAS to be a Hamiltonian object, E has to be complex (WITH eta)
    ###############

    # if there's a self energy enclosed by the frame
    if TBTSE:
        pv = TBTSE.pivot('tip', in_device=True, sort=True).reshape(-1, 1)
        pv_outer = np.in1d(o_outer, pv.reshape(-1, )).nonzero()[0].reshape(-1, 1)


    print('Computing and storing Sigma in TBTGF and dSE format...')
    for i, (HS4GF, _, e) in enumerate(GF):
    #for i, e in enumerate(E):
        print('Doing E # {} of {}  ({} eV)'.format(i+1, len(E), e.real))  # Only for 1 kpt 

        # Read H and S from full TSHS (L+D+R) - no self-energies here!
        if TSHS_n.spin.is_polarized:
            Hfullk = TSHS_n.Hk(format='array', spin=spin)
        else:
            Hfullk = TSHS_n.Hk(format='array')
        Sfullk = TSHS_n.Sk(format='array')

        # Prune H, S to device region
        H_d = pruneMat(Hfullk, o_dev)
        S_d = pruneMat(Sfullk, o_dev)

        # Prune H, S matrices from device region to outer region
        H_o = pruneMat(H_d, o_outer)
        S_o = pruneMat(S_d, o_outer)
        #figure(); im = imshow(H_o.real, cmap='viridis', interpolation='none'); colorbar(im)
        #savefig('H_o.real.png', dpi=200)
        #figure(); im = imshow(S_o.real, cmap='viridis', interpolation='none'); colorbar(im)
        #savefig('S_o.real.png', dpi=200)

        # Coupling matrix from inner to outer (o x i)
        V_oi = couplingMat(H_d, o_outer, o_inner)
        #figure(); im = imshow(V_oi.real, cmap='viridis', interpolation='none'); colorbar(im)
        #savefig('V_oi.real.png', dpi=200)

        invG_o = S_o*e - H_o 

        # if there's a self energy enclosed by the frame
        if TBTSE:
            SE_ext = TBTSE.self_energy('tip', E=e.real, k=[0.,0.,0.], sort=True)
            invG_o[pv_outer, pv_outer.T] -= SE_ext

        G_o = np.linalg.inv(invG_o)
        
        # Self-energy in inner, connecting inner to outer (i x i)
        SE_i = np.dot(np.dot(dagger(V_oi), G_o), V_oi)
        #figure(); im = imshow(SE_i.real, cmap='viridis', interpolation='none'); colorbar(im)
        #savefig('SE_i.real.png', dpi=200)
        #figure(); im = imshow(SE_i.imag, cmap='viridis', interpolation='none'); colorbar(im)
        #savefig('SE_i.imag.png', dpi=200)
        
        # Write Sigma as a dSE file 
        Sigma_in_HS_host = sp.sparse.csr_matrix((len(new_HS_host), len(new_HS_host)), dtype=np.complex128)
        Sigma_in_HS_host[o_dSE_host, o_dSE_host.T] = SE_i
        delta_Sigma = si.Hamiltonian.fromsp(new_HS_host.geom, Sigma_in_HS_host)
        dSE.write_delta(delta_Sigma, E=e.real)

        # Write Sigma as TBTGF
        # One must write the quantity S_i*e - H_i - SE_i
        # Prune H, S matrices from device region to outer region
        H_i = pruneMat(H_d, o_inner)
        S_i = pruneMat(S_d, o_inner)
        #invG_i[pvl_inner, pvl_inner.T] += SE_ext
        if HS4GF:
            GF.write_hamiltonian(H_i, S_i)
        GF.write_self_energy(S_i*e - H_i - SE_i) 


def out2in_frame(TSHS, a_inner, eta_value, energies, TBT,
    HS_host, pzidx=None, pos_dSE=None, area_Delta=None, area_int=None, 
    TBTSE=None, spin=0, reuse_SE=False, tol=None, PBCdir=0, kmesh=None, kfromfile=True, 
    elecnames=['Left', 'Right']):
    """
    TSHS:                   TSHS from unperturbed DFT system
    a_inner:                idx atoms in sub-region A of perturbed DFT system (e.g. frame)
                            \Sigma will live on these atoms
    eta_value:              imaginary part in Green's function
    energies:               energy in eV for which \Sigma should be computed (closest E in TBT will be used )
    TBT:                    *.TBT.nc (or *.TBT.SE.nc) from a TBtrans calc. where energy and kpoints are the desired ones
                            es. it can be for example the TSHS one with the tip or the TSHS_0 without it  
    HS_host:                host (H, S) model (e.g. TSHS from separate system where \Sigma should be used as electrode). 
                            Coordinates of atoms "a_inner" in TSHS will be mapped into this new model.
                            Atomic order will be adjusted so that mapped atoms will be consecutive and at the end of the list
    pzidx (=None):          idx of orbital per atom to be extracted from TSHS, in case HS_host has a reduced basis size
    pos_dSE (=0):           idx of origin where \Sigma atoms should be placed in HS_host 
    TSHS_elec (=None):      TSHS of electrode L and R, in case TSHS was a L+D+R configuration
    dx (=None):             maximum distance (along axis=0) from cell boundary within which "a_inner" atoms are chosen in TSHS.
                            If necessary, remember to exclude the electrode width
    dy (=None):             maximum distance (along axis=1) from cell boundary within which "a_inner" atoms are chosen in TSHS.
                            If necessary, remember to exclude the electrode width
    TBTSE (=None):          *TBT.SE.nc file of self-energies L and R, in case TSHS was a L+D+R configuration

    Important output files:
    "inside_HS_DEV.nc":         HS file for TBtrans (to be used with "TBT.HS" flag)
                                this Hamiltonian is identical to HS_host, but it has no PBC 
                                and \Sigma projected atoms are moved to the end of the atom list   
    "inside_SE_i.delta.nc":     \Delta \Sigma file for TBtrans (to be used as "TBT.dSE" flag)
                                it will contain \Sigma from k-averaged Green's function from TSHS,
                                projected on the atoms "a_inner" equivalent atoms of HS_host
    "inside_SE_i.TBTGF":        Green's function file for usage as electrode in TBtrans 
                                (to be used with "GF" flag in the electrode block for \Sigma)
                                it will contain S^{noPBC}*e - H^{noPBC} - \Sigma from TSHS k-averaged Green's function,
                                projected on the atoms "a_inner" equivalent atoms of HS_host    
    "inside_HS_SE_i.nc":        electrode HS file for usage of TBTGF as electrode in TBtrans
                                (to be used with "HS" flag in the electrode block for \Sigma)
    """

    dR = 0.005

    # a_dev from *TBT.nc and TBT.SE.nc is not sorted correctly in older versions of tbtrans!!! 
    if TBTSE is not None:
        a_dev = np.sort(TBTSE.a_dev)
    else:
        a_dev = np.arange(TSHS.na)
    a_inner = np.sort(a_inner)
    
    # Find indices of atoms and orbitals in device region 
    o_dev = TSHS.a2o(a_dev, all=True)
    
    # Check it's carbon atoms in inner
    #for ia in a_inner:
    #    if TSHS.atom[ia].Z != 6:
    #        print('\nERROR: please select C atoms in inner region \n')
    #        exit(1)

    # Define inner region INSIDE the device region
    if pzidx is not None:
        # Selected 'pzidx' orbitals inside inner region   
        print('WARNING: you are selecting only orbital index \'{}\' in inner region'.format(pzidx))
        o_inner = TSHS.a2o(a_inner) + pzidx  # these are pz indices in the full L+D+R geometry 
    else:
        # ALL orbitals inside inner region
        o_inner = TSHS.a2o(a_inner, all=True)  # these are ALL orbitals indices in the full L+D+R geometry
    # Same atoms but w.r.t. device region 
    o_inner = np.in1d(o_dev, o_inner).nonzero()[0]

    # Check
    v = TSHS.geom.copy()
    v.atom[v.o2a(o_dev, uniq=True)] = si.Atom(8, R=[1.44])
    v.write('inside_o_dev.xyz')
    # Check
    vv = TSHS.geom.sub(a_dev)
    vv.atom[vv.o2a(o_inner, uniq=True)] = si.Atom(8, R=[1.44])
    # Equivalent to vv.atom[a_inner] = si.Atom(8, R=[1.44])
    vv.write('inside_o_inner.xyz')

    # Map a_inner into host geometry (which includes electrodes!)
    # WARNING: we will now rearrange the atoms in the host geometry
    # putting the mapped ones at the end of the coordinates list
    a_dSE_host, sorted_HS_host = map_xyz(A=TSHS, B=HS_host, pos_B=pos_dSE, 
        area_Delta=area_Delta, area_int_for_buffer=area_int, tol=tol)
    v = sorted_HS_host.geom.copy(); v.atom[a_dSE_host] = si.Atom(8, R=[1.44]); v.write('inside_a_dSE_host.xyz')
    sorted_HS_host.write('inside_HS_DEV_periodic.nc')
    
    # Write final host model (same as TSHS, but
    # w/o periodic boundary conditions and with frame at the end of the coor.list)
    new_HS_host = sorted_HS_host.copy()
    new_HS_host.set_nsc([1]*3)    
    new_HS_host.geom.write('inside_HS_DEV.xyz')
    new_HS_host.geom.write('inside_HS_DEV.fdf')
    new_HS_host.write('inside_HS_DEV.nc')

    if reuse_SE:
        return new_HS_host
    else:
        # Energy grid (read from a TBT.nc file which has the desired energy contour)
        if isinstance(energies[0], int):
            Eindices = list(energies)
        else:
            Eindices = [TBT.Eindex(en) for en in energies]
        E = TBT.E[Eindices] + 1j*eta_value
        
        
        ############## initial TSHS w/o periodic boundary conditions
        print('Removing periodic boundary conditions')
        TSHS_n = TSHS.copy()
        TSHS_n.set_nsc([1]*3)

        ##### Setup dSE
        print('Initializing dSE file...')
        o_dSE_host = new_HS_host.a2o(a_dSE_host, all=True).reshape(-1, 1)  # this has to be wrt L+D+R host geometry
        dSE = si.get_sile('inside_SE_i.delta.nc', 'w')

        ##### Setup TBTGF
        print('Initializing TBTGF files...')
        # This is needed already here because of TBTGF (only @ Gamma!)
        if TSHS_n.spin.is_polarized:
            H_tbtgf = TSHS_n.Hk(dtype=np.float64, spin=spin)
        else:
            H_tbtgf = TSHS_n.Hk(dtype=np.float64)
        S_tbtgf = TSHS_n.Sk(dtype=np.float64)
        print(' Hk and Sk: DONE')
        # Prune to dev region
        H_tbtgf_d = pruneMat(H_tbtgf, o_dev)
        S_tbtgf_d = pruneMat(S_tbtgf, o_dev)
        # Prune matrices from device region to inner region
        H_tbtgf_i = pruneMat(H_tbtgf_d, o_inner)
        S_tbtgf_i = pruneMat(S_tbtgf_d, o_inner)

        # Create a geometry (one orb per atom, if DFT2TB) containing only inner atoms
        r = np.linalg.norm(TSHS_n.xyz[1,:]-TSHS_n.xyz[0,:])
        g = si.geom.graphene(r, orthogonal=True)
        geom_dev = TSHS_n.geom.sub(a_dev)
        geom_dev = geom_dev.translate(-geom_dev.xyz[0, :])
        geom_dev.set_sc(g.sc.fit(geom_dev))
        geom_inner = geom_dev.sub(geom_dev.o2a(o_inner, uniq=True))
        geom_inner = geom_inner.translate(-geom_inner.xyz[0, :])
        geom_inner.set_sc(g.sc.fit(geom_inner))
        if pzidx is not None:
            geom_inner.atom[:] = si.Atom(1, R=r[-1]+dR); geom_inner.reduce()  # Necessary when going from DFT to TB
        # Construct the TBTGF form
        Semi = si.Hamiltonian.fromsp(geom_inner, H_tbtgf_i, S_tbtgf_i)
        # It is vital that you also write an electrode Hamiltonian,
        # i.e. the Hamiltonian object passed as "Semi", has to be written:
        Semi.write('inside_HS_SE_i.nc')
        # Brillouin zone. In this case we will have a Gamma-only TBTGF!!!!!
        BZ = si.BrillouinZone(TSHS_n); BZ._k = np.array([[0.,0.,0.]]); BZ._wk = np.array([1.0])
        # Now try and generate a TBTGF file
        GF = si.io.TBTGFSileTBtrans('inside_SE_i.TBTGF')
        GF.write_header(E, BZ, Semi) # Semi HAS to be a Hamiltonian object, E has to be complex (WITH eta)
        ###############

        # if there's a self energy in the initial TSHS, read it now
        if TBTSE:
            pv = []
            for iele, ele in enumerate(elecnames):
                pv.append(TBTSE.pivot(ele, in_device=True, sort=True).reshape(-1, 1))

        # In principle we can average G over a different kpt grid than a one that you would
        # use for a standard tbtrans calculation. 
        # So here we allow the user to define it from scratch.
        # By default we use time-reversal symmetry.
        # If no kmesh is provided, then we read it from TBT.
        #   WARNING: TBT might not contain a good grid. 
        #   Always check that the kpoint selected are OK!!!
        if kfromfile:
            # If there is only one periodic direction, one just needs to set PBCdir
            if kmesh is None:
                kmesh = list(np.ones(3, dtype=np.int8))
                kmesh[PBCdir] = TBT.nkpt
            print('Reading {} kpoints from {}'.format(kmesh, TBT))
            # Restore TRS if absent from file
            if (TBT.kpt < 0.).any():
                print('Time reversal symmetry can be used in file. To speed up we restore it.')
                mp = si.MonkhorstPack(TSHS.geom, kmesh)
                klist = mp.k
                wklist = mp.weight
            else:
                klist = TBT.kpt
                wklist = TBT.wkpt            
        else:
            if kmesh is None:
                print('Please provide "kmesh", or set "kfromfile" to True to use kpts from {}'.format(TBT))
                print('Bye...'); exit(1)
            print('User provided kpoints mesh: {}'.format(kmesh))
            mp = si.MonkhorstPack(TSHS.geom, kmesh)
            klist = mp.k
            wklist = mp.weight
        print('List of selected kpoints | weights:')
        for j1,j2 in zip(klist, wklist):
            print('  {}\t{}'.format(j1,j2))

        print('Computing and storing Sigma in TBTGF and dSE format...')
        ################## Loop over E
        for i, (HS4GF, _, e) in enumerate(GF):
            print('Doing E = {} eV'.format(e.real))  # Only for 1 kpt 
            #Gssum_i = np.zeros((len(o_inner), len(o_inner)), np.complex128)
            Gssum_d = np.zeros((len(o_dev), len(o_dev)), np.complex128)
            
            ################## Loop over transverse k-points and average
            for ikpt, (kpt, wkpt) in enumerate(zip(klist, wklist)):        
                print('Doing kpt # {} of {}  {}'.format(ikpt+1, len(klist), kpt))
                # Read H and S from full TSHS (L+D+R) - no self-energies here!
                if TSHS_n.spin.is_polarized:
                    Hfullk = TSHS.Hk(kpt, format='array', spin=spin)
                else:
                    Hfullk = TSHS.Hk(kpt, format='array')
                Sfullk = TSHS.Sk(kpt, format='array')
                # Prune H, S to device region
                H_d = pruneMat(Hfullk, o_dev)
                S_d = pruneMat(Sfullk, o_dev)
                # Prune H, S matrices from device region to inner region
                #H_i = pruneMat(H_d, o_inner)
                #S_i = pruneMat(S_d, o_inner)

                # G^-1 without SE_electrodes
                #invG_i = S_i*e - H_i
                invG_d = S_d*e - H_d

                # add L and R
                if TBTSE:
                    for iele, ele in enumerate(elecnames):
                        SE_ext = TBTSE.self_energy(ele, E=e.real, k=kpt, sort=True)
                        invG_d[pv[iele], pv[iele].T] -= SE_ext
                    
                # Invert
                #G_i = np.linalg.inv(invG_i)
                G_d = np.linalg.inv(invG_d)
                # Average with previous kpts
                #Gssum_i += (0.5*wkpt)*( G_i + np.transpose(G_i) )
                Gssum_d += (0.5*wkpt)*( G_d + np.transpose(G_d) )
                
            ##################

            # Self-energy in dev
            invGssum_d = np.linalg.inv(Gssum_d)
            SE_d = S_tbtgf_d*e - H_tbtgf_d - invGssum_d

            # Now prune SE_d to inner region
            SE_i = pruneMat(SE_d, o_inner)
            # Check
            SE_i_check = np.zeros((len(SE_d), len(SE_d)), np.complex128)
            SE_i_check[o_inner.reshape(-1, 1), o_inner.reshape(1, -1)] = SE_i
            diff = SE_d - SE_i_check
            print('maxdiff: ', np.amax(np.absolute(diff)))

            # Write Sigma as TBTGF (remember, it's only Gamma!)
            # One must write the quantity S_i*e - H_i - SE_i
            H_tbtgf_i_arr = H_tbtgf_i.toarray()
            S_tbtgf_i_arr = S_tbtgf_i.toarray()
            if HS4GF:
                GF.write_hamiltonian(H_tbtgf_i_arr, S_tbtgf_i_arr)
            GF.write_self_energy(S_tbtgf_i_arr*e - H_tbtgf_i_arr - SE_i) 

            # Write Sigma as a dSE file (remember, it's only Gamma!) 
            Sigma_in_HS_host = sp.sparse.csr_matrix((len(new_HS_host), len(new_HS_host)), dtype=np.complex128)
            Sigma_in_HS_host[o_dSE_host, o_dSE_host.T] = SE_i
            delta_Sigma = si.Hamiltonian.fromsp(new_HS_host.geom, Sigma_in_HS_host)
            dSE.write_delta(delta_Sigma, E=e.real)

            # Compute DOS in device region of new_HS_host with kavg-SE 
            S4dos, H4dos, SE4dos = new_HS_host.Sk(format='array'), new_HS_host.Hk(format='array'), delta_Sigma.Pk(format='array')
            invG4dos = S4dos*e - H4dos - SE4dos
            G4dos = np.linalg.inv(invG4dos)
            dos = -np.trace(np.dot(G4dos, S4dos).imag)/np.pi
            print('dos:\t{}\t{}'.format(e.real, dos))
