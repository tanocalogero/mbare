import matplotlib
matplotlib.use('Agg')
import sisl as si
from lib_bc import plot_bondcurrents

f = 'dft2tb.TBT.nc'
tbt = si.get_sile(f)
# Find device atoms
adev = tbt.a_dev

# Plot bond currents
for ie, en in enumerate(tbt.E):
    print('**** Energy = {} eV'.format(en))
    plot_bondcurrents(f, idx_elec=0, only='+',  zaxis=2, k='avg', E=en, avg=True, scale='%',
        vmin=0, vmax=15, ps=2, lw=15, log=False, adosmap=False, arrows=False, 
        lattice=False, ados=False, atoms=adev, spsite=None, out='bc')
