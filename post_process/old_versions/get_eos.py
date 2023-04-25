# Get averages of pressure and density for EOS

# Thermo output columns as follow
# step , temp , PE , Ebond , Eangle , Edihedral , Epair , Evdwl , Emol , KE , TotE , Press , Density

import numpy as np
import os
import matplotlib.pyplot as plt
from shutil import copyfile

dirs=next(os.walk('.'))[1]

dirs_float = [int(i) for i in dirs]

#def get_eos(outfile):

for i in sorted(dirs_float):
    
    data_i=np.loadtxt('%s/averages.txt' %i ,dtype=float,skiprows=1)
    temp_i=data_i[:,0]
    press_i=data_i[:,1]
    density_i=data_i[:,2]
    
    eos= int(input('Isochore (1) or Isotherm (2): '))
    fig = plt.figure(figsize=(10.,8.))
    ax = fig.add_subplot(111)

    if eos == 1:
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Pressure (MPa)')
        ax.plot(temp_i,press_i,'x',label='%s' %i)
        plt.savefig('p-T.png')

    if eos == 2:
        ax.set_xlabel('Pressure (MPa)')
        ax.set_ylabel('Density (Kg/m^3)')
        ax.plot(press_i,density_i,'x',label='%s' %i)
        #plt.savefig('rho-p.png')



