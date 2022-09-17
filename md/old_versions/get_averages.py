# Get averages of pressure and density for EOS

# Thermo output columns as follow
# step , temp , PE , Ebond , Eangle , Edihedral , Epair , Evdwl , Emol , KE , TotE , Press , Density

import numpy as np
import os
import matplotlib.pyplot as plt
from shutil import copyfile

dump_every=1e3                      # 'thermo' in LAMMPS input
timestep_to_average_from=500e3
prev_equilib=10e3                   # Previous equilibration steps if any

x=(timestep_to_average_from-prev_equilib)/dump_every
skip=(prev_equilib/dump_every)+4

dirs=[x[0] for x in os.walk(os.getcwd())][1:]

def get_averages(outfile):
    data=np.loadtxt('%s/thermo.out' %outfile ,dtype=float,skiprows=int(skip))
  
    global time
    time=data[:,0]
    
    global ave_T
    ave_T=np.mean(data[:,1][int(x):])       # Temperature
    #print ("Average Temperature is %g K" %ave_T)
    
    global ave_density_si
    ave_density=np.mean(data[:,12][int(x):])    # Density (g/cm3) 
    ave_density_si=ave_density*1e3              # convert g/cm3 to kg/m3
    #print ("Average Density is %g kg/m3" %ave_density_si)

    global ave_press_si
    ave_press=np.mean(data[:,11][int(x):])      # Pressure (atm)
    ave_press_si=ave_press*0.101325             # convert atm to MPa
    #print ("Average Pressure is %g MPa" %ave_press_si)
    
    # ----CHECK------
    density_first=data[:,12][int(x)]
    press_first=data[:,11][int(x)]
    #print ("Density at the first timestep used for averaging is %.8f g/cm3"  %density_first)
    #print ("Pressure at the first timestep used for averaging is %.8f g/cm3"  %density_first)
    
    return ave_T,ave_density_si,ave_press_si

average_temp = []
average_density = []
average_press = []

pressure_expt = [60.50,86.80,109.80,160.10,187.40,213.00,225.30,234.90]

cmd= 'cat log.lammps | sed -n "/Step/,/Loop time/p" | head -n-1 > thermo.out'

for i in sorted(dirs):
    #print (i)
    #copyfile('/home/mohamed/tools/extract_thermo.sh','%s' %i)
    os.system(cmd)
    get_averages(i)
    average_temp.append(ave_T)
    average_density.append(ave_density_si)
    average_press.append(ave_press_si)

temp_list = average_temp
density_list = average_density
press_list = average_press

density_array=np.asarray(density_list)
vol = (72.15/density_array)*0.001       # vol in dm3

# Print the averages into a file for fitting
f= open("averages.txt","w+")
f.write("Temperature (K)    Pressure (MPa)   Density(kg/m3)     Pressure(Expt.)     Vol.(dm3) \n")
for i in range (len(temp_list)):
    f.write(" %g    %g      %g      %g      %g\n"  % (temp_list[i],press_list[i],density_list[i],pressure_expt[i],vol[i]))


eos= int(input('Isochore (1) or Isotherm (2): '))

fig = plt.figure(figsize=(10.,8.))
ax = fig.add_subplot(111)

if eos == 1:
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (MPa)')
    ax.plot(temp_list,press_list,'x')
    #plt.savefig('p-T.png')

if eos == 2:
    ax.set_xlabel('Pressure (MPa)')
    ax.set_ylabel('Density (Kg/m^3)')
    ax.plot(temp_list,press_list,'x')
    #plt.savefig('rho-p.png')

