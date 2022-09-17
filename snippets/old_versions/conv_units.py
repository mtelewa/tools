# Python file to convert reduced temp to real temp and vice versa

import sys
import numpy as np

reduced_temp=0.85         
reduced_press=2.544
eps=0.234888              # kCal/mol
boltzmann=0.001987204118  # kCal/mol.K
sigma=3.405               # A

real_temp=(reduced_temp*eps)/boltzmann

reduced_press=25


#print ("Real Temperature is %s K" %real_temp)


#real_dens=1.5425604        #g/cm3
#const=0.6022               # from g/cm^3 to amu/A^3

N=8000
V=386700.34

#if sys.argv == "A"
reduced_dens=(N*sigma**3)/V

#print ("Reduced Density is %s" %reduced_dens)

real_press_cal=(reduced_press*eps)/(sigma**3)
real_press_atm=(real_press_cal/0.14393)*9869.23
real_press_mpa=real_press_atm/9869.23

print "Real pressure is %s KCal.mol^-1.A^-3 equivalent to %s Atm, equivalent to %s Mpa" %(real_press_cal,real_press_atm,real_press_mpa)

### another script

import numpy as np

avog_no=6.0221409e23
boltzmann= 1.380649e-23  # J/K

temp=118.2
atoms= 1

#from J/K to J

epsilon_1=boltzmann*temp    #J

#from j to Kcal

epsilon_2=epsilon_1/4184    #Kcal

moles=atoms/avog_no

epsilon=epsilon_2/moles     #Kcal/mol

print ("Epsilon is %g Kcal/mol" %epsilon)



