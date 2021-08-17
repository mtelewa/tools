#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Input:
------
'profile' files generated from LAMMPS during the simulation.
These files store thermodynamic output in chunks of time and dimension
during LAMMPS simualtion run.

Output:
-------
txt files with the averaged qtts in the chunks
"""
import sys
import argparse
from processor_profile import profile

def get_parser():
    parser = argparse.ArgumentParser(
    description='Extract data from equilibrium and non-equilibrium profiles after \
     spatial binning or "chunking" of the output from LAMMPS ".profile" files.')

    #Positional arguments
    #--------------------
    parser.add_argument('--quantity', metavar='qunatity', action='store', type=str,
                    help='Quantity to average')
    parser.add_argument('--dimension', metavar='dimension', action='store', type=str,
                    help='Dimension to average the quantites in')
    parser.add_argument('--skip', metavar='skip', action='store', type=int,
                    help='Number of timesteps to skip')

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    if args.quantity=='denz':
        plot = profile('denZ.profile',args.skip,4)
    # if 'denz-time' in sys.argv:
    #     plot = profile('denZ.profile',10,4)
    if args.quantity=='denx':
        plot = profile('denX.profile',args.skip,4)
    # if 'denx-time' in sys.argv:
    #     plot = profile('denX.profile',10,4)
    if args.quantity=='mflux':
        plot = profile('momentaX.profile',args.skip,4)
    # if 'mflux-time' in sys.argv:
    #     ptxt.plot_from_txt('flux-stable-time.txt',10,2,0,1,'Time step','Mass flux $j_{x} \;(g/m^2.ns)$',
    #             'flux-time.png')
    if args.quantity=='vx':
        plot = profile('vx.profile',args.skip,4)
        #print("Shear rate %.3e s^-1 \nSlip length %.3f nm \nSlip velocity %.3f m/s"
        #        %(plot.shear_rate,plot.slip_length,plot.slip_velocity))
    # if 'vx-time' in sys.argv:
    #     plot = profile('vx.profile',10,4)
    if args.quantity=='tempx':
        plot = profile('tempX.profile',args.skip,4)
    if args.quantity=='tempz':
        plot = profile('tempZ.profile',args.skip,4)
    # if 'temp-time' in sys.argv:
    #     plot = profile('tempZ.profile',args.skip,4)
    if args.quantity=='virialx':
        plot = profile('virialChunkX.profile',args.skip,6)
    if args.quantity=='virialz':
        plot = profile('virialChunkZ.profile',args.skip,6)
    # if 'virial-time' in sys.argv:
    #     ptxt.plot_from_txt('virial.txt',2,0,1,'Time step','Pressure $(MPa)$','virial_pressure.png')
    if args.quantity=='sigzzL':
        plot = profile('fzL.profile',args.skip,4)
    if args.quantity=='sigzzU':
        plot = profile('fzU.profile',args.skip,4)
    if args.quantity=='sigzz':
        sigzz_upper = profile('fzU.profile',args.skip,4).sigzzU
        sigzz_lower = profile('fzL.profile',args.skip,4).sigzzL
        length = profile('fzU.profile',args.skip,4).length
        sigzz = 0.5*(sigzz_lower-sigzz_upper)
        np.savetxt("sigzz.txt", np.c_[length,sigzz],delimiter="  ",header="Length(nm)       Sigmazz(MPa)")
        #os.system("plot_from_txt.py sigzz.txt 1 0 1 sigzz.png --xlabel 'Length $(nm)$' \
        #                --ylabel 'Pressure $(MPa)$' --label 'Wall $\sigma_{zz}$' ")
    # if 'sigzz-time' in sys.argv:
    #     ptxt.plot_from_txt('stress.txt',2,0,3,1,'nofit','Wall $\sigma_{zz}$','Time step','Pressure $(MPa)$','sigzz-time.png')
    # if 'sigzzU-time' in sys.argv:
    #     ptxt.plot_from_txt('stressU.txt',2,0,3,1,'nofit',' Upper Wall $\sigma_{zz}$','Time step','Pressure $(MPa)$','sigzzU-time.png')
    # if 'sigzzL-time' in sys.argv:
    #     ptxt.plot_from_txt('stressL.txt',2,0,3,1,'nofit',' Lower Wall $\sigma_{zz}$','Time step','Pressure $(MPa)$','sigzzL-time.png')
    # if 'sigxz-time' in sys.argv:
    #     ptxt.plot_from_txt('stress.txt',2,0,1,1,'nofit','Wall $\sigma_{xz}$','Time step','Pressure $(MPa)$','sigxz.png')
    # if 'viscosity' in sys.argv:
    #     shear_rate = profile('vx.profile',10,4).shear_rate
    #     tauxz_lower = avg.columnStats('stressL.txt',1002,1)
    #     tauxz_upper = avg.columnStats('stressU.txt',1002,1)
    #     tauxz = 0.5*(tauxz_lower-tauxz_upper) #MPa
    #     print(tauxz)
    #     mu = tauxz/(shear_rate*1e-15)
    #     print(shear_rate*1e-15)
    #     print("Dynamic viscosity: %4.3f mPas" % (mu*1e-6))
    # if 'all' in sys.argv:
    #     profile('denZ.profile',10,4)
    #     profile('denX.profile',10,4)
    #     profile('momentaX.profile',10,4)
    #     profile('vx.profile',10,4)
    #     profile('virialChunkX.profile',10,6)
