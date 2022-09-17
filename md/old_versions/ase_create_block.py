
# coding: utf-8

# In[1]:

"""Diffusion along rows"""
from __future__ import print_function

from ase import Atoms, Atom
from ase.visualize import view
from ase.neb import NEB, IDPP
from ase.optimize import BFGS, optimize, FIRE, LBFGS

from ase.calculators.lammpsrun import LAMMPS

import os

from numpy import *
from ase import *

from ase.lattice.cubic import BodyCenteredCubic
from ase.io.trajectory import Trajectory

from ase import io
from ase.io import read, write
import numpy as np


# In[2]:

from ase.calculators import bopfox #as bopcalc
import ase.io.bopfox
from ase.io import bopfox as bopio


# In[3]:

#calc = bopfox.BOPfox(model = 'Mrovec-2011', task='force', version= 'bop', scfsteps=100, 
                     #  scftol=0.01, printsc=True, scfbroydenmixpara = 0.002, magconfig= 'cm') #debug=True,
calc = bopfox.BOPfox(modelsbx='models.bx',model = 'AcklandThetford-1987', task='force') 


# In[ ]:

# can create a block in this way --> different directions
'''atoms = BodyCenteredCubic(directions=[[1,0,0], [0,1,0], [1,1,1]],
                           size=(2,2,3), symbol='Mo', pbc=(1,1,0),
                           latticeconstant=3.1472)'''


# In[52]:

LC = 3.1472
struct = BodyCenteredCubic(size=(2,2,2), symbol="Mo", pbc=True, latticeconstant = LC)
struct.get_number_of_atoms()


# In[36]:

# centers the atoms to the origin
#struct.center(about=(0., 0., 0.))


# In[61]:

tags = []
for a in struct:
    if a.x==0.0 and a.y==0.0 and a.z==0.0:
        a.tag=2
        tags.append(1)
    else:
        tags.append(0)


# In[62]:

struct.set_tags(tags)


# In[58]:

#adding vacuum to the structure
struct.center(vacuum=10, axis=(0,1,2))#,about=0.)


# In[ ]:

# can set PBC here also
#struct.pbc = (True, True, False)


# In[55]:

view(struct)


# In[59]:

write('struct.cfg',struct)
write('struct.xyz',struct)
bopio.write_strucbx(struct,filename='struct.bx',coord='direct')


# In[17]:

struct_2 = BodyCenteredCubic(size=(2,2,2), symbol="Mo", pbc=(0,0,1), latticeconstant = LC)
struct_2.get_number_of_atoms()


# In[18]:

struct_2.center(vacuum=10, axis=(0,1,2))


# In[19]:

view(struct_2)


# In[15]:

struct_3 = BodyCenteredCubic(size=(2,2,2), symbol="Mo", pbc=(0,1,1), latticeconstant = LC)
struct_3.get_number_of_atoms()


# In[ ]:

struct_3.center(vacuum=10, axis=(0,1,2))


# In[20]:

view(struct_3)


# In[ ]:

#a.set_pbc((True, True, False))

