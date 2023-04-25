from ovito.data import *
from ovito.io import *
import numpy as np
from ovito.modifiers import SelectExpressionModifier


node = import_file("dump.01.lammpstrj")




def modify(frame, input, output):
    distance = np.linalg.norm( input.particles["Type"][1]-input.particles["Position"][2] )
    print (distance)
    output.attributes["Distance"] = distance


#node = import_file("dump.01.lammpstrj")

#node.modifiers.append(SelectExpressionModifier(expression="PotentialEnergy<-3.9"))

#export_file(node, "potenergy.txt", "txt", multiple_frames=True,
#         columns = ["Frame", "SelectExpression.num_selected"])

export_file(node, "a.txt", "txt", columns = ['Position.X', 'Position.Y', 'Position.Z'],  multiple_frames= False)
