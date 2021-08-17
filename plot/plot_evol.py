import matplotlib.pyplot as plt

plt.figure()
for i in range(0,50):
    plt.plot((-0.1, 0.6),(i, i), color=(i/50., 0, (50-i)/50.)) 
plt.ylim([0,50]) 
plt.xlim([-0.1,0.6]) 
plt.xlabel('Posicion [m]')      
plt.ylabel('Velocidad [m/s]') 
line_segments.set_array(x)
line_segments.set_cmap(cm.coolwarm)  #this is the new line
ax.add_collection(line_segments)
plt.show()
