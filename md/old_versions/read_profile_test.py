#------------------------- Density along the length --------------------------------#

def get_denProfileX(fig):

    A=[]
    with open('denX.profile', 'r') as f:
        for line in f:
            if len(line.split())==4 and line.split()[0]!='#':
                A.append(line.split())

    A=np.asarray(A)
    global numChunks
    numChunks=int(A[-1,0])
    tSteps=int(len(A)/numChunks)

    A=np.reshape(A,(tSteps,numChunks,4))       #timesteps, chunks, value
    A=A.astype(np.float)

    global avg
    avg=np.zeros((numChunks-12,2))

    for i in range(len(avg)):
        avg[i,0]=A[0,i+6,1]
        for j in range(tSteps):
            avg[i,1]+=A[j,i+6,3]

        avg[i,1]/=tSteps

    ax = fig.add_subplot(111)
    ax.set_xlabel('length (nm)')
    ax.set_ylabel('Density (kg/m^3)')
    ax.plot(avg[:,0]/10,avg[:,1]*1e3,'-^')
    amin = 450
    amax = 650
    ax.yaxis.set_ticks(np.arange(amin,amax,25.))

    return
#fig2 = plt.figure(figsize=(10.,8.))
#get_denProfileX(fig2)
#plt.savefig('densityX.png')

#------------------------- Vz along the height --------------------------------#

def get_velProfileZ(fig):

    A=[]
    with open('vz.profile', 'r') as f:
        for line in f:
            if len(line.split())==4 and line.split()[0]!='#':
                A.append(line.split())

    A=np.asarray(A)
    numChunks=int(A[-1,0])
    tSteps=int(len(A)/numChunks)

    A=np.reshape(A,(tSteps,numChunks,4))       #timesteps, chunks, value
    A=A.astype(np.float)

    avg=np.zeros((numChunks-6,2))

    for i in range(len(avg)):
        avg[i,0]=A[0,i+3,1]
        for j in range(tSteps):
            avg[i,1]+=A[j,i+3,3]

        avg[i,1]/=tSteps

    ax = fig.add_subplot(111)
    ax.set_xlabel('Vz (m/s)')
    ax.set_ylabel('Height (nm)')
    ax.plot(avg[:,1]*1e5,avg[:,0]/10,'-^')

#fig4 = plt.figure(figsize=(10.,8.))
#get_velProfileZ(fig4)
#plt.savefig('Vz.png')

#------------------------- Vx along the height --------------------------------#


def get_velProfileX(fig):

    A=[]
    with open('vx.profile', 'r') as f:
        for line in f:
            if len(line.split())==4 and line.split()[0]!='#':
                A.append(line.split())

    A=np.asarray(A)
    global numChunks
    numChunks=int(A[-1,0])
    tSteps=int(len(A)/numChunks)

    A=np.reshape(A,(tSteps,numChunks,4))       #timesteps, chunks, value
    A=A.astype(np.float)

    global avg
    avg=np.zeros((numChunks-12,2))

    for i in range(len(avg)):
        avg[i,0]=A[0,i+6,1]
        for j in range(tSteps):
            avg[i,1]+=A[j,i+6,3]

        avg[i,1]/=tSteps

    slope=np.polyfit(avg[20:-20,0],avg[20:-20,1],1)

    ax = fig.add_subplot(111)
    ax.set_xlabel('Vx (m/s)')
    ax.set_ylabel('Height (nm)')
    ax.plot(avg[:,1]*1e5,avg[:,0]/10,'-^')

    return slope[0]


#fig5 = plt.figure(figsize=(10.,8.))
#get_velProfileX(fig5)
#plt.savefig('Vx.png')


#------------------------- Virial along the height --------------------------------#

#chunks = [1,2,3,4,5,6,7,8]
#plot_virial = np.loadtxt('virial.txt',skiprows=2,dtype=float)

#average_virial = []

#for i in chunks:
#    average_virial.append(np.mean(plot_virial[:,i]))

#figVirial = plt.figure(figsize=(10.,8.))

#plt.plot(np.asarray(chunks),np.asarray(average_virial)*1e-6,'-^', label= 'Virial Pressure along length')

#plt.xlabel('Region')
#plt.ylabel('Pressure (MPa)')
#plt.legend()
#plt.show()
#plt.savefig('virial.png')

#--------------------- Pressure along the channel length ----------------------#

#plot_lower = np.loadtxt('sigzzL.txt',skiprows=2,dtype=float)
#plot_upper = np.loadtxt('sigzzU.txt',skiprows=2,dtype=float)

#average_lower = []
#average_upper = []

#for i in chunks:
#    average_lower.append(np.mean(plot_lower[:,i]))
#    average_upper.append(np.mean(plot_upper[:,i]))

#fig3 = plt.figure(figsize=(10.,8.))

#plt.plot(np.asarray(chunks),np.asarray(average_lower)*1e-6,'-^', label= 'PressureL')
#plt.plot(np.asarray(chunks),np.asarray(average_upper)*1e-6,'-^', label= 'PressureU')

#plt.xlabel('Region')
#plt.ylabel('Pressure (MPa)')
#plt.legend()
#plt.show()
#plt.savefig('pDiff.png')

#--------------------- Virial pressure Vs. Surface traction ----------------------#

#plot_arr = np.loadtxt('virial-surface.txt',skiprows=2,dtype=float)

#time = plot_arr[:,0]
#virial_press = plot_arr[:,1]
#upper_press = plot_arr[:,2]
#lower_press = plot_arr[:,3]


#fig8 = plt.figure(figsize=(10.,8.))


#plt.plot(time*1e-6,np.asarray(virial_press)*1e-6,'-', label= 'Virial')
#plt.plot(time*1e-6,np.asarray(upper_press)*1e-6,'-', label= 'SurfU')
#plt.plot(time*1e-6,np.asarray(lower_press)*1e-6,'-', label= 'SurfL')


#plt.xlabel('Timesteps (10^6)')
#plt.ylabel('Pressure (MPa)')
#plt.legend()
#plt.show()
#plt.savefig('virial-surface.png')


def getTau():
    lower = np.loadtxt('stressL.txt',skiprows=2)
    upper = np.loadtxt('stressU.txt',skiprows=2)
    tau_lower = np.mean(lower,axis=0)[2]
    tau_upper = np.mean(upper,axis=0)[2]
    tau=0.5*(tau_lower-tau_upper)

    return tau

#slope = get_velProfileX(fig5)
#tau = getTau()
#eta = tau/slope

#with open('visocisty.txt', 'w') as f:
#    print("viscosity: %4.3f mPas" % (eta*1e-12), file=f)
