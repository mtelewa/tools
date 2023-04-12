#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from sample_quality import block_ND as bd
from sample_quality import get_err
from scipy.stats import iqr
from scipy.stats import norm

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

class profile:
    """
    Returns a variable averaged over time in spatial bins
    or over spatial bins in each timestep.
    The average is either in the flow direction (x) or in the wall-normal direction (z)
    """

    def __init__(self,inputfile,tSkip,columns):

        # ------------------- Collecting the data -------------------------------#
        with open('info.dat', 'r') as f:
            for line in f:
                if line.split()[0]=='BoxHeight':
                    total_box_height=float(line.split()[1])
                if line.split()[0]=='gapHeight':
                    gapheight=float(line.split()[1])
                if line.split()[0]=='AChunk':
                    ChunkArea=float(line.split()[1])
                if line.split()[0]=='VChunkX':
                    ChunkVolX=float(line.split()[1])
                if line.split()[0]=='VChunkZ':
                    ChunkVolZ=float(line.split()[1])
                if line.split()[0]=='vShear':
                    vwall=float(line.split()[1])

        # total_box_height, gapheight, ChunkArea, ChunkVolX, ChunkVolZ, vwall = 1,1,1,1,1,1

        # Unit conversion
        atm_to_mpa=101325*1e-6
        kcalpermolA3_to_mpa=1/0.00014393
        if 'mflux' or 'all' in sys.argv:
            gramA_per_fs_to_gram_per_m2_ns = 1e26/(ChunkArea*gapheight)
        A_per_fs_to_m_per_s = 1e5

        A = []
        with open(inputfile, 'r') as f:
            for line in f:
                if len(line.split())==columns and line.split()[0]!='#':
                    A.append(line.split())

        numChunks=int(A[-1][0])
        tSteps=int(len(A)/numChunks)
        tSample=tSteps-tSkip

        tSteps_array=np.asarray([i for i in range(tSteps)])
        tSample_array=np.asarray([i for i in range(tSkip,tSteps)])

        # ------------------- Editing the data (Remove outliers)--------------------#

        # All measured values (e.g. all densities) array
        def remove_chunks(chunks):
            values_arr=np.asarray([A[i][3] for i in range(tSkip*chunks,len(A))]).astype(np.float)
            values_arr=np.reshape(values_arr,(tSample,chunks))
            # print(values_arr.shape)

            # the std deviation and mean of each chunk for all time steps
            std_dev=np.std(values_arr,axis=0)
            mean=np.mean(values_arr,axis=0)

            # Get the chunk ids to delete based on their mean (if the mean is zero)
            index=0
            idx=[]

            return mean,index,idx

        # A. Remove chunks with zero mean
        #--------------------------------
        mean,index,idx=remove_chunks(numChunks)

        for i in range(len(mean)):
            if mean[i]==0:
                idx.append(index)
            index+=1

        ignored_chunk_ids=[x+1 for x in idx]

        A=np.asarray(A).astype(np.float)

        # An array of chunks to delete
        delete_chunks=[]
        for i in range(len(A)):
            if A[i][0] in ignored_chunk_ids:
                delete_chunks.append(A[i])

        delete_chunks=np.asarray(delete_chunks)
        # Check how many chunks are deleted in one time step
        # print(len(delete_chunks)/tSteps)

        # The new chunk count
        newChunks=int((len(A)-len(delete_chunks))/tSteps)
        # print('Nchunks={}'.format(newChunks))

        # The difference between the 'all chunks' and 'chunks to delete' array
        def array_diff(B,C):
            cumdims = (np.maximum(B.max(),C.max())+1)**np.arange(C.shape[1])
            return B[~np.in1d(B.dot(cumdims),C.dot(cumdims))]

        # The old chunks count has to be different than the new chunks count
        if newChunks != numChunks:
            A=array_diff(A,delete_chunks)

        # B. Remove chunks not in the Inter Quartile Range after removing the
        # zero mean chunks
        #-------------------------------------------------------------------
        mean2,index2,idx2=remove_chunks(newChunks)

        if inputfile == 'tempZ.profile' or inputfile == 'vx.profile':
            iqr_value=iqr(mean2, axis=0)
            q1=np.quantile(mean2,0.25)
            q3=np.quantile(mean2,0.75)
            delete_lower= q1-1.5*iqr_value      # 1.5 is used to avoid excluding useful data
            delete_upper= q1+1.5*iqr_value
            #print(delete_lower,delete_upper)

            for i in range(len(mean2)):
                if mean2[i]<delete_lower or mean2[i]>delete_upper:
                    index2=A[i][0]
                    idx2.append(index2)

            ignored_chunk_ids2=[x for x in idx2]

            # An array of chunks to delete
            delete_chunks2=[]
            for i in range(len(A)):
                if A[i][0] in ignored_chunk_ids2:
                    delete_chunks2.append(A[i])

            delete_chunks2=np.asarray(delete_chunks2)

            newChunks= numChunks-(len(ignored_chunk_ids)) - (len(ignored_chunk_ids2))

            if len(delete_chunks2) != 0.0:
                A=array_diff(A,delete_chunks2)

        # The final array of values to use for computation and plotting
        A=np.reshape(A,(tSteps,newChunks,columns))
        B=A[:]
        A=A[tSkip:]
        scale = float(total_box_height)/float(gapheight)
        # print(A[-1]*scale)

        # Name the output files according to the input
        base=os.path.basename(inputfile)
        filename=os.path.splitext(base)[0]

        # ------------------- The averaging -------------------------------#

        # A. Spatial average in blocks of time -----------------

        time_avg_of_chunk=np.zeros((newChunks,columns-2))
        # Dimension array in nm
        time_avg_of_chunk[:,0]=A[0,:,1]
        dim = time_avg_of_chunk[:,0]/10


        if 'chunks' in sys.argv:
            time_avg_of_chunk[:,1]+=np.mean(bd(tSample,A[:,:,3],newChunks,100),axis=0)
            var=time_avg_of_chunk[:,1]

            # For the density profiles in x-direction, correct for LAMMPS bin volume calculation
            if inputfile == 'denX.profile':
                scale = float(total_box_height)/float(gapheight)
                var=var*scale

            elif inputfile == 'momentaX.profile':
                time_avg_of_chunk=np.zeros((newChunks,columns-1))
                time_avg_of_chunk[:,1]+=np.mean(bd(tSample,A[:,:,2],newChunks,100),axis=0)           # Count
                time_avg_of_chunk[:,2]+=np.mean(bd(tSample,A[:,:,3],newChunks,100),axis=0)          # momenta

                ChunkCountFluid = time_avg_of_chunk[:,1]
                momenta = time_avg_of_chunk[:,2]               # momenta averaged over time for each spatial bin
                var= ChunkCountFluid*momenta*gramA_per_fs_to_gram_per_m2_ns

            elif inputfile == 'vx.profile':
                var=var*A_per_fs_to_m_per_s
                # The shear rate is averaged only in the bulk (excludes the boundary points)
                slope_vx=np.polyfit(time_avg_of_chunk[4:-4,0],time_avg_of_chunk[4:-4,1],1)
                self.shear_rate=slope_vx[0]*1e15                         # in s^-1
                vx_at_lower_wall=time_avg_of_chunk[0][1]
                self.slip_length=(vx_at_lower_wall/slope_vx[0])*0.1      # in nm
                vx_at_upper_wall=time_avg_of_chunk[-1][1]
                self.slip_velocity=vwall-vx_at_upper_wall

            elif inputfile == 'virialChunkZ.profile' or inputfile == 'virialChunkX.profile':

                n=100
                time_avg_of_chunk=np.zeros((newChunks,columns-1))

                time_avg_of_chunk[:,1]+=np.mean(bd(tSample,A[:,:,2],newChunks,n),axis=0)  #  Count
                time_avg_of_chunk[:,2]+=np.mean(bd(tSample,A[:,:,3],newChunks,n),axis=0)  #  W1
                time_avg_of_chunk[:,3]+=np.mean(bd(tSample,A[:,:,4],newChunks,n),axis=0)  #  W2
                time_avg_of_chunk[:,4]+=np.mean(bd(tSample,A[:,:,5],newChunks,n),axis=0)  #  W3

                ChunkCountBulk = time_avg_of_chunk[:,1]
                W1 = time_avg_of_chunk[:,2]
                W2 = time_avg_of_chunk[:,3]
                W3 = time_avg_of_chunk[:,4]
                if inputfile == 'virialChunkX.profile':
                    #dim = dim[:-1]
                    var=-ChunkCountBulk*(W1+W2+W3)*atm_to_mpa/(3*ChunkVolX)
                else:
                    var=-ChunkCountBulk*(W1+W2+W3)*atm_to_mpa/(3*ChunkVolZ)

            elif inputfile == 'fzL.profile':
                dim = dim[1:-1]
                var=time_avg_of_chunk[1:-1,1]*kcalpermolA3_to_mpa/(ChunkArea)
                self.sigzzL=var

            elif inputfile == 'fzU.profile':
                dim = dim[1:-1]
                var=time_avg_of_chunk[1:-1,1]*kcalpermolA3_to_mpa/(ChunkArea)
                self.sigzzU=var
                self.length=time_avg_of_chunk[1:-1,0]/10

            np.savetxt(filename+'.txt', np.c_[dim,var],
                        delimiter="  ",header="dim           var")

        # B. Temporal Averaging for all the chunks -----------------

        elif 'time' in sys.argv:
            time_avg_of_all_chunks=np.zeros((tSample,columns-2))

            # Timesteps
            time_avg_of_all_chunks[:,0]=tSample_array[:]       # first entry: time
            time=time_avg_of_all_chunks[:,0]

            for i in range(newChunks):
                time_avg_of_all_chunks[:,1]+=(A[:,i,3])       # second entry: state variable
            time_avg_of_all_chunks[:,1]/=newChunks            # Average over the bins for each time step

            var=time_avg_of_all_chunks[:,1]
            if inputfile == 'momentaX.profile':
                var=var*gramA_per_fs_to_gram_per_m2_ns*3652

            if inputfile == 'profile.gk.2d':
                self.var=var

            self.time = time

            np.savetxt(filename+'-time.txt', np.c_[time,var],
                        delimiter="  ",header="time           var" )


                # # FOR ONE CHUNK (1st chunk in this case)

                # ChunkCountBulk = bd(tSample,A[:,:,2],newChunks,n)[:,0]
                # a=bd(tSample,A[:,:,3],newChunks,n)[:,0]
                # b=bd(tSample,A[:,:,4],newChunks,n)[:,0]
                # c=bd(tSample,A[:,:,5],newChunks,n)[:,0]
                # d=np.asarray(-ChunkCountBulk*(a+b+c)*atm_to_mpa/(3*ChunkVolX))
                # get_err(d)

                # # FOR ALL CHUNKS

                #ChunkCountBulk = bd(tSample,A[:,:,2],newChunks,n)
                #a=bd(tSample,A[:,:,3],newChunks,n)
                #b=bd(tSample,A[:,:,4],newChunks,n)
                #c=bd(tSample,A[:,:,5],newChunks,n)
                #d=-ChunkCountBulk*(a+b+c)*atm_to_mpa/(3*ChunkVolX)

                # For all the chunks
                #np.savetxt('press-error3.txt', np.c_[dim,get_err(d)[0],get_err(d)[1]],
                #            delimiter="  ",header="x  p(x)  error" )
