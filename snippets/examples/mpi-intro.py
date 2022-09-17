
#Message Passing Interface (MPI):
#message-passing system designed to function on a wide variety of parallel computers
# Main goal is  to estavlish a portable efficient and flexible standard for Message
# passing that will be widely used for wrtiing mesage passing programs.


#
#
# A buffer, also called buffer memory, is a portion of a computer's memory that is
# set aside as a temporary holding place for data that is being sent to or received
# from an external device, such as a hard disk drive (HDD), keyboard or printer.
#
#
#
#
# Shared memory and distributed memory
#
# shared memory:
# splitting the problem among many processors.
# Limited scalability and high overhead
#
# distributed memory:
# each memory has its own workspace to work on a portion of the problem
#
# each coupled cpu-memory is connected to a network, allowing comm between members
#
# team of workers per table > cores per node
#
# simplifies the process of dealing with multiple processors over a network.
# instead of focusing on memory mangement we concentrate on the proceesing
#
#
# MPI_COMM_WORLD: all mpi procs are grouped into a unified comm env
# A communicator, each communicator contains a list of process
# All procs should share a communicator.
#
# Rank: the processes are ordered or given ranks. no. of each proc is a rank.
# Point to point communication: a proc sends to another.



# Collective comm: a group of procs are communicating at one time.
#
#
# communication for generic py objects (dictionaries,lists,etc) > send(), receive()
# communication for buffer-provider objects (NumPy arrays) > Send(), Receive()
#
# MPI.Comm is the base class of communicators


from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print (size)

print("Hello! I am rank %d out of %d processes running in total running on %s"
        %(comm.rank,comm.size,MPI.Get_processor_name()))

comm.Barrier()

# Collective comm

# * Broadcasting > 1 proc send a msg to every other procs inside the communicator
# * Scatter > 1 proc splits msg to several parts and send individual parts to several procs
# * Gather > inverse of scatter
# * All gather > extension to gather, where gather is done for all procs. Each procs ends
#                 up with the overall info
#
#
# MPI reduce: reduces several values into a single value
