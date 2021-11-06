# !/bin/bash
# A shell script to install Fireworks
# Author: Mohamed Tarek

# Make sure Python 3 is installed as Python 2 has been discontinued!! FireWorks may continue to work with Python 2.7.x for some time, but this is no longer guaranteed.
# Type "python3". If working, go to the first step. If not, install python 3 preferably Download anaconda (Python 3 version) from https://www.anaconda.com/distribution/ and follow the installation instructions on the download page.

# Install the virtual environment to allow you to separate FWs installation from other Python installations. You may need administrator privileges. Contact your system administrator if you have questions about this.

sudo pip3 install python3-venv

# Setup the virtual environment
python3 -m venv $HOME/fireworks

# Activate the virtual environment
source $HOME/fireworks/bin/activate

# Install python packages in the virtual environment
pip install --upgrade pip
pip install pjson
pip install pyaml
pip install python-igraph   # For the igraph package, a C++ compiler and some additional development packages must be installed on your computer ( g++, python3-dev, libxml2-dev ). Could be installed with "sudo apt-get install <package>".

# Install Fireworks
pip install fireworks

###################################
##### Using Remote FireServer #####
###################################

# If you intend to run the rocket launchers on a remote computing cluster then you can configure a launchpad file for an existing FireServer.

mkdir .fireworks
cd .fireworks

# Create a launchpad configuration file "launchpad.yaml" in the folder "$HOME/.fireworks"
# Copy the following lines in the launchpad file

# Testing the connection to remote server (From FWs Documentation)
#host: ds049170.mongolab.com
#port: 49170
#name: fireworks
#username: test_user
#password: testing123

# Execute the following command
lpad -l $HOME/.fireworks/launchpad.yaml get_wflows

# To avoid specifying the launchpad file i.e. to write just "lpad get_wflows", we will need to create a configuration file

echo LAUNCHPAD_LOC: $HOME/.fireworks/launchpad.yaml >> $HOME/.fireworks/FW_config.yaml


# For real usage of FireWorks on a RemoteServer. When provided a hostname, username and password (In my case given to me by Mehdi from SCC at KIT). "Launchpad.yaml" shall be modified to include
#host: scs-mongodb.scc.kit.edu
#port: 27017
#name: <db-name>
#username: <user>
#password: <password>
#ssl:true
#ssl_ca_certs: <path to root CA certificate> (also an be downloaded from this URL http://www.gridka.de/ca/dd4b34ea.pem)

# Again this command shall work fine
lpad -l $HOME/.fireworks/launchpad.yaml get_wflows

# This time it will give "[]", since there are no workflows yet.
# The FireWorks on a remote FireServer are working fine!

# Note: To change the initial provided password, run the python file 'python pymongo_admin.py' in the folder 'Pymongo' found at 'https://git.scc.kit.edu/th7356/mongo_admin' and follow the instructions. For this script to work pymongo must be installed (if you have fireworks then it is installed).'

##################################
##### Using Local FireServer #####
##################################

#You need admin permissions to install MongoDB 
sudo apt-get install mongodb

#Option 1: (Instance created by Johannes) (PROBLEM:When running Python scripts, requires authentication)
#host: localhost
#port: 27017
#name: <db-name>
#username: <username>
#password: <password>

#### On NEMO, copy the ssh key into your NEMO .ssh directory

cd .ssh
cp /work/ws/nemo/fr_jh1130-md-0/ssh/sshclient-frrzvm .

#### On the Workstation/Laptop

cd .ssh
# Secure copy the file from NEMO to your PC .ssh directory
scp ka_lr1762@nemo.login1.freiburg.de:.ssh/sshclient-frrzvm .
# Make the priate key public to the instance with IP 132.230.102.164
ssh -i ~/.ssh/sshclient-frrzvm sshclient@132.230.102.164
# Test if the MongoDB is working fine by logging in into the MongoDB on the instance
mongo
# inside mongo terminal
#use <database name> 
#db.auth("<username>","<password>")

#If everything is fine, MongoDb shall output "1"

# In a new terminal, you can tunnel the connection to the instance from the local PC  (Also you can save this command in the .bashrc file)
ssh -N -i .ssh/sshclient-frrzvm -L 27017:localhost:27017 sshclient@132.230.102.164

# Open a new terminal and connect to MongoDB with the same Mongo commands as written on the instance, you shall get a "1" again
mongo

# Congrats, Now you can use FireWorks on a local FireServer !!

