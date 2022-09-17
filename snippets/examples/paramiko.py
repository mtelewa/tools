import paramiko

hostname = 'login1.nemo.uni-freiburg.de'
myuser = 'ka_lr1762'
mySSHK = '/home/mohamed/.ssh/id_rsa'
ssh = paramiko.SSHClient()

ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname, username=myuser, key_filename=mySSHK)

