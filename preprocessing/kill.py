import subprocess, getpass
#
username = getpass.getuser()
#
bashCommand = 'squeue -u aponte'
output = subprocess.check_output(bashCommand, shell=True)
#
for line in output.splitlines():
    lined = line.decode('UTF-8')
    if username in lined and 'dask' in lined:
        pid = lined.split()[0]
        bashCommand = 'scancel '+str(pid)
        print(bashCommand)
        boutput = subprocess.check_output(bashCommand, shell=True)

