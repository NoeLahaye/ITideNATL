#!/usr/bin/python3

#from itertools import product
import subprocess
#from time import sleep

var = ["p","u","v"] # "p", "u", "v" or list of several
i_day = range(100, 105+1) #[19, 20] #range(7,29)
cluster_conf = {"p": {"NBODES":2, "TASKPNODE":8, "CPUPTASK":3},
                "u": {"NBODES":1, "TASKPNODE":12, "CPUPTASK":2},
                "v": {"NBODES":1, "TASKPNODE":12, "CPUPTASK":2}
                }
time = "auto" # str "hh:mm:ss" or float (hours) or "auto"
do_chain = True  #run job one after another. False or True or "times"

file_in = "job_proj_temp.sh"
file_out = "job_proj_{}_{}.sh"

### make sure we have lists
if isinstance(var, str):
    var = [var]
if isinstance(i_day, int):
    i_day = [i_day]

prev_id = None
### process job submission files
for it in i_day:
    dico = {"I_DAY":str(it)}
    if do_chain == "times":
        prev_id = None
    for va in var:
        dico.update({k:str(v) for k,v in cluster_conf[va].items()})
        if time == "auto":
            tim = 6. if va == "p" else 3.
            tim /= cluster_conf[va]["NBODES"]
        else:
            tim = time
        if isinstance(tim, (int,float)):
            tim = "{:02d}:{:02d}:00".format(int(tim),int((tim-int(tim))*60))
        dico.update({"VAR":va, "TIME":tim})
        # start generating batch submission file
        with open(file_in, "r") as f:
            t_str = f.read()
        for k,v in dico.items():
            t_str = t_str.replace(k,v)
        filout = file_out.format(va,it)
        with open(filout, "w") as f:
            f.write(t_str)
        print("generated script {} for variable {} and day {}".format(filout,va,it), end="; ")
        ### now launching job, possibly with dependency
        command = ["sbatch", "--parsable"]
        if prev_id is not None and do_chain:
            print("run after", prev_id, end="; ")
            command.append("--depend=afterok:{}".format(prev_id))
        command.append(filout)
        res = subprocess.run(command, stdout=subprocess.PIPE)
        if do_chain:
            prev_id = res.stdout.decode().rstrip("\n")
        print("JOB ID:", res.stdout)
    #var.append(var.pop(0)) ### rotate var list by one argument
    
