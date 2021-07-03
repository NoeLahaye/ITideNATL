#!/usr/bin/python3

from itertools import product
import subprocess

var = "v" # "u", "v" or "p"
i_day = range(5,6)
n_nodes = 2
task_per_node = 12
cpu_per_task = 2
time = "auto" # str "hh:mm:ss" or float (hours) or "auto"

file_in = "job_proj_temp.sh"
file_out = "job_proj_{}_{}.sh"

dico = {"NBODES":str(n_nodes),
        "TASKPNODE":str(task_per_node),
        "CPUPTASK":str(cpu_per_task),
        }

### make sure we have lists
if isinstance(var, str):
    var = [var]
if isinstance(i_day, int):
    i_day = [i_day]

### process job submission files
for va,it in product(var, i_day):
    if time == "auto":
        tim = 4. if va == "p" else 2.5
        tim /= n_nodes
    else:
        tim = time
    if isinstance(tim, (int,float)):
        tim = "{:02d}:{:02d}:00".format(int(tim),int((tim-int(tim))*60))
    dico.update({"VAR":va, "I_DAY":str(it), "TIME":tim})
    with open(file_in, "r") as f:
        t_str = f.read()
    for k,v in dico.items():
        t_str = t_str.replace(k,v)
    filout = file_out.format(va,it)
    with open(filout, "w") as f:
        f.write(t_str)
    print("generated script {} for variable {} and day {}".format(filout,va,it))
    res = subprocess.run(["sbatch", filout], stdout=subprocess.PIPE)
    print("SBATCH:", res.stdout)

