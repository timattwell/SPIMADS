import subprocess
import os
import time
import sys
import pandas
import numpy as np

def bb(x0=0, x1=0, x2=1, x3=-1, x4=0):
    #os.system("sh loadstar.sh")
    start_time = time.time()
    x0 = (x0+1)*10  #(0,20)
    x1 = (x1+1.2)*10   #(2,22)
    x2 = (x2+1)*1.5   #(0,3)
    x3 = (x3+2)*2   #(2,6)
    x4 = x4*180     #(-180,180)
    print(len(x0))
    print(len(x1))
    print(len(x2))
    print(len(x3))
    print(len(x4))
    samples = []
    for ii in range(len(x0)):
        samples.append([list(x0)[ii],list(x1)[ii],list(x2)[ii],list(x3)[ii],list(x4)[ii]])
    # length, pitch length, amplitude, offset distance, total angle
    print(samples)
    cmd1 = "./macro.sh"
    cmd2 = "./runSim.sh"
    processes = set()
    
    if len(x0) > 8:
        max_processes = 8
        bracket = 1
    elif len(x0) > 4:
        max_processes = 4
        bracket = 2
    elif len(x0) > 2:
        max_processes = 2
        bracket = 4
    else:
        max_processes = 1
        bracket = 8

    n = 0
    print("Creating geometry definitions and STAR CCM+ macros.")
    for vector in samples:
        time.sleep(1)
        print("process: "+str(n)+" - make csv")
        processes.add(subprocess.Popen([cmd1, str(n), str(vector[0]), str(vector[1]), str(vector[2]), str(vector[3]), str(vector[4])]))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update(
                [p for p in processes if p.poll() is not None])
        n=n+1
    #Check if all the child processes were closed
    for p in processes:
        if p.poll() is None:
            p.wait()
    print("Geo. def. and macros complete.")

    print("Beginning simulations.")
    processes = set()
    simtime = []
    c=0
    nodesfree={}
    for ii in range(max_processes):
        nodesfree.add(ii)

    for vector in samples:
        time.sleep(1)
        n = nodesfree.pop()
        processes.add(subprocess.Popen([cmd2, str(c), str(n)]))
        simtime.append(time.time())
        print("Running simulation "+str(c)+" on node ",str(n))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update(
                [p for p in processes if p.poll() is not None])
        nodesfree.add(n)
        c=c+1
        

    #Check if all the child processes were closed
    for p in processes:
        if p.poll() is None:
            p.wait()
            print("--- Sim "+str(n)+" done in %s seconds ---" % (time.time() - start_time))

    results=list()
    for ii in range(len(samples)):
        data = pandas.read_csv("./tmp/out/CostFuncDat"+str(ii)+".csv", header=0)
        x=data['CostFunc Monitor: CostFunc Monitor'].tolist()
        results.append(max(x))
    print(results)

    print("--- All done in %s seconds ---" % (time.time() - start_time))
    return results
