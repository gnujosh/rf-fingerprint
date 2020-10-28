# -----------------------------------------------------------------------------------
# ---------------------------------- UNCLASSIFIED -----------------------------------
# This code was developed under the Defense Advanced Research Projects Agency (DARPA)
# Radio Frequency Machine Learning System (RFMLS) program: contract FA8750-18-C-0150.
# Government Unlimited Rights
# BAE Systems, Inc. (C) 2020.
# -----------------------------------------------------------------------------------
"""gpu scheduler for working on BAE SYSTEMS mabnunxlssep gpu servers."""

import os
import time

def reserve_gpu_resources(
    numgpus=1, selectfrom=["0", "1", "2", "3", "4", "5", "6", "7"]
):
    """Reserve gpu resources for a job.
    
    Key Word Arguments

        numgpus: (integer) number of gpus to reserve for the job

        selectfrom: (list of gpu ids) list of gpu ids to select
            from. gpu ids must be a string integer 0-7.

    Returns

        claimedGpus: (list of gpu ids) gpu ids claimed for job

        statusFiles: (list of strings) list containing paths to
            the status files written. Status files indicate a gpu
            is in use by the job.

    Note: The release_gpu_resources function should be used to release
        the reserved gpu resources. Also safest to use try and except
        clauses so that status files don't get created and then never
        deleted when Exceptions get raised.
    """
    begin_experiment = False
    claimedGpus = []
    statusFiles = []
    print("\n********************************")
    print("* Waiting for {} gpu(s)...".format(numgpus))
    while not begin_experiment:
        for gpu_i in selectfrom:
            status_filename = "/tmp/.gpu_in_use_" + gpu_i
            if os.path.isfile(status_filename):
                # If 'status_filename' already exists, then check next gpu
                print("* Status gpu_{} : in use".format(gpu_i))
                time.sleep(1)
            else:
                # If 'status_filename' does not exist, create the file and use the gpu.
                f = open(status_filename, "w+")
                f.close()
                claimedGpus.append(gpu_i)
                statusFiles.append(status_filename)
                print("* Status gpu_{} : free, claiming it now".format(gpu_i))
                if len(claimedGpus) == numgpus:
                    break
        if len(claimedGpus) == numgpus:
            begin_experiment = True
        else:
            print("* looking for {} free gpu(s)".format(numgpus - len(claimedGpus)))

    print("* Using gpus: {}".format(claimedGpus))
    print("* wrote: {}".format(statusFiles))
    print("********************************\n")

    return claimedGpus, statusFiles


def release_gpu_resources(claimedGpus, statusFiles):
    """Release gpu resources for a job that is finished.
    
    Arguments

        claimedGpus: (list of gpu ids) gpu ids claimed for job

        statusFiles: (list of strings) list containing paths to the status
            files written. Status files indicate a gpu is in use by the job.

    Note: The release_gpu_resources function should be used to release gpu
        resources reserved by the reserve_gpu_resources function. Also safest
        to use try and except clauses so that status files don't get created
        and then never deleted when Exceptions get raised.
    """
    print("\n********************************")
    print("* Releasing gpu(s): {}".format(claimedGpus))
    print("********************************\n")
    for f in statusFiles:
        os.remove(f)
