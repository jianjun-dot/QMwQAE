import os
from datetime import datetime
import numpy as np
import json


def create_new_directory(dir_name: str):
    curr_dir = os.getcwd()
    try:
        os.makedirs(dir_name)
        print("directory " + dir_name +" created in " + curr_dir)
    except FileExistsError:
        print("directory " + dir_name + " already exists")
    
def get_time_stamp():
    return (datetime.today().strftime('%Y-%m-%d'), datetime.today().strftime('%H-%M-%S'))

def save(data, header, sim_params):
    # create directory
    time_stamp = get_time_stamp()
    new_dir = sim_params.data_dir + time_stamp[0] + "/" + time_stamp[1]
    create_new_directory(new_dir)
    # dump files
    fname = sim_params.get_fname()
    dump_json(new_dir +"/"+ fname, sim_params.__dict__)
    np.savetxt(new_dir +"/"+ fname + ".csv", data, delimiter = ',', header = header)
    

def dump_json(fname, json_object):
    with open(fname, "w") as out_file:
        json.dump(json_object, out_file)

if __name__ == "__main__":
    pass