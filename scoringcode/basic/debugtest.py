import json
import psutil
import os


info = psutil.virtual_memory()

data_dir = "data/scoring"
data_type="train"
data_path = os.path.join(data_dir, "data_{}.json".format(data_type))
shared_path = os.path.join(data_dir, "shared_{}.json".format(data_type))
with open(data_path, 'r') as fh:
    data = json.load(fh)

print (psutil.Process(os.getpid()).memory_info().rss)
print (info.total)
with open(shared_path, 'r') as fh:
    shared = json.load(fh)

print (psutil.Process(os.getpid()).memory_info().rss)
print (info.total)