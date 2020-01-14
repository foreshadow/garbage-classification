import glob
import os
import random
from collections import defaultdict

root_dir = os.path.expanduser('~/garbage_classify')

list_ = []
for file in glob.glob(f'{root_dir}/train_data/*.txt'):
    with open(file) as f:
        path, label = f.readline().split(', ')
    list_.append((str(path), int(label)))
counter = defaultdict(list)
for path, label in list_:
    counter[label].append(path)

n = int(.05 * len(list_)) // len(counter)
train = []
val = []

for label, paths in counter.items():
    random.shuffle(paths)
    train += [f'{path}, {label}' for path in paths[:-n]]
    val += [f'{path}, {label}' for path in paths[-n:]]

with open(f'{root_dir}/train.lst', 'w') as f:
    f.write('\n'.join(train))
with open(f'{root_dir}/val.lst', 'w') as f:
    f.write('\n'.join(val))

