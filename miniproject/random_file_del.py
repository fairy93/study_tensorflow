import os
import numpy as np

path_dir = os.listdir('face/train_neutral')

selected_dogs = np.random.choice(path_dir, 500)

for file_ in selected_dogs:
    try:
        os.remove('face/train_neutral/'+file_)
    except:
        print('except')
        pass
