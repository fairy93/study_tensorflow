import os
import pandas as pd
os.chdir('face/1unret')

from PIL import Image

folder_list = os.listdir()
train_data = pd.DataFrame(columns = ['emotion','pixels','Usage'])

cnt=0
for folder in folder_list:
    cnt+=1
    img_name = folder

    img = Image.open(img_name)
    rawData = img.load()
    data = []
    for y in range(48):
        for x in range(48):
            data.append(rawData[x,y][0])
    StrA = " ".join(map(str, data))
    if cnt%5==0:
        train_data.loc[cnt] = (2,StrA,'PrivateTest')
    else:
        train_data.loc[cnt] = (2,StrA,'Training')

train_data.to_csv("train_converted.csv",index = False)


path_dir = 'face/val_mid/'
folder_list = os.listdir(path_dir)

index=1
for folder in folder_list:
    makeface(folder,index)
    index+=1