import os, shutil
import pandas as pd

train = pd.read_csv('data/train.csv')
train.set_index('id', inplace=True)

G='data/glasses/G/'

NoG='data/glasses/NoG/'

os.makedirs(G, exist_ok=True)

os.makedirs(NoG, exist_ok=True)

folder = "data/faces-spring-2020/faces-spring-2020/"

for i in range (1,4501):

    oldpath=f"{folder}face-{i}.png"

    if train.loc[i]['glasses']==0:
        newpath=f"{NoG}face-{i}.png"
    elif train.loc[i]['glasses']==1:
        newpath=f"{G}face-{i}.png"

    shutil.move(oldpath, newpath)
