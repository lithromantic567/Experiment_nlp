import numpy as np
import random

import torch

from Dataset import *
from Agents_old import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from GuessAction import *
import matplotlib.pyplot as plt

def plotfig(fp):
    x=np.arange(0,Param.epoch,10 )
    y=[]
    with open(fp,'r') as f:
        data=f.read().strip().split()
        y.extend([float(i) for i in data])
    plt.plot(x,y)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("TRAIN")
    plt.show()
    plt.savefig("pictures/gr_train_gru.png")

if __name__ == "__main__":
    plotfig("results/gr_train_gru.txt")