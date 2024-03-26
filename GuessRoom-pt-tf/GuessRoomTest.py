import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from Agents_old import *
from Dataset import EnvDataset
from Param import *
import random
from GuessRoom import *

def guess_room_test(model):
    """
    evaluation
    :param model:
    :return:
    """
    model.eval()
    tgt = []; pred = []
    acc_test=[]
    dataset = EnvDataset(Param.test_env_dir)
    loader = DataLoader(dataset, batch_size=Param.batch_size)
    for step, data in enumerate(loader):
        # TODO maybe it is better to use this random way
        # tgt_rooms = np.random.randint(np.zeros(cur_obs_info.shape[0]), num_room)
        tgt_rooms = np.random.randint(np.zeros(data.shape[0]), Param.room_num)
        tgt.append(tgt_rooms)
        cur_obs_info = data.to(torch.float32)
        room_idxes, token_probs, room_probs, sent = model(cur_obs_info,tgt_rooms, choose_method="greedy")
        pred.append(room_idxes)
    tgt = np.concatenate(tgt, axis=0)
    pred = np.concatenate(pred, axis=0)
    acc_test=np.mean(tgt == pred)
        
    print("test acc = {}".format(acc_test))
        
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    setup_seed(50)
    task = GuessRoom()
    task.load_state_dict(torch.load('gr_model_gru.pth'))
    guess_room_test(task)
