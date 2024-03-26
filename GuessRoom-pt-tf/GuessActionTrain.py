import numpy as np
import random

import torch

from Dataset import *
from Agents_old import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from GuessAction import *


def guess_action_train():
    """
    NOTE warm-up for stage3
    :return:
    """
    dataset = EnvDataset(Param.env_dir)
    loader = DataLoader(dataset, batch_size=Param.batch_size)
    task = GuessAction()
    opt = Adam(task.parameters(), lr=Param.lr_task, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    accum_tgt = []; accum_pred = []
    e_acc = []
    for i in range(Param.epoch):
        tgt = []; pred = []
        total_loss_A = 0; total_loss_B = 0
        opt.zero_grad()
        for step, data in enumerate(loader):
            task.train(); task.agentA.train(); task.agentB.train()
            #assert cur_gate_info.shape == (cur_gate_info.shape[0], Param.max_room_num, Param.max_gate_num, Param.gate_feat_in_num)
            #assert cur_num_gate.shape == (cur_num_gate.shape[0], Param.max_room_num)
            # --- FORWARD ---
            # NOTE only samples a few rooms as the rooms where the agent A is,
            # NOTE anther way is that new_batch = batch * max_room_num
            # select rooms
            #cur_rooms = np.random.randint(np.zeros(data.shape[0]), action_num)
            #selected_rooms_gates_info = cur_gate_info[np.arange(cur_rooms.shape[0]), cur_rooms, :, :]
            #selected_rooms_gates_num = cur_num_gate[np.arange(cur_rooms.shape[0]), cur_rooms]
            #assert selected_rooms_gates_info.shape == (num_room.shape[0], Param.max_gate_num, Param.gate_feat_in_num)
            #assert selected_rooms_gates_num.shape == (num_room.shape[0], )
            # select gates
            tgt_actions = np.random.randint(np.zeros(data.shape[0]), Param.action_num)
            tgt.append(tgt_actions)
            #(50,3,1)
            actions_info=np.array([[0],[1],[2]]*Param.batch_size).reshape((Param.batch_size,3,1))
            actions_info=torch.tensor(actions_info).float()
            #print(actions_info)
            
            action_idxes, token_probs, action_probs, sent = task(actions_info, tgt_actions)
            pred.append(action_idxes)
            # --- BACKWARD ---
            reward = np.ones_like(action_idxes)
            reward[action_idxes.numpy() != tgt_actions] = -1

            reward *= Param.reward
            cur_loss_B, cur_loss_A = task.backward(token_probs, action_probs[0], reward.tolist())
            total_loss_A += cur_loss_A; total_loss_B += cur_loss_B
        opt.step()
        tgt = np.concatenate(tgt, axis=0); pred = np.concatenate(pred, axis=0)
        accum_tgt.append(tgt); accum_pred.append(pred)
        if i % 50 == 0:
            task.eval(); task.agentA.eval(); task.agentB.eval()
            accum_pred = np.concatenate(accum_pred, axis=0)
            accum_tgt = np.concatenate(accum_tgt, axis=0)
            print("epoch{}: \nacc={}, loss A={}, loss B={}".format(i, np.mean(accum_tgt == accum_pred), total_loss_A, total_loss_B))
            accum_pred = []; accum_tgt = []
            e_acc.append(guess_gate_evaluate(task))
    return e_acc


def guess_gate_evaluate(model):
    model.eval()
    with torch.no_grad():
        tgt = []; pred = []
        dataset = EnvDataset(Param.eval_env_dir)
        loader = DataLoader(dataset, batch_size=Param.batch_size)
        for step, data in enumerate(loader):
            tgt_actions = np.random.randint(np.zeros(data.shape[0]), Param.action_num)
            tgt.append(tgt_actions)
            actions_info=np.array([[0],[1],[2]]*Param.batch_size).reshape((Param.batch_size,3,1))
            actions_info=torch.tensor(actions_info).float()
            action_idxes, token_probs, action_probs, sent = model(actions_info, tgt_actions, "greedy")
            pred.append(action_idxes)
        tgt = np.concatenate(tgt, axis=0); pred = np.concatenate(pred, axis=0)
        #print("eval acc = {}".format(np.mean(tgt == pred)))
        eval_acc = np.mean(tgt == pred)
        with open("Eval_acc.txt",'a') as fp:
            fp.write(str(eval_acc)+'\n')
        print("eval acc = {}".format(eval_acc))
        return eval_acc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    setup_seed(0)
    e_acc = guess_action_train()
    print(e_acc)