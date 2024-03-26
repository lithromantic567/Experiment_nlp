import math

import torch
from Param import *
from torch import nn
from EnvGraph import *
import json


def _init_weights(m):
    if type(m) == nn.Linear:
        fanin = m.weight.data.size(0)
        fanout = m.weight.data.size(1)
        nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2.0/(fanin + fanout)))
class GridEmbedding(nn.Module):
    def __init__(self):
        super(GridEmbedding,self).__init__()
        self.cnn_po_room = nn.Sequential(
            nn.Conv2d(3,16,(3,3)),
            nn.ReLU(),
            nn.MaxPool2d((3,3)),
            nn.Conv2d(16,16,(2,2)),
            nn.ReLU()
        )
        self.cnn_po_obs = nn.Sequential(
            nn.Conv2d(3,16,(3,3)),
            nn.ReLU()
        )
        self.fcnn_grid=nn.Sequential(
            nn.Linear(Param.grid_emb_size_in,Param.grid_emb_size_out)
        )
        self.fcnn_po_room=nn.Sequential(
            nn.ReLU(),
            nn.Linear(9*Param.grid_emb_size_out, Param.room_emb_size)
        )
        self.fcnn_room=nn.Sequential(
            nn.ReLU(),
            nn.Linear(64*Param.grid_emb_size_out,Param.room_emb_size)
        )
        self.mlp_room=nn.Sequential(
            nn.Linear(16,Param.room_emb_size)
        )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, Param.room_emb_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64, Param.room_emb_size)
        )
        #self.fcnn_room=nn.Sequential(
        #    nn.ReLU(),
        #    nn.Linear(64*Param.grid_emb_size_out, Param.room_emb_size)
        #)
    def forward(self,env_info,env_ids=None, route_len=None,method=None):
        #grid_num=env_info.shape[2]*env_info.shape[3]
        #(50,9,9,3)
        #x=env_info.reshape((env_info.shape[0],env_info.shape[1],grid_num,env_info.shape[4]))
        #result=self.fcnn_grid(x)
        
        x=env_info
        '''
        x= x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])
        
        x=x.transpose(1,3).transpose(2,3)
        if method=="po":
            result=self.cnn_po_obs(x)
            result=result.reshape(result.shape[0],-1)
        else:
            result=self.cnn_po_room(x)
            result=result.reshape(result.shape[0],-1)
            #result=self.conv_layers(x)
            #result=result.view(result.size(0),-1)
        room_emb=self.mlp_room(result)
        #room_emb=self.fc_layers(result)
        room_emb=room_emb.reshape(env_info.shape[0],env_info.shape[1],Param.room_emb_size)
        '''
        x=x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3],x.shape[4])
        result=[]
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                data=x[i,j,:,:]
                result.append(self.fcnn_grid(data)) 
        result=torch.stack(result, dim=1)           
        
        if method=="po":
            result=result.reshape((Param.batch_size,Param.room_num,9*Param.grid_emb_size_out))
            room_emb=self.fcnn_po_room(result)
        else:
            #(50,9,90)
            result=result.reshape((Param.batch_size,Param.room_num,64*Param.grid_emb_size_out))
            #(50,9,50)
            room_emb=self.fcnn_room(result)
        
        #print(room_emb)
        return room_emb

class ObsEmbedding(nn.Module):
    def __init__(self):
        super(ObsEmbedding, self).__init__()
        self.fcnn_po = nn.Sequential(
            nn.ReLU(),
            nn.Linear(Param.subroom_emb_size_in, Param.room_emb_size)
        )
        self.fcnn_gate.apply(_init_weights)
        self.fcnn_obs.apply(_init_weights)
        

    def forward(self, sub_obs_info, sub_gate_info, method="cat", env_ids=None, route_len=None):
        #[50,4,10]
        obs_emb = self.fcnn_obs(sub_obs_info)
        gate_emb = self.fcnn_gate(sub_gate_info)
        if method == "cat":
            #[50,40]
            cat_obs_emb = obs_emb.reshape((Param.batch_size,  Param.max_subobs_num * Param.obs_feat_out_num))
            #[50,20]
            cat_gate_emb = gate_emb.reshape((Param.batch_size,Param.max_subgate_num * Param.gate_feat_out_num))
            #[50,60]
            room_emb = torch.cat((cat_obs_emb, cat_gate_emb), dim=1)
            # NOTE add another fcnn layer
            #[50,25]
            room_emb = self.fcnn_po(room_emb)
        else:
            print("there is no method called {}".format(method))
            raise NameError
        # NOTE mask for now
        # if env_ids is not None:
        #     ori_room_emb_shape = room_emb.shape
        #     room_emb = self.env_graph.cal_node_emb(env_ids, room_emb, route_len)
        #     assert ori_room_emb_shape == room_emb.shape
        return room_emb
    

class RoomEmbedding(nn.Module):
    def __init__(self):
        super(RoomEmbedding, self).__init__()
        self.fcnn_obs = nn.Sequential(
            nn.Linear(Param.obs_feat_in_num, Param.obs_feat_out_num)
        )
        self.fcnn_gate = nn.Sequential(
            nn.Linear(Param.gate_feat_in_num, Param.gate_feat_out_num)
        )
        self.fcnn_room = nn.Sequential(
            nn.ReLU(),
            nn.Linear(Param.room_emb_size_in, Param.room_emb_size)
        )
        self.fcnn_gate.apply(_init_weights)
        self.fcnn_obs.apply(_init_weights)
        # env_graph -> used for cal node emb with graph structure
        self.env_graph = EnvGraph()

    def forward(self, obs_info, gate_info, method="cat", env_ids=None, route_len=None):
        obs_emb = self.fcnn_obs(obs_info)
        gate_emb = self.fcnn_gate(gate_info)
        if method == "avg":
            avg_obs_emb = self._avg_emb(obs_emb)
            avg_gate_emb = self._avg_emb(gate_emb)
            assert avg_obs_emb.shape == (Param.batch_size, Param.max_room_num, Param.obs_feat_out_num)
            assert avg_gate_emb.shape == (Param.batch_size, Param.max_room_num, Param.gate_feat_out_num)
            #沿着第三维拼接
            room_emb = torch.cat((avg_obs_emb, avg_gate_emb), dim=2)
            room_emb = self.fcnn_room(room_emb)
        elif method == "cat":
            cat_obs_emb = obs_emb.reshape((Param.batch_size, Param.max_room_num, Param.max_obs_num * Param.obs_feat_out_num))
            cat_gate_emb = gate_emb.reshape((Param.batch_size, Param.max_room_num, Param.max_gate_num * Param.gate_feat_out_num))
            room_emb = torch.cat((cat_obs_emb, cat_gate_emb), dim=2)
            # NOTE add another fcnn layer
            room_emb = self.fcnn_room(room_emb)
        else:
            print("there is no method called {}".format(method))
            raise NameError
        # NOTE mask for now
        # if env_ids is not None:
        #     ori_room_emb_shape = room_emb.shape
        #     room_emb = self.env_graph.cal_node_emb(env_ids, room_emb, route_len)
        #     assert ori_room_emb_shape == room_emb.shape
        return room_emb
    def _avg_emb(self, emb):
        res = torch.mean(emb, dim=2)
        return res

    def _LSTM_emb(self, room_info):
        # TODO
        raise NotImplementedError
'''
class RoomEmbedding(nn.Module):
    def __init__(self):
        super(RoomEmbedding, self).__init__()
        self.fcnn_room = nn.Sequential(
            nn.Linear(Param.room_emb_size_in, Param.room_emb_size)
        )
        self.fcnn_room.apply(_init_weights)
        # env_graph -> used for cal node emb with graph structure
        self.env_graph = EnvGraph()

    def forward(self, obs_info, gate_info, method="cat", env_ids=None, route_len=None):
        obs_emb = obs_info
        gate_emb = gate_info
        if method == "avg":
            avg_obs_emb = self._avg_emb(obs_emb)
            avg_gate_emb = self._avg_emb(gate_emb)
            assert avg_obs_emb.shape == (Param.batch_size, Param.max_room_num, Param.obs_feat_in_num)
            assert avg_gate_emb.shape == (Param.batch_size, Param.max_room_num, Param.gate_feat_in_num)
            #沿着第三维拼接
            room_emb = torch.cat((avg_obs_emb, avg_gate_emb), dim=2)
            room_emb = self.fcnn_room(room_emb)
        elif method == "cat":
            cat_obs_emb = obs_emb.reshape((Param.batch_size, Param.max_room_num, Param.max_obs_num * Param.obs_feat_in_num))
            cat_gate_emb = gate_emb.reshape((Param.batch_size, Param.max_room_num, Param.max_gate_num * Param.gate_feat_in_num))
            room_emb = torch.cat((cat_obs_emb, cat_gate_emb), dim=2)
            # NOTE add another fcnn layer
            room_emb = self.fcnn_room(room_emb)
        else:
            print("there is no method called {}".format(method))
            raise NameError
        # NOTE mask for now
        # if env_ids is not None:
        #     ori_room_emb_shape = room_emb.shape
        #     room_emb = self.env_graph.cal_node_emb(env_ids, room_emb, route_len)
        #     assert ori_room_emb_shape == room_emb.shape
        return room_emb
'''

class ActionEmbedding(nn.Module):
    def __init__(self):
        super(ActionEmbedding,self).__init__()
        self.fcnn_action=nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 20)
        )
    def forward(self,actions_info):
        action_emb=self.fcnn_action(actions_info)
        return action_emb

class GateEmbedding(nn.Module):
    """
    TODO maybe it is better to share the fcnn_gate in RoomEmbedding
    """
    def __init__(self):
        super(GateEmbedding, self).__init__()
        self.fcnn_gate = nn.Sequential(
            nn.Linear(Param.gate_feat_in_num, Param.gate_feat_out_num)
        )
        self.fcnn_transform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(Param.gate_feat_out_num, Param.room_emb_size)  # NOTE same shape as RoomEmbedding
        )

    def forward(self, gates_info):
        gates_emb = self.fcnn_gate(gates_info)
        gates_emb = self.fcnn_transform(gates_emb)
        return gates_emb


class Utils(object):
    @staticmethod
    def construct_room_graph(env_ids, is_train=True):
        """
        :param env_ids:
        :return: room_graph[room][gate] = room [room_graph1, ....]
        """
        room_graph_dict = {}
        for cur_env_id in env_ids:
            if is_train and Param.is_dynamic_data is False: cur_path = "{}/env{}.txt".format(Param.env_dir, cur_env_id)
            elif is_train and Param.is_dynamic_data is True: cur_path = "{}/env{}.txt".format(Param.dynamic_env_dir, cur_env_id)
            else: cur_path = "{}/env{}.txt".format(Param.eval_env_dir, cur_env_id)
            with open(cur_path, 'r') as f:
                cur_env_info = json.load(f)
                cur_room_graph = Utils._room_graph(cur_env_info)
                room_graph_dict[int(cur_env_id)] = cur_room_graph
        return room_graph_dict

    @staticmethod
    def _room_graph(env_info):
        room_graph = {}
        for room_id, room_info in env_info.items():
            room_id_int = int(room_id)
            if room_id not in room_graph: room_graph[room_id_int] = {}
            gates_info = room_info["gates"]
            for gate_id, cur_gate_info in gates_info.items():
                neighbor_id_int = int(cur_gate_info["neighbor"])
                gate_id_int = int(gate_id)
                room_graph[room_id_int][gate_id_int] = neighbor_id_int
        return room_graph


