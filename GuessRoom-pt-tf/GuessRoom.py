from Agents_old import *
from ceiling import *


class GuessRoom(nn.Module):
    def __init__(self, agentA=None, agentB=None):
        super(GuessRoom, self).__init__()
        self.agentA = AgentA() if agentA is None else agentA
        self.agentB = AgentB() if agentB is None else agentB
        
        # 创建ConvA和ConvB模型实例
        self.room_embedding_A = ConvNet()
        self.room_embedding_B = ConvNet()
        
        # 加载预训练的模型参数
        #self.room_embedding_A.load_state_dict(torch.load('autoencoder_encoder.pth'))
        #self.room_embedding_B.load_state_dict(torch.load('autoencoder_encoder.pth'))

        # 将模型设置为评估模式
        #self.room_embedding_A.eval()
        #self.room_embedding_B.eval()
        
        
        #self.room_embedding_A = GridEmbedding()  # initial state
        #self.room_embedding_B = GridEmbedding()
        # self.room_embedding_B = self.room_embedding_A  # share

    def forward(self, env_info,tgt_rooms,obs_info=None, choose_method="sample", history_sents=None, env_ids=None, route_len=None):
        #batch_size*1 50
        tgt_rooms_arr = np.array(tgt_rooms).astype(int)
        #(50,9,3,3,3)
        #obs_info= env_info[:,:,2:5,2:5,:]
        #(50,9,50)->(50,50)
        room_embs_A = self.room_embedding_A(env_info)[np.arange(tgt_rooms_arr.shape[0]),tgt_rooms_arr,:]
        sent, token_probs = self.agentA.describe_room(room_embs_A, Param.max_sent_len, choose_method)
        if history_sents is not None:
            #(50,3,5)
            sent = torch.cat([history_sents, sent.unsqueeze(1)], dim=1)
            room_embs_B = self.room_embedding_B(env_info, env_ids=env_ids, route_len=route_len)
        else:
            #(50,9,50)
            room_embs_B = self.room_embedding_B(env_info)
        room_idx, room_prob = self.agentB.guess_room(room_embs_B, sent, choose_method)
        return room_idx, token_probs, room_prob, sent

    def backward(self, token_probs, room_prob, reward, step=0):
        lossA = self.agentA.cal_guess_room_loss(token_probs, reward)
        lossB = self.agentB.cal_guess_room_loss(room_prob, reward)
        lossA.backward()
        lossB.backward()
        return lossA, lossB