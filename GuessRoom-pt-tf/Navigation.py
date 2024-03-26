import torch
from torch import nn
from Agents_old import *
from GuessAction import *
from GuessRoom import *


class Navigation(nn.Module):
    def __init__(self):
        super(Navigation, self).__init__()
        self.elu_A = ELU_A(); self.elg_A = ELG_A()
        # self.elu_B = ELU(); self.elg_B = ELG()  # dont share
        self.elu_B = ELU_B(); self.elg_B = ELG_B()
        self.agentA = AgentA(self.elu_A, self.elg_A)
        self.agentB = AgentB(self.elu_B, self.elg_B)
        self.guess_action_task = GuessAction(self.agentA, self.agentB)
        self.guess_room_task = GuessRoom(self.agentA, self.agentB)

    def _move(self,env_info, agentA_pos, agentA_dir,action_idx):   
        #agentAdir---0:up 1:right 2:down 3:left
        #3*3
        next_agentA_dir=[]
        next_agentA_pos=[]
        for cur_action,cur_agentA_dir,cur_agentA_pos in zip(action_idx,agentA_dir,agentA_pos):
            col=cur_agentA_pos[0]
            row=cur_agentA_pos[1]           
            #根据动作agent变方位或者位置
            if cur_action==0: #left
                next_agentA_dir.append((cur_agentA_dir+3)%4)
            elif cur_action==1: #right
                next_agentA_dir.append((cur_agentA_dir+1)%4)
            elif cur_action==2: #forward
                next_agentA_dir.append(cur_agentA_dir)
                if cur_agentA_dir in [0,2]:
                    if cur_agentA_dir==0:
                        row=max(cur_agentA_pos[1]-1,1)   
                    else:
                        row=min(cur_agentA_pos[1]+1,6)
                elif cur_agentA_dir in [1,3]:
                    if cur_agentA_dir==1:
                        col=min(cur_agentA_pos[0]+1,6)
                    else:
                        col=max(cur_agentA_pos[0]-1,1)
            next_agentA_pos.append([col,row])
            #根据目前的方位和位置得到部分可观测的环境信息
        new_obs_info=self._getObs(env_info,next_agentA_pos,next_agentA_dir)
            #env_info:[env_id,room_id,col,row,emb]
                  
        return new_obs_info
    def _getObs(self,env_info,agentA_pos,agentA_dir):
        obs_info=[]
        for i in range(env_info.shape[0]):
            top=agentA_pos[i][1];down=agentA_pos[i][1]
            left=agentA_pos[i][0];right=agentA_pos[i][0]               
            if agentA_dir[i] in [0,2]:   
                left=max(left-1,0)
                right=min(right+1,7)
                if agentA_dir[i]==0:
                    top=max(top-2,0)
                else:
                    down=min(down+2,7)
            elif agentA_dir[i] in [1,3]:
                top=max(top-1,0)
                down=min(down+1,7)
                if agentA_dir[i]==1:
                    right=min(right+2,7)
                else:
                    left=max(left-2,0)
            #env_info:[env_id,room_id,col,row,emb]
            #--pad--
            obs=env_info[i,:,left:right+1,top:down+1,:]
            #(9,3,3,3)
            obs=np.pad(obs,((0,0),(0,3-obs.shape[1]),(0,3-obs.shape[2]),(0,0)),'constant',constant_values=-1)
            obs_info.append(obs)
        obs_info=torch.tensor(np.array(obs_info)).float()
        return obs_info 
    def forward(self,env_info,actions_info, tgt_rooms, max_move_len, agentA_dir,agentA_pos,is_train=True, choose_method="sample"):
        """
        TODO have not dealt with early stop problem, A should know when it should stop, because it could see the goal at the right room
        TODO it is also possible to add an room emb which means end of routes, add a action emb which means do not choose any action
        NOTE just ignore rooms and actions after the tgt rooms, and loss will not count them in
        guess room, guess action, guess room, ....
        :param env_ids: used for reading env files, then construct graph
        :param room_graph: room_graph[room_id][action_id] = neighbor_id, then env can track the real path of A
        :param cat_info: (obs_info, action_info)
        :param start_room: B does not know where the start room is
        :param goal_room: B knows where the goal room is
        :param max_move_len: max num of moving actions of A
        :param is_train:
        if True, the input of guess action is the real pos of A.
        if False, the input of guess action is the output of last guess room
        :return:
        action probs -> (token_prob, guess_prob),
        total route (used for cal loss, a strong signal)
        sents (analysis)
        """
        token_probs_room = []; token_probs_action = []
        guess_probs_room = []; guess_probs_action = []
        actual_route_action=[]
        sents_room = []; sents_action = []
        total_guess_room_idx = []; total_guess_action_idx = []
        is_right_room = []; is_right_action = []
        #cur_env_info,cur_actions_info
        #assert obs_info.shape == (obs_info.shape[0], Param.max_room_num, Param.max_obs_num, Param.obs_feat_in_num)
        #assert action_info.shape == (action_info.shape[0], Param.max_room_num, Param.max_action_num, Param.action_feat_in_num)
        #real_room_A =tgt_rooms # ; actual_route_room.append(real_room_A)
        history_sents_room = None; history_sents_action = None
        #list
        obs_info= self._getObs(env_info,agentA_pos,agentA_dir)
        #(50,9,3,3,3)
        #obs_info=torch.tensor(np.array(obs_info)).float()
        for cur_step in range(max_move_len):            
            # ----- Guess Room ----
            # TODO do not add history sents
            # guess_room_idx, cur_token_probs_room, cur_room_prob, cur_sent_room = self.guess_room_task(obs_info, action_info, real_room_A, history_sents=None, env_ids=env_ids, route_len=cur_step + 1)
            if is_train:
                guess_room_idx, cur_token_probs_room, cur_room_prob, cur_sent_room = self.guess_room_task(env_info,tgt_rooms,obs_info, history_sents=history_sents_room)
            else:
                guess_room_idx, cur_token_probs_room, cur_room_prob, cur_sent_room = self.guess_room_task(env_info, tgt_rooms,obs_info, history_sents=history_sents_room, choose_method=choose_method)
            #(50,1,5)(50,2,5)
            history_sents_room = cur_sent_room  # include all history sents
            if len(history_sents_room.shape) == 2: history_sents_room = history_sents_room.unsqueeze(1)
            if is_train:
                token_probs_room.append(cur_token_probs_room); guess_probs_room.append(cur_room_prob[0])
            sents_room.append(cur_sent_room)
            total_guess_room_idx.append(guess_room_idx)
            # ------ is right room ----
            cur_is_right_room = torch.zeros_like(guess_room_idx)
            #([50])
            cur_is_right_room[torch.Tensor(tgt_rooms) == guess_room_idx] = 1
            
            is_right_room.append(cur_is_right_room)
            # ----- Route Plan ----
            # TODO in training process, input of guess action should be the real pos
            # TODO in test process, input should be the res of guess room task
            #if is_train is True:
            #    cur_next_doors, cur_expected_next_rooms = self.agentB.next_movement(env_ids, torch.Tensor(real_room_A), goal_room)
            #else:
            #    cur_next_doors, cur_expected_next_rooms = self.agentB.next_movement(env_ids, guess_room_idx, goal_room)
            #guess_room_actions_info = action_info[np.arange(action_info.shape[0]), guess_room_idx, :, :]
            #assert guess_room_actions_info.shape == (guess_room_actions_info.shape[0], Param.max_action_num, Param.action_feat_in_num)
            # ----- Guess action ----
            # TODO in training process, input of guess action should be the real pos
            # TODO in test process, input should be the res of guess room task
            #real_room_actions_info = action_info[np.arange(action_info.shape[0]), real_room_A, :, :]
            #real_room_actions_num = actions_num[np.arange(action_info.shape[0]), real_room_A]
            #assert real_room_actions_info.shape == (real_room_actions_info.shape[0], Param.max_action_num, Param.action_feat_in_num)
            #assert real_room_actions_num.shape == (real_room_actions_info.shape[0],)
            cur_action_idx = np.random.randint(np.zeros(actions_info.shape[0]), Param.action_num)
            if is_train is True:
                guess_action_idx, cur_token_probs_action, cur_action_prob, cur_sent_action = self.guess_action_task(actions_info, cur_action_idx,  history_sents=None)
            else:
                guess_action_idx, cur_token_probs_action, cur_action_prob, cur_sent_action = self.guess_action_task(actions_info, cur_action_idx,  choose_method=choose_method, history_sents=None)
            history_sents_action = cur_sent_action
            if len(history_sents_action.shape) == 2: history_sents_action = history_sents_action.unsqueeze(1)
            if is_train is True:
                token_probs_action.append(cur_token_probs_action); guess_probs_action.append(cur_action_prob[0])
            sents_action.append(cur_sent_action)
            total_guess_action_idx.append(guess_action_idx)
            # ----- is right action ----
            cur_is_right_action = torch.zeros_like(guess_action_idx)
            cur_is_right_action[torch.Tensor(cur_action_idx) == guess_action_idx] = 1        
            is_right_action.append(cur_is_right_action)
            # ----- Actual Movement ---- -> update real_room_A
            # 根据真实的动作移动
            obs_info = self._move(env_info,  agentA_pos, agentA_dir,cur_action_idx)
            
            actual_route_action.append(cur_action_idx)
        # return (total_guess_room_idx, total_guess_action_idx), (token_probs_room, token_probs_action), (guess_probs_room, guess_probs_action), (actual_route_room, actual_route_action), (sents_room, sents_action)
        is_right_room = torch.stack(is_right_room); is_right_action = torch.stack(is_right_action)
        return (sents_room, sents_action), actual_route_action, (token_probs_room, token_probs_action), (guess_probs_room, guess_probs_action), (is_right_room, is_right_action)

    def backward(self, token_probs, guess_probs, rewards):
        """
        :param token_probs: (token_probs_room, token_probs_action)
        :param guess_probs: (room_prob, action_prob)
        :param rewards: (reward_room, reward_action)
        :return:
        """
        token_probs_room, token_probs_action = token_probs
        room_prob, action_prob = guess_probs
        reward_room, reward_action = rewards
        lossA_room = []; lossB_room = []; lossA_action = []; lossB_action = []
        for cur_step in range(Param.max_move_len):
            cur_token_prob_room = token_probs_room[cur_step]; cur_token_prob_action = token_probs_action[cur_step]
            cur_room_prob = room_prob[cur_step]; cur_action_prob = action_prob[cur_step]
            cur_room_reward = reward_room[cur_step]; cur_action_reward = reward_action[cur_step]

            cur_lossA_room, cur_lossB_room = self.guess_room_task.backward(cur_token_prob_room, cur_room_prob, cur_room_reward)
            cur_lossA_action, cur_lossB_action = self.guess_action_task.backward(cur_token_prob_action, cur_action_prob, cur_action_reward)
            lossA_room.append(cur_lossA_room); lossB_room.append(cur_lossB_room)
            lossA_action.append(cur_lossA_action); lossB_action.append(cur_lossB_action)
        return (lossA_room, lossB_room), (lossA_action, lossB_action)
