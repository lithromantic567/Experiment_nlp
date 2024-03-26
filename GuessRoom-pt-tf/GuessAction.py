from Agents_old import *


class GuessAction(nn.Module):
    def __init__(self, agentA=None, agentB=None):
        super(GuessAction, self).__init__()
        self.agentA = AgentA() if agentA is None else agentA
        self.agentB = AgentB() if agentB is None else agentB
        self.action_embedding_A = ActionEmbedding()
        self.action_embedding_B = ActionEmbedding()

    def forward(self, actions_info, tgt_actions_idx, choose_method="sample", guess_actions_info=None, history_sents=None):
        """
        :param gates_info: (batch, max_gate_num, gate_feat_num)
        :param tgt_gates_idx: (batch)
        :param choose_method:
        :param guess_gates_info: if rooms of A is not consistent with room which B thought,
        then guess_gates_info describes rooms that B thought
        :param gates_num: num of gates in each room
        :param history_sents: (history_sents_room, history_sents_gate)
        :return:
        """
        # TODO need to check whether this mask way is right
        # --- split tgt & distractor ---
        cur_batch_size = len(tgt_actions_idx)
        #assert gates_info.shape == (cur_batch_size, Param.max_gate_num, Param.gate_feat_in_num)
        # --- forward ---
        if guess_actions_info is not None:
            ordered_actions_emb_B = self.action_embedding_B(guess_actions_info)[np.arange(cur_batch_size), tgt_actions_idx, :]
        else:
            ordered_actions_emb_B = self.action_embedding_B(actions_info)[np.arange(cur_batch_size), tgt_actions_idx, :]
        # unitize emb size of gate(stage1) and room(stage3)
        #assert ordered_gates_emb_B.shape == (cur_batch_size, Param.room_emb_size)
        sent, token_probs = self.agentB.describe_action(ordered_actions_emb_B, choose_method)
        ordered_actions_emb_A = self.action_embedding_A(actions_info)
        if history_sents is not None:
            history_sents_room, history_sents_action = history_sents
            if history_sents_action is None:
                history_sents_action = sent.unsqueeze(1)
            else:
                history_sents_action = torch.cat([history_sents_action, sent.unsqueeze(1)], dim=1)
            sent = torch.cat([history_sents_room, history_sents_action], dim=2)
            assert sent.shape[-1] == Param.max_sent_len + Param.max_sent_len  # NOTE sent len = room sent len + gate sent len
        gate_idx, gate_prob = self.agentA.guess_gate(ordered_actions_emb_A, sent, choose_method)
        return gate_idx, token_probs, gate_prob, sent if history_sents is None else history_sents_action

    def backward(self, token_probs, gate_prob, reward):
        lossB = self.agentB.cal_guess_action_loss(token_probs, reward)
        lossA = self.agentA.cal_guess_action_loss(gate_prob, reward)
        lossB.backward()
        lossA.backward()
        return lossB, lossA
