import torch
from torch import nn
from models.modules import CrossAttention
from models.modules import BPRLoss, MLP

class CRAFT(torch.nn.Module):

    def __init__(self, n_layers, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, initializer_range, n_nodes, max_seq_length, device, loss_type, use_pos=True, input_cat_time_intervals=False, output_cat_time_intervals=True, output_cat_repeat_times=False, num_output_layer=1,  emb_dropout_prob=0.1, skip_connection=False):
        super(CRAFT, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size 
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.n_nodes = n_nodes
        self.max_seq_length = max_seq_length
        self.output_cat_time_intervals = output_cat_time_intervals
        self.output_cat_repeat_times = output_cat_repeat_times
        self.emb_dropout_prob = emb_dropout_prob
        self.node_embedding = nn.Embedding(
            self.n_nodes+1, self.hidden_size, padding_idx=0
        )
        self.use_pos = use_pos
        self.input_cat_time_intervals = input_cat_time_intervals
        if use_pos:
            self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        output_dim = 0 
        if self.input_cat_time_intervals:
            trm_input_dim = self.hidden_size * 2
        else:
            trm_input_dim = self.hidden_size
        self.cross_attention = CrossAttention(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=trm_input_dim,
            inner_size=trm_input_dim*4,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        output_dim += trm_input_dim
        if self.output_cat_time_intervals or self.input_cat_time_intervals:
            self.time_projection = MLP(num_layers=1, input_dim=1, hidden_dim=self.hidden_size, output_dim=self.hidden_size, dropout=self.hidden_dropout_prob, use_act=True, skip_connection=skip_connection)
        if self.output_cat_repeat_times:
            self.repeat_times_projection = MLP(num_layers=1, input_dim=1, hidden_dim=self.hidden_size, output_dim=self.hidden_size, dropout=self.hidden_dropout_prob, use_act=True, skip_connection=skip_connection)
        if self.output_cat_time_intervals:
            output_dim += self.hidden_size
        if self.output_cat_repeat_times:
            output_dim += self.hidden_size
        self.output_layer = MLP(num_layers=num_output_layer, input_dim=output_dim, hidden_dim=output_dim, output_dim=1, dropout=self.hidden_dropout_prob, use_act=True, skip_connection=skip_connection)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_time_intervals = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_repeat_times = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.emb_dropout = nn.Dropout(self.emb_dropout_prob)
        self.loss_type = loss_type
        if self.loss_type == "BCE":
            self.loss_fct = nn.BCELoss()
        elif self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)
        self.device = device
        
    def set_min_idx(self, src_min_idx, dst_min_idx):
        self.src_min_idx = src_min_idx
        self.dst_min_idx = dst_min_idx

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            # nn.init.kaiming_normal_(module.weight)
        # stdv = 1.0 / np.sqrt(self.hidden_size)
        # for weight in self.parameters():
        #     weight.data.uniform_(-stdv, stdv)

    def forward(self, src_neighb_seq, src_neighb_seq_len, neighbors_interact_times, cur_times, test_dst = None, dst_last_update_times = None):
        bs = src_neighb_seq.shape[0]
        src_neighb_seq_len[src_neighb_seq_len == 0] = 1
        neighb_emb = self.node_embedding(src_neighb_seq)
        if self.output_cat_time_intervals: # for all datasets
            dst_last_update_intervals = cur_times.view(-1,1) - dst_last_update_times
            dst_last_update_intervals[dst_last_update_times<-1]=-100000 
            dst_last_update_intervals = dst_last_update_intervals.to(self.device)
            dst_node_time_intervals_feat = self.time_projection(dst_last_update_intervals.float().view(-1, 1)).view(dst_last_update_intervals.shape[0], dst_last_update_intervals.shape[1], -1)
            dst_node_time_intervals_feat = self.LayerNorm_time_intervals(dst_node_time_intervals_feat)
            dst_node_time_intervals_feat = self.dropout(dst_node_time_intervals_feat)
        test_dst_emb = self.node_embedding(test_dst)
        test_dst_emb = self.LayerNorm(test_dst_emb.view(bs, -1, self.hidden_size))
        test_dst_emb = self.emb_dropout(test_dst_emb)
        if self.output_cat_repeat_times: # only for seen-dominant datasets
            repeat_times = test_dst.view(bs, test_dst.shape[1], 1) == src_neighb_seq.view(bs, 1, src_neighb_seq.shape[1])
            repeat_times = repeat_times.sum(dim=-1).unsqueeze(-1).float()
            repeat_times_feat = self.repeat_times_projection(repeat_times.float()).view(bs, -1, self.hidden_size)
            repeat_times_feat = self.LayerNorm_repeat_times(repeat_times_feat)
            repeat_times_feat = self.dropout(repeat_times_feat)
        if self.use_pos: # default
            position_ids = torch.arange(
                src_neighb_seq.size(1), dtype=torch.long, device=src_neighb_seq.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(src_neighb_seq)
            position_embedding = self.position_embedding(position_ids)
            input_emb = neighb_emb + position_embedding
        else:
            input_emb = neighb_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.emb_dropout(input_emb)
        if self.input_cat_time_intervals:# comparison experiments: using time encoding instead of position encoding
            src_neighbor_interact_time_intervals = cur_times.view(-1,1) - neighbors_interact_times
            src_neighbor_interact_time_intervals[src_neighb_seq==0]=-100000 # the neighbor is null
            src_neighb_time_embedding = self.time_projection(src_neighbor_interact_time_intervals.to(self.device).float().view(-1,1)).view(src_neighb_seq.shape[0], src_neighb_seq.shape[1], -1)
            src_neighb_time_embedding = self.LayerNorm_time_intervals(src_neighb_time_embedding)
            src_neighb_time_embedding = self.dropout(src_neighb_time_embedding)
            input_emb = torch.cat([input_emb, src_neighb_time_embedding], dim=-1)
        
        attention_mask = src_neighb_seq != 0
        test_dst_mask = torch.ones(test_dst_emb.shape[0], test_dst_emb.shape[1]).to(self.device)
        extended_attention_mask = self.get_attention_mask(test_dst_mask, mask_b=attention_mask)
        output = self.cross_attention(
            test_dst_emb, extended_attention_mask, input_emb, output_all_encoded_layers=True
        )[-1]
        if self.output_cat_time_intervals:
            if output is None:
                output = dst_node_time_intervals_feat
            else:
                output = torch.cat([output, dst_node_time_intervals_feat], dim=-1).float()
        if self.output_cat_repeat_times:
            if output is None:
                output = repeat_times_feat
            else:
                output = torch.cat([output, repeat_times_feat], dim=-1).float()
        output = self.output_layer(output.view(-1,output.shape[-1])).view(output.shape[0], output.shape[1], -1)
        return output
    
    def get_attention_mask(self, mask_a, mask_b):
        extended_attention_mask = torch.bmm(mask_a.unsqueeze(1).transpose(1,2), mask_b.unsqueeze(1).float()).bool().unsqueeze(1)
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask
    
    def predict(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst, dst_last_update_times):
        """
        [0]src_neighb_seq: [B, L]
        [1]src_neighb_seq_len: [B]
        [2]test_dst: [B, 1+num_negs], num_negs=0,1,...
        [3]src_neighb_interact_times: [B, L]
        [4]cur_pred_times: [B]
        [5]dst_last_update_times: [B, 1+num_negs]
        [6]src_neighb_last_update_times: [B, L]
        [7]src_slot_encoding: [B, W], W is the time slot window size
        [8]dst_slot_encoding: [B, W]
        """
        src_neighb_seq = src_neighb_seq.to(self.device) - self.dst_min_idx + 1
        test_dst = test_dst.to(self.device) - self.dst_min_idx + 1
        src_neighb_seq[src_neighb_seq < 0] = 0
        src_neighb_interact_times = src_neighb_interact_times.to(self.device)
        src_neighb_seq_len = src_neighb_seq_len.to(self.device)
        logits = self.forward(src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times.to(self.device), test_dst=test_dst, dst_last_update_times=dst_last_update_times.to(self.device))
        if self.loss_type == 'BPR':
            positive_probabilities = logits[:,0].flatten()
            negative_probabilities = logits[:,1:].flatten()
        else:
            positive_probabilities = logits[:,0].sigmoid().flatten()
            negative_probabilities = logits[:,1:].sigmoid().flatten()
        return positive_probabilities, negative_probabilities

    def calculate_loss(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst, dst_last_update_times):
        positive_probabilities, negative_probabilities = self.predict(src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst, dst_last_update_times)
        bs = test_dst.shape[0]
        if self.loss_type == 'BPR': 
            negative_probabilities = negative_probabilities.flatten()
            positive_probabilities = positive_probabilities.flatten()
            loss = self.loss_fct(positive_probabilities, negative_probabilities)
        predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
        labels = torch.cat([torch.ones(bs), torch.zeros(bs)], dim=0).to(self.device)
        if self.loss_type == 'BCE':
            loss = self.loss_fct(predicts, labels)
        elif self.loss_type != 'BPR':
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented! Only BCE and BPR are supported!")
        return loss, predicts, labels
