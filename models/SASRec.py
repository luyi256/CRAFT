import torch
from torch import nn
import torch.nn.functional as fn
import copy
import math

class SASRec(torch.nn.Module):

    def __init__(self, n_layers, n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, initializer_range, n_items, max_seq_length, device):
        super(SASRec, self).__init__()

        # load parameters info
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size  # same as embedding_size
        self.inner_size = inner_size  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.n_items = n_items
        self.max_seq_length = max_seq_length
        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items+1, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoderbyHand(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # parameters initialization
        self.apply(self._init_weights)
        self.device = device

    def set_min_idx(self, src_min_idx, dst_min_idx):
        self.src_min_idx = src_min_idx
        self.dst_min_idx = dst_min_idx

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_seq_len[item_seq_len == 0] = 1
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask
    
    def calculate_loss(self, batch_data):
        item_seq = batch_data[0].to(self.device) - self.dst_min_idx + 1
        item_seq[item_seq < 0] = 0
        item_seq_len = batch_data[1].to(self.device)
        seq_output = self.forward(item_seq, item_seq_len)
        test_item = batch_data[2].to(self.device) - self.dst_min_idx + 1
        return seq_output, (self.item_embedding(test_item.long()))
        # if self.loss_type == "BPR":
        #     pass
        # else:  # self.loss_type = 'CE'
        #     test_item_emb = self.item_embedding.weight
        #     logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        #     loss = self.loss_fct(logits, pos_items)
        #     return loss

    def predict(self, interaction):
        item_seq = interaction[0].to(self.device) - self.dst_min_idx + 1
        item_seq[item_seq < 0] = 0
        item_seq_len = interaction[1].to(self.device)
        test_item = interaction[2].to(self.device) - self.dst_min_idx + 1
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = (seq_output.view(-1, 1, seq_output.shape[-1]) * test_item_emb).sum(dim=-1)
        return scores[:, 0], scores[:, 1:]

    def set_neighbor_sampler(self, neighbor_sampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

class MultiHeadAttentionbyHand(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        rotary_emb_type=None,  
        rotary_emb=None,
        post_ln=False,
    ):
        super(MultiHeadAttentionbyHand, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.rotary_emb_type = rotary_emb_type
        self.rotary_emb = rotary_emb
        self.post_ln = post_ln
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, query, attention_mask, key=None, interaction_time=None):
        # input_tensor: [batch_size, seq_len, hidden_size]
        if key is None:
            key = query
            query_time_slots = interaction_time
            key_time_slots = interaction_time
        else:
            if interaction_time is not None:
                query_time_slots = interaction_time[1]
                key_time_slots = interaction_time[0]
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(key)

        # query_layer: [batch_size, seq_len, num_heads, head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.rotary_emb is not None:
            if self.rotary_emb_type == 'time':
                query_layer = self.rotary_emb(query_layer, input_pos=query_time_slots)
                key_layer = self.rotary_emb(key_layer, input_pos=key_time_slots)
            else:
                query_layer = self.rotary_emb(query_layer)
                key_layer = self.rotary_emb(key_layer)

        # 重新排列以便矩阵乘法
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 3, 1)
        value_layer = value_layer.permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        if self.post_ln: 
            hidden_states = self.LayerNorm(hidden_states+query)
        else:
            hidden_states = self.LayerNorm(hidden_states)
            hidden_states = hidden_states + query

        return hidden_states

class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps, output_dim=None, post_ln=False
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        if output_dim is not None:
            self.output_dim = output_dim
            self.dense_2 = nn.Linear(inner_size, output_dim)
        else:
            self.output_dim = -1
            self.dense_2 = nn.Linear(inner_size, hidden_size)
            self.dropout = nn.Dropout(hidden_dropout_prob)
        self.post_ln = post_ln
    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        if self.output_dim==-1:
            hidden_states = self.dropout(hidden_states)
            if self.post_ln: 
                hidden_states = self.LayerNorm(hidden_states+input_tensor)
            else:
                hidden_states = self.LayerNorm(hidden_states)
                hidden_states = hidden_states + input_tensor

        return hidden_states

class TransformerLayerbyHand(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        rotary_emb_type=None,
        rotary_emb=None,
        output_dim=None,
        post_ln=False,
    ):
        super(TransformerLayerbyHand, self).__init__()
        self.multi_head_attention = MultiHeadAttentionbyHand(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, rotary_emb_type, rotary_emb, post_ln
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
            output_dim=output_dim,
            post_ln=post_ln
        )

    def forward(self, query, attention_mask, key=None, interaction_time=None):
        attention_output = self.multi_head_attention(query, attention_mask, key, interaction_time)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoderbyHand(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
        rotary_emb(nn.Module, optional): use RoPE. Default: None

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        rotary_emb=None,
        rotary_emb_type=None,
        output_dim=None,
        post_ln=False
    ):
        super(TransformerEncoderbyHand, self).__init__()
        layer = TransformerLayerbyHand(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
            rotary_emb_type=rotary_emb_type,
            rotary_emb=rotary_emb,
            output_dim=output_dim,
            post_ln=post_ln
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, query, attention_mask, key=None, output_all_encoded_layers=True, interaction_time=None):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(query, attention_mask, key, interaction_time)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers