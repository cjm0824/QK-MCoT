'''
Adapted from https://github.com/huggingface/transformers and https://github.com/j-min/VL-T5
'''

from torch_geometric.typing import torch_scatter
from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, __HEAD_MASK_WARNING_MSG, T5Block, T5LayerNorm
import copy
from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import math
import os
import warnings
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, RGCNConv

class GraphEncoder(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=768, edge_type_count=34, edge_emb_dim=64):
        super(GraphEncoder, self).__init__()
        self.edge_emb = nn.Embedding(edge_type_count, edge_emb_dim)

        # 第一层：Relational GAT（简化可用 RGCN 或 GAT + edge embedding 模拟）
        self.rgat = RGCNConv(in_channels=in_dim, out_channels=hidden_dim, num_relations=edge_type_count)

        # 第二层：GCN（标准图卷积）
        self.gcn = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_type):
        """
        参数:
        - x: 节点初始表示 (num_nodes, in_dim)
        - edge_index: 边的连接信息，维度为 (2, num_edges)
        - edge_type: 每条边的类型，维度为 (num_edges,)
        """

        # Relational GAT层
        h1 = self.rgat(x, edge_index, edge_type)

        # GCN层（标准卷积）
        h2 = self.gcn(h1, edge_index)

        return h2  # 返回最终的节点表示 Hkg

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim // 2)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(input_dim // 2, output_dim)
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

'''
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, temperature=0.1, top_k=100, max_triples=500):
        super(SiameseNetwork, self).__init__()
        self.graph_encoder = GraphEncoder(embedding_dim, embedding_dim)
        self.mlp = MLP(embedding_dim, embedding_dim)
        self.triple_mlp = MLP(embedding_dim, embedding_dim)  # 新增MLP用于三元组
        self.top_k = top_k
        self.max_triples = max_triples
        self.temperature = nn.Parameter(torch.ones(1))  # 可学习参数

    def forward(self, question_embed, triples_embed):
        B, num_tri, len_tri, D = triples_embed.size()
        if num_tri > self.max_triples:
            triples_embed = triples_embed[:, :self.max_triples, :, :]
            num_tri = self.max_triples
        # 图编码
        triples_encoded = self.graph_encoder(triples_embed)  # [B, num_tri, D]
        triples_encoded = self.triple_mlp(triples_encoded)   # [B, num_tri, D]，新增MLP
        question = self.mlp(question_embed)  # [B, len_q, D]
        # 计算相似性（矩阵乘法）
        # question.mean(dim=1): [B, D] -> [B, 1, D]
        # question_vec = question.mean(dim=1, keepdim=True)    # [B, 1, D]
        similarity = torch.bmm(triples_encoded, question.transpose(1, 2)) # [B, num_tri, len_q]
        similarity = torch.mean(similarity, dim=-1, keepdim=True)   # [B, num_tri, 1]
        similarity = similarity / self.temperature
        similarity = torch.sigmoid(similarity)
        # 稀疏化
        if self.top_k < num_tri:
            topk_values, topk_indices = torch.topk(similarity, self.top_k, dim=1)
            mask = torch.zeros_like(similarity) # [B, num_tri, 1]
            mask.scatter_(1, topk_indices, 1)   # [B, num_tri, 1]
            similarity = similarity * mask  # [B, num_tri, 1]
        weighted_triples = similarity * triples_encoded  # [B, num_tri, D]
        return weighted_triples
'''
'''
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, temperature=0.1):
        super(SiameseNetwork, self).__init__()
        self.mlp = MLP(embedding_dim, embedding_dim)
    def forward(self, question_embed, triples_embed):
        _, len_q, D = question_embed.size()
        B, num_tri, len_tri, D  = triples_embed.size()

        question = self.mlp(question_embed)
        triples_flat = triples_embed.view(B*num_tri, len_tri, D)
        triples = self.mlp(triples_flat)
        question_expanded = question.unsqueeze(1).expand(-1, num_tri, -1, -1)
        question_flat = question_expanded.contiguous().view(B*num_tri, len_q, D)

        similarity = torch.bmm(triples, question_flat.transpose(1, 2))
        similarity = torch.mean(similarity, dim=-1, keepdim=True)
        similarity = similarity / self.temperature
        similarity = torch.sigmoid(similarity)

        weighted_triples= similarity * triples
        weighted_triples  = weighted_triples.view(B, num_tri, len_tri, D)
        weighted_triples = weighted_triples.mean(dim=2)
        return weighted_triples
'''
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, temperature=0.1):
        super(SiameseNetwork, self).__init__()
        self.mlp = MLP(embedding_dim, embedding_dim)
        # self.node_mlp  = MLP(embedding_dim, embedding_dim)
        self.temperature = temperature
    def forward(self, question_embed, nodes_embed):
        _, D = question_embed.size()
        B, num_nodes, D  = nodes_embed.size()

        question = self.mlp(question_embed) #[B, D]
        question_expanded = question.unsqueeze(1).expand(-1, num_nodes, -1) #[B, num_nodes, D]
        nodes = self.mlp(nodes_embed)   # [B, num_nodes, D]
        nodes_flat = nodes.view(B*num_nodes, D) # [B*num_nodes, D]
        question_flat = question_expanded.contiguous().view(B*num_nodes, D) # [B*num_nodes, D]

        # similarity = torch.bmm(nodes_flat, question_flat.transpose(1, 2))
        # similarity = torch.mean(similarity, dim=-1, keepdim=True)
        similarity = F.cosine_similarity(question_flat, nodes_flat, dim=-1)
        similarity = similarity / self.temperature
        similarity = torch.sigmoid(similarity).unsqueeze(-1)

        weighted_nodes = similarity * nodes_flat
        weighted_nodes = weighted_nodes.view(B, num_nodes, D)
        return weighted_nodes
# JointEncoder：多模态编码器，继承自T5Stack，能够同时处理文本和图像输入
class JointEncoder(T5Stack):
    def __init__(self, config, embed_tokens=None, patch_size=None, num_queries=64):
        super().__init__(config)
        #初始化
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.patch_num, self.patch_dim = patch_size        
        self.image_dense = nn.Linear(self.patch_dim, config.d_model)    #将图像特征映射到文本特征空间
               
        # 改进的查询生成器 - 基于注意力的方法
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, config.hidden_size))
        self.query_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size, 
            num_heads=8, 
            batch_first=True
        )
        self.query_norm = nn.LayerNorm(config.hidden_size)
        self.image_query_cross_attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=8, batch_first=True)
        
        # 添加残差连接的层归一化
        self.img_query_norm = nn.LayerNorm(config.hidden_size)

        self.num_queries = num_queries
        self.hidden_dim = config.hidden_size
        
        #多头注意力机制，用于计算文本特征（查询Q）对图像特征（键K和值V）的注意力权重
        self.direct_mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size, vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.queried_mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size, vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )

        self.node_question_mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size, vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.fusion_layer = nn.Linear(3*config.hidden_size, 3)
        self.gate_dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.siamese_network = SiameseNetwork(config.hidden_size)

        self.Hkg_embed = GraphEncoder()

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
    
    #parallelize和deparallelize用于支持模型并行处理，parallelize将模型分布到多个GPU上，deparallelize将模型移回CPU
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        image_ids=None,
        subgs=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):  #处理输入并生成多模态表示
        # Model parallel,模型并行处理，将模型分布到多个GPU上
        if self.model_parallel: # self.model_parallel=False
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)

        # 设置各种控制参数的默认值
        use_cache = use_cache if use_cache is not None else self.config.use_cache   # False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions   # False
        output_hidden_states = (    # False
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict   # True

        # 验证输入参数，提供输入IDs或嵌入向量，不能同时提供，然后获取输入的形状信息
        if input_ids is not None and inputs_embeds is not None: #input_ids is not None but inputs_embeds is  None
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()  #torch.Size([2, 512]) [batch_size, input_len]
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        # 如果没有直接提供嵌入向量，则使用嵌入层将输入的token_ids转换为嵌入向量
        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)    # torch.Size([2, 512, 1024])

        # 获取批次大小和序列长度
        batch_size, seq_length = input_shape    #batch_size=2, seq_length=512

        #计算掩码序列的长度，如果存在past_key_values，则使用past_key_values的长度加当前序列长度，否则使用当前序列长度
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length    #512

        # 验证缓存使用的合法性，只有解码器模型才能使用缓存
        if use_cache is True:   #use_cache=False
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"
        
        # 注意力掩码处理
        if attention_mask is None:  #torch.Size([2, 512])
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)  # len(self.block)=24

        #将注意力掩码扩展为适合多头注意力机制的形状
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:   # self.is_decoder=False, encoder_hidden_states = None
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)   # head_mask=None, self.config.num_layers=24 [None]*24
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers) #[None]*24
        present_key_value_states = () if use_cache else None    # present_key_value_states=None
        all_hidden_states = () if output_hidden_states else None    #None
        all_attentions = () if output_attentions else None  # None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None  # None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds) #dropout操作，防止过拟合并生成隐藏状态，torch.Size([2, 512, 1024])
        
        # T5编码层处理
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel: # False
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:    # False
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:#self.gradient_checkpointing=False, self.training=True
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]# torch.Size([2, 512, 1024])，None

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]    # torch.Size([2, 16, 512, 512])
            if self.is_decoder and encoder_hidden_states is not None:   # self.is_decoder=False, encoder_hidden_states=None
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:   # False
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states) # torch.Size([B, 512, 768])

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # 获取输入问题文本的句子级表示
        if attention_mask is not None:
            # Expand attention_mask to match hidden_states dimensions for broadcasting
            # attention_mask_expanded: [batch_size, seq_length, 1]
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            # Sum embeddings where attention_mask is 1
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
            # Count valid tokens for each sentence in the batch
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)  # Avoid division by zero
            question_embeddings = sum_embeddings / sum_mask # torch.Size([B, 768])
        else:
            # If no attention_mask, assume all tokens are valid (less ideal)
            question_embeddings = torch.mean(hidden_states, dim=1)
        

        # 对子图进行编码
        subgs = subgs.to(self.device)

        edge_index = subgs.edge_index
        edge_type = subgs.edge_type

        node_tokens = subgs.node_tokens
        node_input_ids = node_tokens["input_ids"]
        node_attention_mask = node_tokens["attention_mask"]
        node_embeds = self.embed_tokens(node_input_ids)
        masked_embeds = node_embeds * node_attention_mask.unsqueeze(-1)  # [num_nodes, seq_len, D]
        sum_embeds = masked_embeds.sum(dim=1)
        lengths = node_attention_mask.sum(dim=1).clamp(min=1)
        node_representations = sum_embeds / lengths.unsqueeze(-1) 

        Hkg_embeds = self.Hkg_embed(node_representations, edge_index, edge_type)  # [sum(num_nodes), D]
        ptr = subgs.ptr  # [B+1]
        node_embeds_per_sample = [Hkg_embeds[ptr[i]:ptr[i+1]] for i in range(ptr.numel()-1)]  # 列表，每个元素形状[n_i, D]
        # 若需补齐为 [B, n_max, D] 可用 pad_sequence:

        node_embeds_padded = pad_sequence(node_embeds_per_sample, batch_first=True)  # [B, n_max, D]

        '''
        if triple_ids is not None:  # 输入的是邻域三元组
            # 处理四维的三元组张量 [batch_size, num_tri, 3, triple_len]
            batch_size, num_tri, num_elements, len_tri = triple_ids.size()
            
            # 将四维张量重塑为三维，便于处理
            triple_ids_reshaped = triple_ids.view(batch_size * num_tri * num_elements, len_tri)
            triple_embeds_flat = self.embed_tokens(triple_ids_reshaped) # [batch_size * num_tri * num_elements, len_tri, D]

            # 重塑回原始维度结构
            triple_embeds = triple_embeds_flat.view(batch_size, num_tri, num_elements, len_tri, -1) #[batch_size, num_tri, num_elements, len_tri, D]
            
        if triple_attention_masks is not None:
            # 处理四维的三元组掩码 [batch_size, num_tri, 3, triple_len]
            triple_attention_masks_reshaped = triple_attention_masks.view(batch_size * num_tri * num_elements, len_tri)
            
            # 扩展掩码维度以匹配嵌入
            triple_mask_expanded = triple_attention_masks_reshaped.unsqueeze(-1).expand_as(triple_embeds_flat)
            
            # 应用掩码
            masked_embeds = triple_embeds_flat * triple_mask_expanded
            
            # 计算平均值（沿序列长度维度）
            # 首先计算每个位置的掩码和
            mask_sum = triple_attention_masks_reshaped.sum(dim=1, keepdim=True)
            # 防止除零
            mask_sum = torch.clamp(mask_sum, min=1.0)
            # 计算平均嵌入
            avg_embeds = masked_embeds.sum(dim=1) / mask_sum
            
            # 重塑回原始维度结构，但每个元素现在只有一个向量表示
            triple_embeddings = avg_embeds.view(batch_size, num_tri, num_elements, -1)
        else:
            # 如果没有掩码，简单地取平均值
            triple_embeddings = triple_embeds.mean(dim=3)  # 在序列长度维度上取平均
        '''
        '''
        if triple_ids is not None:  # 输入的是路径节点三元组
            batch_size, num_tri, len_tri = triple_ids.size()
            triple_ids_flat = triple_ids.view(-1, len_tri)
            triple_embeds_flat = self.embed_tokens(triple_ids_flat)
            triple_embeds = triple_embeds_flat.view(batch_size, num_tri, len_tri, -1)
            
        if triple_attention_masks is not None:
            triple_attention_masks_flat = triple_attention_masks.view(-1, len_tri)
            triple_mask_expanded = triple_attention_masks_flat.unsqueeze(-1).expand_as(triple_embeds_flat)
            # 应用掩码
            masked_embeds = triple_embeds_flat * triple_mask_expanded
            triple_embeddings = masked_embeds.view(batch_size, num_tri, len_tri, -1)
        else:
            # 如果没有掩码，直接平均池化
            triple_embeddings = triple_embeds  # [batch_size*num_tri, hidden_size]
        '''
        # weight_triple_embeddings = self.siamese_network(hidden_states, triple_embeddings)   # [B, num_tri, hidden_size]
        #
        # triple_att, _ = self.triple_question_mha_layer(hidden_states, weight_triple_embeddings, weight_triple_embeddings)
        weight_node_embeddings = self.siamese_network(question_embeddings, node_embeds_padded)  # [B, n_max, 768]
        weight_att, _ = self.node_question_mha_layer(hidden_states, weight_node_embeddings, weight_node_embeddings) #[B, 512, 768]

        #将图像特征映射到文本特征空间
        image_embedding = self.image_dense(image_ids) # torch.Size([2, 145, 1024])-->torch.Size([2, 145, 768])，用于将图像特征映射到文本特征空间
        
        #多头注意力机制，计算文本和图像之间的注意力，输出形状与hidden_states相同
        direct_att, _ = self.direct_mha_layer(hidden_states, image_embedding, image_embedding)  #[B, 512, 768]
        

        '''
            BLIP-2 + Querying方法：动态生成查询向量
            根据文本特征生成任务特定的查询向量，而不是直接使用hidden_states
        '''

        # 构造Query向量
        B = image_embedding.shape[0]
        
        # 扩展查询嵌入到批次维度
        query_embeds = self.query_embeddings.unsqueeze(0).expand(B, -1, -1)  # [num_queries, hidden_size]->[B, num_queries, hidden_size]
        
        # 使用问题特征引导查询生成
        guided_queries, _ = self.query_attention(query=query_embeds,
                                                 key=question_embeddings.unsqueeze(1),
                                                 value=question_embeddings.unsqueeze(1)
                                                 )  # [B, num_queries, hidden_size]
        guided_queries = self.query_norm(guided_queries + query_embeds)  # 残差连接 [B, num_queries, hidden_sie]

        # 图像-查询交叉注意力
        img_attended_queries, _ = self.image_query_cross_attention(query=guided_queries,
                                                                   key=image_embedding,
                                                                   value=image_embedding
                                                                   ) # [B, num_queries, hidden_size]
        img_attended_queries = self.img_query_norm(img_attended_queries + guided_queries)  # 残差连接 [B, num_queries, hidden_size]

        queried_att, _ = self.queried_mha_layer(hidden_states, img_attended_queries, img_attended_queries)  #[B, seq_len, hidden_size]
        
        # 自适应融合
        fusion_input = torch.cat([direct_att, queried_att], dim=-1) # [B, seq_len, 2*hidden_size]
        fusion_weight = self.fusion_gate(fusion_input)  # [B, seq_len, 1]
        image_att = fusion_weight * direct_att + (1 - fusion_weight) * queried_att  # [B, seq_len, hidden_size]
        
        '''
        #将文本表示和图像注意力结果拼接起来
        merge = torch.cat([hidden_states, image_att], dim=-1)   # torch.Size([B, seq_len, 2*hidden_size])

        #通过门控机制动态融合文本和图像信息，门控融合机制可以动态调整文本特征和视觉特征的贡献度
        gate = self.sigmoid(self.gate_dense(merge)) # 计算门控权重， g= sigmoid(W_g * (h, m)+b_g),    torch.Size([B, seq_len, hidden_size])
        hidden_states = (1 - gate) * hidden_states + gate * image_att   # 门控权重作用于输入特征，得到融合后的特征 H =g*h+(1-g)*m torch.Size([B, seq_len, hidden_size])
        '''


        concat_features = torch.cat([hidden_states, image_att, weight_att], dim=-1) #  torch.Size([2, 512, 3*768])
        fusion_score = self.fusion_layer(concat_features)   #  torch.Size([2, 512, 3])
        fusion_weight = torch.softmax(fusion_score, dim=-1) #  torch.Size([2, 512, 3])
        alpha = fusion_weight[:, :, 0].unsqueeze(-1)    #  torch.Size([2, 512, 1])
        beta = fusion_weight[:, :, 1].unsqueeze(-1)
        gamma = fusion_weight[:, :, 2].unsqueeze(-1)
        hidden_states = alpha * hidden_states + beta * image_att + gamma * weight_att   #[B, 512, 768]


        if not return_dict: #return_dict=True
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        #返回融合后的多模态表示
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

#多模态生成模型，使用JointEncoder作为编码器处理多模态输入，T5解码器生成输出文本
class T5ForMultimodalGeneration(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, patch_size):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)

        #将标准的编码器替换为多模态编码器JointEncoder
        self.encoder = JointEncoder(encoder_config, self.shared, patch_size)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False #模型并行设置为False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_ids=None,
        attention_mask: Optional[torch.FloatTensor] = None,
        subgs = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None: # head_mask=None, decoder_head_mask=None
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask



        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None: #encoder_outputs=None，使用多模态编码器处理文本和图像输入
            # Convert encoder inputs in embeddings if needed
            # encoder_outputs = self.encoder( # 返回的是融合后的多模态表示
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     inputs_embeds=inputs_embeds,
            #     image_ids=image_ids,
            #     head_mask=head_mask,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            # )
            encoder_outputs = self.encoder( # 返回的是融合后的多模态表示
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                image_ids=image_ids,
                subgs=subgs,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]  # torch.Size([2, 512, 1024])

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode，使用T5解码器生成文本输出
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]    # torch.Size([2, 512, 1024])

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
        
        #将解码器输出映射到词汇表上的概率分布
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        #计算交叉熵损失
        lm_logits = self.lm_head(sequence_output)   #torch.Size([2, 512, 32128])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        #返回完整的序列到序列输出，包括损失、预测和各种中间状态
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    #确保在生成过程中正确处理图像特征
    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
    # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        output = {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

        if "image_ids" in kwargs:
            output["image_ids"] = kwargs['image_ids']

        return output

    #评估模型的性能，生成预测文本并与目标进行比较
    def test_step(self, tokenizer, batch, **kwargs):
        device = next(self.parameters()).device
        #获取输入数据
        input_ids = batch['input_ids'].to(device)
        image_ids = batch['image_ids'].to(device)
        #生成输出文本
        output = self.generate(
            input_ids=input_ids,
            image_ids=image_ids,
            **kwargs
        )
        #解码生成的文本
        generated_sents = tokenizer.batch_decode(output, skip_special_tokens=True)
        targets = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        result = {}
        result['preds'] = generated_sents
        result['targets'] = targets

        return result
