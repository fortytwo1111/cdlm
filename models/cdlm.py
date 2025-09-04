
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import DefaultDict
from torch.nn import Parameter, Linear


class CDLMClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        in_dim = opt.hidden_dim
        self.opt = opt
        self.gcn_model = GCNAbsaModel(embedding_matrix=embedding_matrix, opt=opt)
        self.classifier = nn.Linear(in_dim, opt.polarities_dim)  # 线性前向传播
        print('a:', embedding_matrix.shape)

    def forward(self, inputs):
        outputs = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, None


class GCNAbsaModel(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt
        self.embedding_matrix = embedding_matrix

        # print('ab:',embedding_matrix.shape)

        self.emb = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0) if opt.pos_dim > 0 else None  # POS emb
        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim,
                                     padding_idx=0) if opt.post_dim > 0 else None  # position emb
        embeddings = (self.emb, self.pos_emb, self.post_emb)
        # print('embeddings:', embeddings)

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt.hidden_dim, opt.num_layers)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, short_mask = inputs
        # print('l.data:',l.data)
        maxlen = max(l.data)
        # print(maxlen)
        mask = mask[:, :maxlen]
        # print('mask:', mask.shape)
        h = self.gcn(inputs)

        asp_wn = mask.sum(dim=1).unsqueeze(-1)
        p = len(mask)
        b = len(mask[0])
        mask = mask.unsqueeze(-1).repeat(1, 1, self.opt.hidden_dim)
        outputs = (h * mask).sum(dim=1) / asp_wn

        return outputs


class GCN(nn.Module):
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()  # 继承和初始化模型，确保nn.Module类的构造函数被执行
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = opt.embed_dim + opt.post_dim + opt.pos_dim  # 300+30+30
        self.emb, self.pos_emb, self.post_emb = embeddings

        # dpcnn部分
        input_channels = opt.embed_dim + opt.pos_dim + opt.post_dim
        self.dpcnn = DPCNN(input_channels, opt.num_feature_maps, opt.num_conv_blocks)

        input_size = self.in_dim
        self.rnn = nn.LSTM(input_size, opt.rnn_hidden, opt.rnn_layers, batch_first=True, \
                           dropout=opt.rnn_dropout, bidirectional=opt.bidirect)
        if opt.bidirect:
            self.in_dim = opt.rnn_hidden * 2
        else:
            self.in_dim = opt.rnn_hidden
        # dropout
        self.rnn_drop = nn.Dropout(opt.rnn_dropout)
        self.in_drop = nn.Dropout(opt.input_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        self.W = nn.Linear(self.in_dim, self.in_dim)
        self.Wxx = nn.Linear(self.in_dim, self.mem_dim)
        self.aggregate_W = nn.Linear(self.in_dim * 3, self.mem_dim)

        self.attention_heads = opt.attention_heads
        self.head_dim = self.mem_dim // self.layers

        # 初始化MHA类
        self.attn = MultiHeadAttention(self.attention_heads, self.mem_dim * 2)
        self.weight_list = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.Wx = nn.Linear(self.attention_heads + self.mem_dim * 4, self.attention_heads)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        # Convert seq_lens to a 1D CPU int64 tensor
        seq_lens = seq_lens.cpu().long()
        h0, c0 = rnn_zero_state(batch_size, self.opt.rnn_hidden, self.opt.rnn_layers, self.opt.bidirect)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        # print('shape:', rnn_outputs.shape)
        # print(rnn_outputs)
        # 返回值：
        # 返回值=(batch_size,max_seq_len,self.opt.rnn_hidden*2)
        return rnn_outputs

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, short_mask = inputs
        # print(l)
        # print('tok_shape:', tok.shape)
        # print('short_mask',short_mask.shape)
        src_mask = (tok != 0).unsqueeze(-2)
        # print('src_mask', src_mask.shape)
        maxlen = max(l.data)
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]
        short_mask = short_mask[:, :, :maxlen, :maxlen]


        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.opt.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.opt.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)  # ([16, 85, 360])
        # print('embs.shape:',embs.shape)

        # 使用DPCNN
        # print('embshape',embs.shape)
        dpcnn_features = self.dpcnn(embs)
        dpcnn_features = dpcnn_features.permute(0, 2, 1)  # Reshape back if needed
        # print('dpcnn_features', dpcnn_features.shape)

        self.rnn.flatten_parameters()
        gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, l, tok.size()[0]))
        # emb大小 (batch_size, max_length, pos+post+embed_dim)

        # 假设 rnn_output 和 dpcnn_output 分别是 encode_with_rnn 和 DPCNN 的输出
        # 调整 dpcnn_output 的长度以匹配 rnn_output
        if dpcnn_features.size(2) != gcn_inputs.size(1):
            dpcnn_features = F.interpolate(dpcnn_features, size=gcn_inputs.size(1), mode='nearest')

        device = dpcnn_features.device
        # 如果通道数不匹配，可以使用线性层进行变换（假设 num_feature_maps 和 rnn_hidden * 2 不同）
        if dpcnn_features.size(1) != gcn_inputs.size(2):
            linear_transform = nn.Linear(dpcnn_features.size(1), gcn_inputs.size(2)).to(device)
            dpcnn_features = linear_transform(dpcnn_features.permute(0, 2, 1))  # .permute(0, 2, 1)

        # # 权重拼接，法1：直接拼
        # fused_features = torch.cat((gcn_inputs, dpcnn_features), dim=2)
        #
        # target_dim = gcn_inputs.size(2)  # rnn_hidden * 2
        # linear_transform_fuse = nn.Linear(fused_features.size(2), target_dim).to(device)
        # fused_features_resized = linear_transform_fuse(fused_features).to(device)

        # 法2
        alpha = torch.sigmoid(nn.Parameter(torch.randn(1))).to(device)
        beta = 1 - alpha
        # 加权融合
        fused_features = alpha * gcn_inputs + beta * dpcnn_features
        assert fused_features.size() == gcn_inputs.size()
        fused_features_resized = fused_features

        # 消融RNN
        # fused_features_resized = dpcnn_features

        # 消融DPCNN
        # fused_features_resized = gcn_inputs

        # print(fused_features_resized.shape)
        # 法3
        # fused_features = gcn_inputs + dpcnn_features
        #
        # # 使用线性层将融合后的特征降维到与 RNN 最初的输出大小一致
        # target_dim = gcn_inputs.size(2)  # rnn_hidden * 2
        # linear_transform_fuse = nn.Linear(fused_features.size(2), target_dim).to(device)
        # fused_features_resized = linear_transform_fuse(fused_features)
        #
        # assert fused_features_resized.size() == gcn_inputs.size()

        # 现在，fused_features_resized 的形状为 (batch_size, max_seq_len, target_dim)
        # print(fused_features_resized.shape)

        # print('input_shape:', gcn_inputs.shape)
        # print('gcn_inputs:', gcn_inputs)
        asp_wn = mask.sum(dim=1).unsqueeze(-1)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.opt.hidden_dim * 2)
        mask = mask[:, :maxlen, :]


        # 以下五个fused_features_resized原为gcn_inputs
        aspect_outs = (fused_features_resized * mask).sum(dim=1) / asp_wn
        # print('aspect_outs', aspect_outs.shape)
        short_mask = torch.zeros_like(short_mask)
        attn_tensor = self.attn(fused_features_resized, fused_features_resized, src_mask, short_mask, aspect_outs)
        # attn_tensor = self.attn(fused_features_resized, fused_features_resized, None, None, aspect_outs)
        # print('attention tensor: ', attn_tensor.shape)
        # print(attn_tensor)
        weight_adj = attn_tensor
        # print(weight_adj.shape)
        gcn_outputs = fused_features_resized
        layer_list = [fused_features_resized]

        if self.opt.gcn == 'gcn':
            for i in range(self.layers):
                # len(l) 长度等于batch_size
                # gcn_outputs:(len(l), self.attention_heads, maxlen, self.mem_dim * 2)
                gcn_outputs = gcn_outputs.unsqueeze(1).expand(len(l), self.attention_heads, maxlen, self.mem_dim * 2)
                Ax = torch.matmul(weight_adj, gcn_outputs)
                Ax = Ax.mean(dim=1)

                Ax = self.W(Ax)  # (batch_size,max_sql_length,mem_dim*2)
                weights_gcn_outputs = F.selu(Ax)  # (batch_size,max_sql_length,mem_dim*2)

                gcn_outputs = weights_gcn_outputs
                layer_list.append(gcn_outputs)
                gcn_outputs = self.gcn_drop(gcn_outputs) if i < self.layers - 1 else gcn_outputs

                weight_adj = weight_adj.permute(0, 2, 3, 1).contiguous()
                node_outputs1 = gcn_outputs.unsqueeze(1).expand(len(l), maxlen, maxlen, self.mem_dim * 2)
                node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()

                node = torch.cat([node_outputs1, node_outputs2], dim=-1)
                # print("weight_adj shape:", weight_adj.shape)
                # print("node shape:", node.shape)
                edge_n = torch.cat([weight_adj, node], dim=-1)
                edge = self.Wx(edge_n)
                edge = self.gcn_drop(edge) if i < self.layers - 1 else edge
                weight_adj = edge.permute(0, 3, 1, 2).contiguous()

                # outputs = torch.cat(layer_list, dim=-1)
        else:
            # gat
            for i in range(self.layers):
                # gcn_outputs 初始形状: (batch_size, max_seq_length, mem_dim * 2)
                # 扩展 gcn_outputs 以匹配 weight_adj 形状
                gcn_outputs_exp = gcn_outputs.unsqueeze(1).expand(-1, self.attention_heads, -1, -1)

                # 计算注意力系数
                # weight_adj 形状: (batch_size, attention_heads, max_seq_length, max_seq_length)
                attention_coefficients = F.softmax(weight_adj, dim=-1)

                # 应用注意力机制进行节点信息的聚合
                Ax = torch.einsum('bhij,bhjd->bhid', attention_coefficients, gcn_outputs_exp)

                # 聚合头后的降维
                Ax = Ax.mean(dim=1)
                # 线性变换
                Ax = self.W(Ax)
                gcn_outputs = F.selu(Ax)  # 使用 SELU 激活函数
                # 每层后的dropout（除了最后一层）
                gcn_outputs = self.gcn_drop(gcn_outputs) if i < self.layers - 1 else gcn_outputs

                # 更新 weight_adj 以用于下一轮注意力计算
                weight_adj = weight_adj.permute(0, 2, 3, 1).contiguous()
                # 首先基于当前节点输出计算新的特征组合
                node_outputs1 = gcn_outputs.unsqueeze(1).expand(-1, maxlen, -1, -1)
                node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()

                node = torch.cat([node_outputs1, node_outputs2], dim=-1)
                # print("weight_adj shape:", weight_adj.shape)
                # print("node shape:", node.shape)
                edge_n = torch.cat([weight_adj, node], dim=-1)
                edge = self.Wx(edge_n)
                edge = self.gcn_drop(edge) if i < self.layers - 1 else edge
                weight_adj = edge.permute(0, 3, 1, 2).contiguous()

        node_outputs = self.Wxx(gcn_outputs)
        # node_outputs=self.aggregate_W(outputs)
        node_outputs = F.relu(node_outputs)

        return node_outputs


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def attention(query, key, short, aspect, weight_m, bias_m, mask, dropout, ):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    batch = len(scores)
    p = weight_m.size(0)
    max = weight_m.size(1)
    # att 消融实验
    # weight_m = weight_m.unsqueeze(0).repeat(batch,1,1,1)
    weight_m = weight_m.unsqueeze(0).expand(batch, p, max, max)

    aspect_scores = torch.tanh(
        torch.add(torch.matmul(torch.matmul(aspect, weight_m), key.transpose(-2, -1)), bias_m))  # [16,5,41,41]
    scores = torch.add(scores, aspect_scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # if short is None:
    #    short = torch.zeros_like(scores).to(scores.device)
    scores = torch.add(scores, short).cuda()
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # 确保d_model能被整除，否则抛出异常
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.weight_m = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k))
        self.bias_m = nn.Parameter(torch.Tensor(1))
        self.dense = nn.Linear(d_model, self.d_k)

    def forward(self, query, key, mask, short, aspect):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        batch, aspect_dim = aspect.size()[0], aspect.size()[1]
        aspect = aspect.unsqueeze(1).expand(batch, self.h, aspect_dim)
        aspect = self.dense(aspect)
        aspect = aspect.unsqueeze(2).expand(batch, self.h, query.size()[2], self.d_k)
        attn = attention(query, key, short, aspect, self.weight_m, self.bias_m, mask, self.dropout)
        return attn


class DPCNN(nn.Module):
    def __init__(self, input_channels, num_feature_maps, num_conv_blocks):
        super(DPCNN, self).__init__()
        self.num_conv_blocks = num_conv_blocks
        self.region_embedding = nn.Conv1d(input_channels, num_feature_maps, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.conv_block = nn.Sequential(
            nn.Conv1d(num_feature_maps, num_feature_maps, kernel_size=3, padding=1),
            # nn.BatchNorm1d(num_feature_maps),
            nn.ReLU(),
            nn.Conv1d(num_feature_maps, num_feature_maps, kernel_size=3, padding=1),
            # nn.BatchNorm1d(num_feature_maps),
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)


    def forward(self, x):
        num_conv_blocks = self.num_conv_blocks
        x = x.permute(0, 2, 1)  # 调整维度为 (batch_size, embedding_dim, max_length)
        x = self.relu(self.region_embedding(x))
        x = self.max_pool(x)

        # 多个重复的卷积块与池化层
        residual = x
        for _ in range(num_conv_blocks):  # 调整迭代次数以适应具体需求
            y = self.conv_block(x)
            y = self.max_pool(y)

            # 调整残差尺寸以匹配 y
            residual_resized = F.interpolate(residual, size=y.size(2), mode='nearest')
            y = y + residual_resized
            residual = y

        return y
