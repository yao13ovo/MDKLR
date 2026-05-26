import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from utils.utils_general import _cuda
import math
import copy


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class RNN_Residual(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0., batch_first=True):
        super(RNN_Residual, self).__init__()
        for i in range(n_layers):
            if i == 0:
                setattr(self, 'forward_rnn_{}'.format(i), nn.GRU(input_dim, hidden_dim))
                setattr(self, 'backward_rnn_{}'.format(i), nn.GRU(input_dim, hidden_dim))
            else:
                setattr(self, 'forward_rnn_{}'.format(i), nn.GRU(hidden_dim, hidden_dim))
                setattr(self, 'backward_rnn_{}'.format(i), nn.GRU(hidden_dim, hidden_dim))
        self.n_layers = n_layers
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first

    def run_rnn(self, rnn, embedded, input_lengths, batch_first=True):
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=batch_first)
        outputs, hidden = rnn(embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first)
        return outputs, hidden

    def flipByLength(self, input, lengths):
        output = _cuda(torch.zeros_like(input))
        for i, l in enumerate(lengths):
            output[i, :l, :] = torch.flip(input[i, :l, :], (0,))
        return output

    def forward(self, input, input_lengths):
        input_forward = input
        input_backward = self.flipByLength(input, input_lengths)
        for i in range(self.n_layers):
            output_forward, hidden_forward = self.run_rnn(self.__getattr__('forward_rnn_{}'.format(i)), input_forward,
                                                          input_lengths, self.batch_first)
            output_backward, hidden_backward = self.run_rnn(self.__getattr__('backward_rnn_{}'.format(i)),
                                                            input_backward, input_lengths, self.batch_first)
            if i == 0:
                input_forward = F.dropout(output_forward, self.dropout, self.training)
                input_backward = F.dropout(output_backward, self.dropout, self.training)
            else:
                input_forward = F.dropout(output_forward + input_forward, self.dropout, self.training)
                input_backward = F.dropout(output_backward + input_backward, self.dropout, self.training)
        output = torch.cat((input_forward, input_backward), dim=-1)
        hidden = torch.cat((hidden_forward, hidden_backward), dim=-1)
        return output, hidden


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CNNClassifier(nn.Module):
    def __init__(self, input_dim, output_channel, filter, output_dim, dropout):
        super(CNNClassifier, self).__init__()

        self.cnn = nn.ModuleList([nn.Conv2d(1, output_channel, (f, input_dim)) for f in filter])

        linear_dim = output_channel * len(filter)
        self.layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(linear_dim, output_dim, bias=False),
            nn.LeakyReLU(0.1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.contiguous().unsqueeze(1)
        conv = [F.relu(cnn_(input)).squeeze(3) for cnn_ in self.cnn]
        conv = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv]
        return self.layer(torch.cat(conv, 1))


class SelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context



class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.scorer = nn.Linear(d_model, 1)

        self.pre_softmax_proj = nn.Linear(n_head, n_head) # talking-heads attention
        self.post_softmax_proj = nn.Linear(n_head, n_head)


    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)
        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        # apply pre-softmax projection
        attn_score = self.pre_softmax_proj(attn_score) # p_l
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None].bool(), -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None].bool(), -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)

        # apply post-softmax projection
        attn_prob = self.post_softmax_proj(attn_prob) # p_w

        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)
        # output = self.scorer(output)
        output = torch.mean(output, dim=1)

        return output



class MLPSelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, d_out, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, mask):
        batch_size, seq_len, d_feat, domains = inp.size()
        inp = self.dropout(inp)
        temp = inp.contiguous().view(batch_size, seq_len, -1)
        scores_ = self.scorer(inp.contiguous().view(batch_size, seq_len, -1))
        scores_ = scores_.masked_fill((mask == 0).unsqueeze(-1), -1e9)
        scores = F.softmax(scores_, dim=-1)
        context = scores.unsqueeze(-2).expand_as(inp).mul(inp).sum(-1)
        return context, scores_

#修改模块
class ContextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, domains, n_layers=args['layer_r']):
        super(ContextEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.domains = domains
        # self.mix_attention = MultiHeadAttn(n_head = 2, d_model = len(domains) * 2 * self.hidden_size, d_head = 4, d_out　= len(self.domains), dropout=0.1)
        # self.mix_attention_c = MultiHeadAttn(n_head = 2, d_model = len(domains) * 2 * self.hidden_size, d_head = 4, d_out　= len(self.domains), dropout=0.1)

        self.mix_attention = MLPSelfAttention(2 * 2 * self.hidden_size, 2, dropout)
        self.mix_attention_c = MLPSelfAttention(len(domains) * 2 * self.hidden_size, len(domains), dropout)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, args['embeddings_dim'], padding_idx=PAD_token)
        self.odim = args['embeddings_dim']
        self.global_gru = RNN_Residual(self.odim, hidden_size, n_layers, dropout=dropout)
        self.selfatten = MultiHeadAttn(n_head=2, d_model=2 * self.hidden_size, d_head=64, dropout=dropout)

        # self.selfatten = SelfAttention(2 * self.hidden_size, dropout=self.dropout)
        for domain in domains.keys():
            setattr(self, '{}_gru'.format(domain),
                    RNN_Residual(self.odim, hidden_size, n_layers, dropout=self.dropout))
        self.MLP_H = nn.Sequential(
            nn.Linear(4 * self.hidden_size, 4 * self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(4 * self.hidden_size, 2 * self.hidden_size),
        )

        self.W = nn.Linear(2 * hidden_size, hidden_size)
        self.W_hid = nn.Linear(2 * hidden_size, hidden_size)
        self.global_classifier = nn.Sequential(
            GradientReversal(),
            CNNClassifier(2 * hidden_size, hidden_size, [2, 3], len(domains), dropout))

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, return_fusion_energy=False):
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long())
        embedded = embedded.view(input_seqs.size() + (embedded.size(-1),))  # 4,193,6,128
        embedded = torch.sum(embedded, 2).squeeze(2)
        embedded = self.dropout_layer(embedded.transpose(0, 1))
        global_outputs, global_hidden = self.global_gru(embedded, input_lengths)
        local_outputs = []
        local_outputs_AB = []
        local_outputs_AC = []
        local_outputs_BC = []
        mask = _cuda(torch.zeros((len(input_lengths), input_lengths[0])))
        for i, length in enumerate(input_lengths):
            mask[i, :length] = 1
        if args['dataset'] == 'kvr':
            for domain in {'navigate': 0, 'weather': 1}:
                local_rnn = getattr(self, '{}_gru'.format(domain))
                local_output, _ = local_rnn(embedded, input_lengths)
                local_outputs_AB.append(local_output)
            for domain in {'navigate': 0, 'schedule': 2}:
                local_rnn = getattr(self, '{}_gru'.format(domain))
                local_output, _ = local_rnn(embedded, input_lengths)
                local_outputs_AC.append(local_output)
            for domain in {'weather': 1, 'schedule': 2}:
                local_rnn = getattr(self, '{}_gru'.format(domain))
                local_output, _ = local_rnn(embedded, input_lengths)
                local_outputs_BC.append(local_output)
        elif args['dataset'] == 'woz':
            for domain in {'restaurant': 0, 'hotel': 1}:
                local_rnn = getattr(self, '{}_gru'.format(domain))
                local_output, _ = local_rnn(embedded, input_lengths)
                local_outputs_AB.append(local_output)
            for domain in {'restaurant': 0, 'attraction': 2}:
                local_rnn = getattr(self, '{}_gru'.format(domain))
                local_output, _ = local_rnn(embedded, input_lengths)
                local_outputs_AC.append(local_output)
            for domain in {'hotel': 1, 'attraction': 2}:
                local_rnn = getattr(self, '{}_gru'.format(domain))
                local_output, _ = local_rnn(embedded, input_lengths)
                local_outputs_BC.append(local_output)
        # for domain in self.domains:
        #     local_rnn = getattr(self, '{}_gru'.format(domain))
        #     local_output, _ = local_rnn(embedded, input_lengths)
        #     local_outputs.append(local_output)
        # woz:
        # for domain in {'restaurant':0,'hotel':1}:
        #     local_rnn = getattr(self, '{}_gru'.format(domain))
        #     local_output, _ = local_rnn(embedded, input_lengths)
        #     local_outputs_AB.append(local_output)
        # for domain in {'restaurant':0,'attraction':2}:
        #     local_rnn = getattr(self, '{}_gru'.format(domain))
        #     local_output, _ = local_rnn(embedded, input_lengths)
        #     local_outputs_AC.append(local_output)
        # for domain in {'hotel':1,'attraction':2}:
        #     local_rnn = getattr(self, '{}_gru'.format(domain))
        #     local_output, _ = local_rnn(embedded, input_lengths)
        #     local_outputs_BC.append(local_output)
        # kvr:
        # for domain in {'navigate':0,'weather':1}:
        #     local_rnn = getattr(self, '{}_gru'.format(domain))
        #     local_output, _ = local_rnn(embedded, input_lengths)
        #     local_outputs_AB.append(local_output)
        # for domain in {'navigate':0,'schedule':2}:
        #     local_rnn = getattr(self, '{}_gru'.format(domain))
        #     local_output, _ = local_rnn(embedded, input_lengths)
        #     local_outputs_AC.append(local_output)
        # for domain in {'weather':1,'schedule':2}:
        #     local_rnn = getattr(self, '{}_gru'.format(domain))
        #     local_output, _ = local_rnn(embedded, input_lengths)
        #     local_outputs_BC.append(local_output)

        # local_outputs, scores = self.mix_attention(torch.stack(local_outputs, dim=-1), mask)

        #AB拼,AC拼,BC拼→AB+AC+BC拼
        local_outputs_AB, scores_AB = self.mix_attention(torch.stack(local_outputs_AB, dim=-1), mask)
        local_outputs_AC, scores_AC = self.mix_attention(torch.stack(local_outputs_AC, dim=-1), mask)
        local_outputs_BC, scores_BC = self.mix_attention(torch.stack(local_outputs_BC, dim=-1), mask)
        attn_ab = F.softmax(scores_AB, dim=-1)
        attn_ac = F.softmax(scores_AC, dim=-1)
        attn_bc = F.softmax(scores_BC, dim=-1)
        local_outputs.append(local_outputs_AB)
        local_outputs.append(local_outputs_AC)
        local_outputs.append(local_outputs_BC)

        local_outputs, scores_fuse = self.mix_attention_c(torch.stack(local_outputs, dim=-1), mask)
        attn_fuse = F.softmax(scores_fuse, dim=-1)

        outputs = self.MLP_H(torch.cat((F.dropout(local_outputs, self.dropout, self.training),
                                        F.dropout(global_outputs, self.dropout, self.training)), dim=-1))

        attention_weights = {
            'AB': attn_ab.detach().cpu().numpy(),
            'AC': attn_ac.detach().cpu().numpy(),
            'BC': attn_bc.detach().cpu().numpy(),
            'FUSED': attn_fuse.detach().cpu().numpy(),
        }
        if return_fusion_energy:
            attention_weights['H_group_L2'] = (
                local_outputs.detach().float().norm(dim=-1).cpu().numpy()
            )
            attention_weights['H_integrated_L2'] = (
                global_outputs.detach().float().norm(dim=-1).cpu().numpy()
            )
            attention_weights['H_fused_L2'] = outputs.detach().float().norm(dim=-1).cpu().numpy()

        hidden = self.selfatten(outputs)
        outputs_ = self.W(outputs)
        hidden_ = self.W(hidden)
        label = self.global_classifier(global_outputs)
        return outputs_, hidden_, label, scores_fuse, attention_weights


class ExternalKnowledge(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        for bi in range(full_memory.size(0)):
            start, end = kb_len[bi], kb_len[bi] + conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    def get_ck(self, hop, story, story_size):
        embed = self.C[hop](story.contiguous().view(story_size[0], -1))
        embed = embed.view(story_size + (embed.size(-1),))
        embed = torch.sum(embed, 2).squeeze(2)
        return embed

    def get_ck_local(self, hop, story, story_size, domains):
        embed = _cuda(torch.zeros((story_size + (self.embedding_dim,))))
        for i, domain in enumerate(domains):
            embed[i] = self.__getattribute__('C_{}_'.format(domain))[hop](story.contiguous()[i])
        embed = torch.sum(embed, 2).squeeze(2)
        return embed

    def load_memory(self, story, kb_len, conv_len, hidden, dh_outputs, domains):
        # Forward multiple hop mechanism
        u = [hidden.squeeze(0)]
        story_size = story.size()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.get_ck(hop, story, story_size)
            embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, dh_outputs)
            embed_A = self.dropout_layer(embed_A)

            if (len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
            prob_logit = torch.sum(embed_A * u_temp, 2)
            prob_ = self.softmax(prob_logit)

            embed_C = self.get_ck(hop + 1, story, story_size)
            embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, dh_outputs)

            prob = prob_.unsqueeze(2).expand_as(embed_C)
            o_k = torch.sum(embed_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
            self.m_story.append(embed_A)
        self.m_story.append(embed_C)
        return self.sigmoid(prob_logit), u[-1]

    def forward(self, query_vector, global_pointer):
        u = [query_vector]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A)
            if (len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_logits = torch.sum(m_A * u_temp, 2)
            prob_soft = self.softmax(prob_logits)
            m_C = self.m_story[hop + 1]
            m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)
            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return prob_soft, prob_logits


class LocalMemoryDecoder(nn.Module):
    def __init__(self, shared_emb, lang, hidden_dim, hop, dropout, domains=None):
        super(LocalMemoryDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.max_hops = hop
        self.C = shared_emb
        self.embedding_dim = shared_emb.embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.domains = domains

        self.sketch_rnn_global = nn.GRU(self.embedding_dim, hidden_dim, dropout=dropout)
        for index, domain in enumerate(domains):
            local = nn.GRU(self.embedding_dim, hidden_dim, dropout=dropout)
            self.add_module("sketch_rnn_local_{}".format(index), local)
        self.sketch_rnn_local = AttrProxy(self, "sketch_rnn_local_")
        self.mix_attention = MLPSelfAttention(len(domains) * hidden_dim, len(domains), dropout)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(2 * hidden_dim, hidden_dim)
        self.MLP = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        self.attn_table = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.projector2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.domain_emb = nn.Embedding(len(domains), self.embedding_dim)

        self.global_classifier = nn.Sequential(GradientReversal(),
                                               CNNClassifier(hidden_dim, hidden_dim, [2, 3], len(domains), dropout))

    def get_p_vocab(self, hidden, H):
        cond = self.attn_table(torch.cat((H, hidden.unsqueeze(1).expand_as(H)), dim=-1))
        cond = F.softmax(cond.squeeze(-1), dim=-1)
        hidden_ = cond.unsqueeze(-1).expand_as(H).mul(H).sum(-2)
        context = F.tanh(self.projector2(torch.cat((hidden, hidden_), dim=-1).unsqueeze(0)))
        p_vocab = self.attend_vocab(self.C.weight, context.squeeze(0))
        return p_vocab, context

    def forward(self, extKnow, story_size, story_lengths, copy_list, encode_hidden, target_batches, max_target_length,
                batch_size, use_teacher_forcing, get_decoded_words, global_pointer, H=None, global_entity_type=None,
                domains=None, sample_temperature=0.0, return_path_logprob=False):
        # Initialize variables for vocab and pointer
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, story_size[1]))
        decoder_input = _cuda(self.domain_emb(domains.view(-1, ))) + self.C(
            _cuda(torch.LongTensor([SOS_token] * batch_size)))
        memory_mask_for_step = _cuda(torch.ones(story_size[0], story_size[1]))
        decoded_fine, decoded_coarse = [], []
        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)
        hidden_locals = []
        for i in range(len(self.domains)):
            hidden_locals.append(hidden.clone())

        mask = _cuda(torch.ones((len(story_lengths), 1)))
        global_hiddens = []
        local_hiddens = []
        scores = []
        path_logprob = None
        active = None
        if return_path_logprob and (not use_teacher_forcing):
            dev = encode_hidden.device
            path_logprob = torch.zeros(batch_size, device=dev)
            active = torch.ones(batch_size, dtype=torch.bool, device=dev)
        # long, withen = target_batches.size()
        # if withen != max_target_length:
        #     print(withen)
        # Start to generate word-by-word
        for t in range(max_target_length):
            if t != 0:
                decoder_input = self.C(decoder_input)
            embed_q = self.dropout_layer(decoder_input)
            embed_q = embed_q.view(1, -1, self.embedding_dim)
            _, hidden = self.sketch_rnn_global(embed_q, hidden)
            hidden_locals_ = []
            for domain in self.domains.values():
                hidden_locals_.append(self.sketch_rnn_local[domain](embed_q, hidden_locals[domain])[1])
            hidden_locals = hidden_locals_
            hidden_local, score = self.mix_attention(torch.stack(hidden_locals, dim=-1).transpose(0, 1),
                                                     mask)
            hidden_local, score = hidden_local.transpose(0, 1), score.transpose(0, 1)
            scores.append(score)
            query_vector = self.MLP(torch.cat((F.dropout(hidden, self.dropout, self.training),
                                               F.dropout(hidden_local, self.dropout, self.training)), dim=-1))
            global_hiddens.append(hidden)
            local_hiddens.append(hidden_local)

            p_vocab, context = self.get_p_vocab(query_vector[0], H)

            all_decoder_outputs_vocab[t] = p_vocab
            if sample_temperature and sample_temperature > 0 and (not use_teacher_forcing):
                probs = F.softmax(p_vocab / float(sample_temperature), dim=-1)
                topvi = torch.multinomial(probs, num_samples=1)
            else:
                _, topvi = p_vocab.max(dim=-1, keepdim=True)

            if return_path_logprob and (not use_teacher_forcing) and path_logprob is not None and active is not None:
                log_sm = F.log_softmax(p_vocab, dim=-1)
                idx = topvi.long().squeeze(1).clamp(0, log_sm.size(1) - 1)
                tok_lp = log_sm.gather(1, idx.unsqueeze(1)).squeeze(1)
                path_logprob = path_logprob + tok_lp * active.float()
                active = active & ~idx.eq(EOS_token)

            # query the external konwledge using the hidden state of sketch RNN
            prob_soft, prob_logits = extKnow(context[0], global_pointer)
            all_decoder_outputs_ptr[t] = prob_logits

            if use_teacher_forcing:
                decoder_input = target_batches[:, t]
            else:
                decoder_input = topvi.squeeze()

            if get_decoded_words:

                search_len = min(5, min(story_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.topk(search_len, dim=1)

                ptr_w = float(args.get('decode_ptr_score_weight', 1.0) or 0.0)
                ptr_logsm = None
                if (
                    ptr_w != 0.0
                    and return_path_logprob
                    and (not use_teacher_forcing)
                    and path_logprob is not None
                    and active is not None
                ):
                    logits_m = prob_logits.masked_fill(memory_mask_for_step == 0, -1e9)
                    ptr_logsm = F.log_softmax(logits_m, dim=1)

                temp_f, temp_c = [], []

                for bi in range(batch_size):
                    token = topvi[bi].item()
                    temp_c.append(self.lang.index2word[token])

                    if '@' in self.lang.index2word[token]:
                        gold_type = self.lang.index2word[token]
                        cw = 'UNK'
                        sid_used = None
                        for i in range(search_len):
                            if toppi[bi, i].item() < story_lengths[bi] - 1:
                                sid = int(toppi[bi, i].item())
                                sid_used = sid
                                cw = copy_list[bi][sid]
                                if ptr_logsm is not None and active[bi]:
                                    path_logprob[bi] = path_logprob[bi] + ptr_w * ptr_logsm[bi, sid]
                                break
                        temp_f.append(cw)

                        if args['record'] and sid_used is not None:
                            memory_mask_for_step[bi, sid_used] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])

                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)

        label = self.global_classifier(torch.cat(global_hiddens, dim=0).transpose(0, 1))
        scores = torch.cat(scores, dim=0).transpose(0, 1).contiguous()
        return all_decoder_outputs_vocab, all_decoder_outputs_ptr, decoded_fine, decoded_coarse, label, scores, path_logprob

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        return scores_


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
