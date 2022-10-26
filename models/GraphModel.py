import torch
import torch.nn as nn
import torch.nn.functional as F
from models.allennlp_beamsearch import BeamSearch
import math

FRAME_LENGTH = 26
OBJECT_LENGTH = 20

class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size, graph_size, dropout=0.5):
        super(GraphConvolution, self).__init__()
        self.graph_size = graph_size
        self.dropout = dropout
        """RELATION FEATURE GRAPH"""
        self.rel_linear = nn.Linear(input_size, graph_size)
        nn.init.xavier_normal_(self.rel_linear.weight)
        self.relation_size = 53
        self.rel_embed = nn.Embedding(self.relation_size, graph_size)
        nn.init.xavier_normal_(self.rel_embed.weight)
        self.rel_drop = nn.Dropout(dropout)
        """TEMPORAL FEATURE GRAPH"""
        self.tem_linear = nn.Linear(input_size, graph_size)
        nn.init.xavier_normal_(self.tem_linear.weight)
        """OUTPUT"""
        self.drop = nn.Dropout(p=dropout)
        self.activation = nn.Tanh()
        self.bn = nn.BatchNorm1d(FRAME_LENGTH * OBJECT_LENGTH)

    def forward(self, region_feats, region_masks, rel_feats, tem_feats):
        self.batch_size = region_feats.size(0)
        self.frame_size = region_feats.size(1)
        self.object_size = region_feats.size(2)
        """RELATIONAL FEATURE GRAPH"""
        rel_region_feats = self.rel_linear(region_feats)
        rel_region_feats = rel_region_feats.view(self.batch_size * self.frame_size, self.object_size, -1)  # [832, 20, 1000]
        rel_feats_feature = self.rel_embed(rel_feats.long())  # [32, 26, 20, 20, 1000]
        rel_feats_feature = rel_feats_feature.view(self.batch_size * self.frame_size, self.object_size, self.object_size, -1)  # [832, 20, 20, 1000]
        rel_feats_feature = self.rel_drop(rel_feats_feature)
        rel_feats_masks = (rel_feats > 0).float()
        rel_feats_masks = rel_feats_masks.view(self.batch_size * self.frame_size, self.object_size, self.object_size)  # [832, 20, 20]
        rel_feats_feature = rel_feats_feature * rel_feats_masks.unsqueeze(-1)  # [832, 20, 20, 1000]
        rel_region_feats = rel_region_feats.unsqueeze(2).expand(self.batch_size * self.frame_size, self.object_size, self.object_size, -1)
        rel_region_feats = rel_region_feats * rel_feats_masks.unsqueeze(-1)  # [832, 20, 20, 1000]
        value = rel_feats_feature + rel_region_feats
        value = torch.sum(value, dim=2) / torch.sum(rel_feats_masks, dim=2).unsqueeze(-1)  # [832, 20, 1000]
        rel_region_feats = value.view(self.batch_size, self.frame_size * self.object_size, -1)
        """TEMPORAL FEATURE GRAPH"""
        tem_region_feats = self.tem_linear(region_feats)
        tem_region_feats = tem_region_feats.view(self.batch_size, self.frame_size * self.object_size, -1)
        tem_sum = torch.sum(tem_feats, dim=2)
        tem_region_feats = torch.bmm(tem_feats, tem_region_feats) / tem_sum.unsqueeze(-1)
        tem_region_feats = tem_region_feats.view(self.batch_size, self.frame_size * self.object_size, -1)
        """OUTPUT"""
        output = torch.cat([rel_region_feats, tem_region_feats], dim=-1)
        output = self.drop(self.bn(self.activation(output)))
        return output

class BiEncode(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(BiEncode, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        """BILSTM"""
        self.left_lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.left_h_drop = nn.Dropout(p=dropout)
        self.left_c_drop = nn.Dropout(p=dropout)
        nn.init.orthogonal_(self.left_lstm.weight_hh)
        nn.init.xavier_normal_(self.left_lstm.weight_ih)
        nn.init.constant_(self.left_lstm.bias_ih, 0)
        nn.init.constant_(self.left_lstm.bias_hh, 0)
        self.right_lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.right_h_drop = nn.Dropout(p=dropout)
        self.right_c_drop = nn.Dropout(p=dropout)
        nn.init.orthogonal_(self.right_lstm.weight_hh)
        nn.init.xavier_normal_(self.right_lstm.weight_ih)
        nn.init.constant_(self.right_lstm.bias_ih, 0)
        nn.init.constant_(self.right_lstm.bias_hh, 0)

    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(batch_size, self.hidden_size).zero_()
        lstm_state_c = d.data.new(batch_size, self.hidden_size).zero_()
        return lstm_state_h, lstm_state_c

    def forward(self, embed_feats):
        batch_size = embed_feats.size(0)
        """STATE INITIALIZATION"""
        left_h, left_c = self._init_lstm_state(embed_feats)
        right_h, right_c = self._init_lstm_state(embed_feats)
        """OUTPUTS"""
        outputs = embed_feats.new_zeros(batch_size, FRAME_LENGTH, self.hidden_size * 2)
        """BIDIRECTIONAL ENCODE"""
        for t in range(FRAME_LENGTH):
            """LEFT ENCODE"""
            left_feat = embed_feats[:, t]
            left_h, left_c = self.left_lstm(left_feat, (left_h, left_c))
            left_h = self.left_h_drop(left_h)
            left_c = self.left_c_drop(left_c)
            outputs[:, t, :self.hidden_size] = left_h
            """RIGHT ENCODE"""
            right_t = FRAME_LENGTH - 1 - t
            right_feat = embed_feats[:, right_t]
            right_h, right_c = self.right_lstm(right_feat, (right_h, right_c))
            right_h = self.right_h_drop(right_h)
            right_c = self.right_c_drop(right_c)
            outputs[:, right_t, self.hidden_size:] = right_h
        """RETURN"""
        return outputs

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.a_feature_size = opt.a_feature_size
        self.m_feature_size = opt.m_feature_size
        self.hidden_size = opt.feat_projected_size
        self.input_size = opt.feat_projected_size
        """FRAME FEATURE EMBEDDING"""
        self.frame_feature_embed = nn.Linear(opt.a_feature_size, self.input_size)
        self.frame_bn = nn.BatchNorm1d(FRAME_LENGTH)
        self.frame_activation = nn.Tanh()
        self.frame_drop = nn.Dropout(p=opt.dropout)
        nn.init.xavier_normal_(self.frame_feature_embed.weight)
        """I3D FEATURE EMBEDDING"""
        self.i3d_feature_embed = nn.Linear(opt.m_feature_size, self.input_size)
        self.i3d_bn = nn.BatchNorm1d(FRAME_LENGTH)
        self.i3d_activation = nn.Tanh()
        self.i3d_drop = nn.Dropout(p=opt.dropout)
        nn.init.xavier_normal_(self.i3d_feature_embed.weight)
        """BILSTM"""
        self.bi_lstm = BiEncode(self.input_size * 2, self.hidden_size, opt.dropout)
        """REGION FEATURE EMBEDDING"""
        self.region_feature_embed = nn.Linear(opt.region_feature_size, opt.feat_projected_size)
        self.region_bn = nn.BatchNorm2d(FRAME_LENGTH, OBJECT_LENGTH)
        self.region_activation = nn.Tanh()
        self.region_drop = nn.Dropout(p=opt.dropout)
        nn.init.xavier_normal_(self.region_feature_embed.weight)
        """GRAPH CONVOLUTION"""
        self.graph_convolution = GraphConvolution(opt.feat_projected_size, opt.feat_projected_size, opt.graph_size, opt.dropout)
        """VISUAL EMBEDDING"""
        self.visual_feature_embed = nn.Linear(opt.graph_size * 2, opt.feat_projected_size)
        self.visual_bn = nn.BatchNorm1d(FRAME_LENGTH * OBJECT_LENGTH)
        self.visual_activation = nn.Tanh()
        self.visual_drop = nn.Dropout(p=opt.dropout)
        nn.init.xavier_normal_(self.visual_feature_embed.weight)
        nn.init.constant_(self.visual_feature_embed.bias, 0)

    def forward(self, cnn_feats, region_feats, rel_feats, tem_feats):
        assert self.a_feature_size + self.m_feature_size == cnn_feats.size(2)
        frame_feats = cnn_feats[:, :, :self.a_feature_size].contiguous()
        i3d_feats = cnn_feats[:, :, -self.m_feature_size:].contiguous()
        """FRAME FEATURE"""
        embedded_frame_feats = self.frame_activation(self.frame_feature_embed(frame_feats))
        embedded_frame_feats = self.frame_drop(self.frame_bn(embedded_frame_feats))
        """I3D FEATURE"""
        embedded_i3d_feats = self.i3d_activation(self.i3d_feature_embed(i3d_feats))
        embedded_i3d_feats = self.i3d_drop(self.i3d_bn(embedded_i3d_feats))
        """BILSTM"""
        input_feats = torch.cat([embedded_frame_feats, embedded_i3d_feats], dim=-1)
        i3d_feats = self.bi_lstm(input_feats)
        """REGION FEATURE"""
        object_masks = (torch.sum(region_feats, dim=-1) != 0).float()
        embedded_region_feats = self.region_activation(self.region_feature_embed(region_feats))
        embedded_region_feats = self.region_drop(self.region_bn(embedded_region_feats))
        """GRAPH CONVOLUTION"""
        self.batch_size = region_feats.size(0)
        self.frame_size = region_feats.size(1)
        self.object_size = region_feats.size(2)
        object_feats = self.graph_convolution(embedded_region_feats, object_masks, rel_feats, tem_feats)
        """VISUAL EMBEDDING"""
        object_feats = self.visual_activation(self.visual_feature_embed(object_feats))
        object_feats = self.visual_drop(self.visual_bn(object_feats))
        object_masks = object_masks.view(self.batch_size, self.frame_size * self.object_size)
        return i3d_feats, object_feats, object_masks

class SoftAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, att_size):
        super(SoftAttention, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.wh = nn.Linear(hidden_size, att_size)
        self.wv = nn.Linear(feat_size, att_size)
        self.wa = nn.Linear(att_size, 1)

    def forward(self, feats, key, mask=None):
        v = self.wv(feats)
        inputs = self.wh(key).unsqueeze(1).expand_as(v) + v
        alpha = self.wa(torch.tanh(inputs)).squeeze(-1)
        if mask is not None:
            alpha = alpha.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(alpha, dim = -1)
        att_feats = torch.bmm(alpha.unsqueeze(1), feats).squeeze(1)
        return att_feats, alpha

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.region_attn = SoftAttention(opt.feat_projected_size, opt.hidden_size, opt.att_size)
        self.frame_attn = SoftAttention(opt.feat_projected_size * 2, opt.hidden_size, opt.att_size)
        self.feature_fuse = nn.Linear(opt.feat_projected_size * 3, opt.hidden_size)
        self.feature_activation = nn.Tanh()
        self.dropout = nn.Dropout(p=opt.dropout)
        nn.init.xavier_normal_(self.feature_fuse.weight)
        nn.init.constant_(self.feature_fuse.bias, 0)

    def forward(self, frame_feats, object_feats, object_masks, hidden_state):
        object_feats_att, _ = self.region_attn(object_feats, hidden_state, object_masks)
        motion_feats_att, _ = self.frame_attn(frame_feats, hidden_state)
        loc_feat = torch.cat([object_feats_att, motion_feats_att], dim = -1)
        output = self.dropout(self.feature_activation(self.feature_fuse(loc_feat)))
        return output

class Decoder(nn.Module):
    def __init__(self, opt, vocab):
        super(Decoder, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.word_size = opt.word_size
        self.max_words = opt.max_words
        self.vocab = vocab
        self.vocab_size = len(vocab)
        """WORD EMBEDDING"""
        self.word_embed = nn.Embedding(self.vocab_size, self.word_size)
        nn.init.xavier_normal_(self.word_embed.weight)
        self.word_drop = nn.Dropout(p=opt.dropout)
        """LSTM INIT"""
        self.lstm_init = nn.Linear(opt.feat_projected_size * 3, self.word_size)
        self.lstm_init_activation = nn.Tanh()
        self.lstm_init_drop = nn.Dropout(p=opt.dropout)
        nn.init.xavier_normal_(self.lstm_init.weight)
        """ATT LSTM"""
        self.att_lstm = nn.LSTMCell(opt.word_size + opt.hidden_size + opt.feat_projected_size * 3, opt.hidden_size)
        nn.init.orthogonal_(self.att_lstm.weight_hh)
        nn.init.xavier_normal_(self.att_lstm.weight_ih)
        nn.init.constant_(self.att_lstm.bias_ih, 0)
        nn.init.constant_(self.att_lstm.bias_hh, 0)
        self.att_lstm_drop = nn.Dropout(p=opt.dropout)
        self.att_cell_drop = nn.Dropout(p=opt.dropout)
        """LANG LSTM"""
        self.attention = Attention(opt)
        self.lang_lstm = nn.LSTMCell(opt.hidden_size * 2, opt.hidden_size)
        nn.init.orthogonal_(self.lang_lstm.weight_hh)
        nn.init.xavier_normal_(self.lang_lstm.weight_ih)
        nn.init.constant_(self.lang_lstm.bias_ih, 0)
        nn.init.constant_(self.lang_lstm.bias_hh, 0)
        self.lang_lstm_drop = nn.Dropout(p=opt.dropout)
        self.lang_cell_drop = nn.Dropout(p=opt.dropout)
        """OUTPUT"""
        self.out_fc = nn.Linear(opt.hidden_size * 3, self.hidden_size)
        self.out_fc_activation = nn.Tanh()
        self.out_fc_dropout = nn.Dropout(p=opt.dropout)
        nn.init.xavier_normal_(self.out_fc.weight)
        nn.init.constant_(self.out_fc.bias, 0)
        self.word_restore = nn.Linear(self.hidden_size, self.vocab_size)
        nn.init.xavier_normal_(self.word_restore.weight)
        nn.init.constant_(self.word_restore.bias, 0)
        """BEAM SEARCH"""
        self.beam_search = BeamSearch(vocab('<end>'), self.max_words, opt.beam_size, per_node_beam_size=opt.beam_size)

    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(batch_size, self.hidden_size).zero_()
        lstm_state_c = d.data.new(batch_size, self.hidden_size).zero_()
        return lstm_state_h, lstm_state_c

    def decode_tokens(self, tokens):
        words = []
        for token in tokens:
            if token == self.vocab('<end>'):
                break
            word = self.vocab.idx2word[token]
            words.append(word)
        captions = ' '.join(words)
        return captions

    def sample_next_word(self, logprobs, sample_method, temperature=1.0):
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()
        else:
            logprobs = logprobs / temperature
            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))  # gather the logprobs at sampled positions
        return it, sampleLogprobs

    def forward(self, frame_feats, object_feats, object_masks, captions, teacher_forcing_ratio=1.0):
        self.batch_size = frame_feats.size(0)
        infer = True if captions is None else False
        """GLOBAL FEATURE REPRESENTATION"""
        global_frame_feat = torch.mean(frame_feats, dim=1)
        global_object_feat = torch.sum(object_feats, dim=1) / torch.sum(object_masks.unsqueeze(-1), dim=1)
        global_feat = torch.cat([global_frame_feat, global_object_feat], dim=1)
        """STATE INITIALIZATION"""
        init_word = self.lstm_init_drop(self.lstm_init_activation(self.lstm_init(global_feat)))
        lang_lstm_h, lang_lstm_c = self._init_lstm_state(global_feat)
        att_lstm_h, att_lstm_c = self._init_lstm_state(global_feat)
        att_lstm_h, att_lstm_c = self.att_lstm(torch.cat([lang_lstm_h, init_word, global_feat], dim=1), (att_lstm_h, att_lstm_c))
        att_lstm_h = self.att_lstm_drop(att_lstm_h)
        att_lstm_c = self.att_cell_drop(att_lstm_c)
        feats = self.attention(frame_feats, object_feats, object_masks, att_lstm_h)
        decoder_input = torch.cat([feats, att_lstm_h], dim=-1)
        lang_lstm_h, lang_lstm_c = self.lang_lstm(decoder_input, (lang_lstm_h, lang_lstm_c))
        lang_lstm_h = self.lang_lstm_drop(lang_lstm_h)
        lang_lstm_c = self.lang_cell_drop(lang_lstm_c)
        """START TOKEN"""
        start_id = self.vocab('<start>')
        start_id = global_feat.data.new(global_feat.size(0)).long().fill_(start_id)
        word = self.word_embed(start_id)
        word = self.word_drop(word)
        """PREDICTION"""
        outputs = []
        logprobs_output = []
        if not infer or self.opt.beam_size == 1:
            for i in range(self.max_words):
                if not infer and captions[:, i].data.sum() == 0:
                    break
                """ATT LSTM"""
                att_lstm_h, att_lstm_c = self.att_lstm(torch.cat([lang_lstm_h, word, global_feat], dim=1), (att_lstm_h, att_lstm_c))
                att_lstm_h = self.att_lstm_drop(att_lstm_h)
                att_lstm_c = self.att_cell_drop(att_lstm_c)
                """LANG LSTM"""
                feats = self.attention(frame_feats, object_feats, object_masks, att_lstm_h)
                decoder_input = torch.cat([feats, att_lstm_h], dim=-1)
                lang_lstm_h, lang_lstm_c = self.lang_lstm(decoder_input, (lang_lstm_h, lang_lstm_c))
                lang_lstm_h = self.lang_lstm_drop(lang_lstm_h)
                lang_lstm_c = self.lang_cell_drop(lang_lstm_c)
                """WORD RESTORE"""
                decoder_output = self.out_fc_dropout(self.out_fc_activation(self.out_fc(torch.cat([lang_lstm_h, decoder_input], dim=1))))
                word_logits = self.word_restore(decoder_output)
                use_teacher_forcing = not infer
                if use_teacher_forcing:
                    word_id = captions[:, i]
                else:
                    logprobs = F.log_softmax(word_logits, dim=-1)
                    word_id, sampleLogprobs = self.sample_next_word(logprobs, self.opt.sample_method)
                word = self.word_embed(word_id)
                word = self.word_drop(word)
                if infer:
                    outputs.append(word_id)
                    logprobs_output.append(sampleLogprobs)
                else:
                    outputs.append(word_logits)
            outputs = torch.stack(outputs, dim=1)
        else:
            start_state = {'att_lstm_h': att_lstm_h, 'att_lstm_c': att_lstm_c,
                           'lang_lstm_h': lang_lstm_h, 'lang_lstm_c': lang_lstm_c, 'global_feat': global_feat,
                           'object_feats': object_feats, 'frame_feats': frame_feats, 'object_masks': object_masks}
            predictions, log_prob = self.beam_search.search(start_id, start_state, self.beam_step)
            max_prob, max_index = torch.topk(log_prob, 1)  # b*1
            max_index = max_index.squeeze(1)  # b
            for i in range(self.batch_size):
                outputs.append(predictions[i, max_index[i], :])
            outputs = torch.stack(outputs)
        if not infer:
            return outputs
        else:
            if self.opt.beam_size > 1:
                return outputs, None
            else:
                logprobs_output = torch.stack(logprobs_output, dim=1)
                return outputs, logprobs_output

    def beam_step(self, last_predictions, current_state):
        group_size = last_predictions.size(0)  # batch_size or batch_size*beam_size
        batch_size = self.batch_size
        log_probs = []
        new_state = {}
        num = int(group_size / batch_size)  # 1 or beam_size
        for k, state in current_state.items():
            if isinstance(state, list):
                state = torch.stack(state, dim=1)
            _, *last_dims = state.size()
            current_state[k] = state.reshape(batch_size, num, *last_dims)
            new_state[k] = []
        for i in range(num):
            # read current state
            att_lstm_h = current_state['att_lstm_h'][:, i, :]
            att_lstm_c = current_state['att_lstm_c'][:, i, :]
            lang_lstm_h = current_state['lang_lstm_h'][:, i, :]
            lang_lstm_c = current_state['lang_lstm_c'][:, i, :]
            frame_feats = current_state['frame_feats'][:, i, :]
            object_feats = current_state['object_feats'][:, i, :]
            object_masks = current_state['object_masks'][:, i, :]
            global_feat = current_state['global_feat'][:, i, :]
            word_id = last_predictions.reshape(batch_size, -1)[:, i]
            word = self.word_embed(word_id)
            """ATT LSTM"""
            att_lstm_h, att_lstm_c = self.att_lstm(torch.cat([lang_lstm_h, word, global_feat], dim=1), (att_lstm_h, att_lstm_c))
            att_lstm_h = self.att_lstm_drop(att_lstm_h)
            att_lstm_c = self.att_cell_drop(att_lstm_c)
            """LANG LSTM"""
            feats = self.attention(frame_feats, object_feats, object_masks, att_lstm_h)
            decoder_input = torch.cat([feats, att_lstm_h], dim=-1)
            lang_lstm_h, lang_lstm_c = self.lang_lstm(decoder_input, (lang_lstm_h, lang_lstm_c))
            lang_lstm_h = self.lang_lstm_drop(lang_lstm_h)
            lang_lstm_c = self.lang_cell_drop(lang_lstm_c)
            """WORD RESTORE"""
            decoder_output = self.out_fc_dropout(self.out_fc_activation(self.out_fc(torch.cat([lang_lstm_h, decoder_input], dim=1))))
            word_logits = self.word_restore(decoder_output)
            batch_size = word_logits.size(0)
            logprobs_masks = frame_feats.new_ones(batch_size, self.vocab_size)
            for bdash in range(batch_size):
                logprobs_masks[bdash, word_id[bdash]] = 0
                logprobs_masks[bdash, self.vocab('<unk>')] = 0
            word_logits = word_logits.masked_fill(logprobs_masks == 0, -1e9)
            log_prob = F.log_softmax(word_logits, dim=1)  # b*v
            log_probs.append(log_prob)
            # update new state
            new_state['att_lstm_h'].append(att_lstm_h)
            new_state['att_lstm_c'].append(att_lstm_c)
            new_state['lang_lstm_h'].append(lang_lstm_h)
            new_state['lang_lstm_c'].append(lang_lstm_c)
            new_state['frame_feats'].append(frame_feats)
            new_state['object_feats'].append(object_feats)
            new_state['object_masks'].append(object_masks)
            new_state['global_feat'].append(global_feat)
        log_probs = torch.stack(log_probs, dim=0).permute(1, 0, 2).reshape(group_size, -1)  # group_size*vocab_size
        # transform new state
        # from list to tensor(batch_size*beam_size, *)
        for k, state in new_state.items():
            new_state[k] = torch.stack(state, dim=0)  # (beam_size, batch_size, *)
            _, _, *last_dims = new_state[k].size()
            dim_size = len(new_state[k].size())
            dim_size = range(2, dim_size)
            new_state[k] = new_state[k].permute(1, 0, *dim_size)  # (batch_size, beam_size, *)
            new_state[k] = new_state[k].reshape(group_size, *last_dims)  # (batch_size*beam_size, *)
        return (log_probs, new_state)

class GraphModel(nn.Module):
    def __init__(self, opt, vocab):
        super(GraphModel, self).__init__()
        print('--GraphModel.py--')
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt, vocab)

    def forward(self, cnn_feats, region_feats, rel_feats, tem_feats, rel_masks, captions, teacher_forcing_ratio=1.0):
        frame_feats, object_feats, object_masks = self.encoder(cnn_feats, region_feats, rel_feats, tem_feats)
        outputs = self.decoder(frame_feats, object_feats, object_masks, captions, teacher_forcing_ratio)
        return outputs