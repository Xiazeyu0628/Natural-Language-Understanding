import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils
from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
from seq2seq.models import register_model, register_model_architecture


@register_model('lstm')
class LSTMModel(Seq2SeqModel):
    """ Defines the sequence-to-sequence model class. """

    def __init__(self,
                 encoder,
                 decoder):

        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # 实例化 parser 对象后，使用add_argument函数添加参数，使用parse_args解析参数
        parser.add_argument('--encoder-embed-dim',  default=64,type=int, help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path',default=None, help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-hidden-size', default=64, type=int, help='encoder hidden size')
        parser.add_argument('--encoder-num-layers', default=1, type=int, help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional',default='True',  help='bidirectional encoder')
        parser.add_argument('--encoder-dropout-in', default=0.25, help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', default=0.25,help='dropout probability for encoder output')

        parser.add_argument('--decoder-embed-dim',default=64, type=int, help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path',default=None, help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', default=128, type=int, help='decoder hidden size')
        parser.add_argument('--decoder-num-layers', default=1, type=int, help='number of decoder layers')
        parser.add_argument('--decoder-dropout-in',default=0.25, type=float, help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out',default=0.25, type=float, help='dropout probability for decoder output')
        parser.add_argument('--decoder-use-attention',default='True', help='decoder attention')
        parser.add_argument('--decoder-use-lexical-model',default='False', help='toggle for the lexical model')

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """ Constructs the model. """
        #base_architecture(args)   # 把默认值都读进入了
        encoder_pretrained_embedding = None
        decoder_pretrained_embedding = None

        # Load pre-trained embeddings, if desired
        if args.encoder_embed_path:
            encoder_pretrained_embedding = utils.load_embedding(args.encoder_embed_path, src_dict)
        if args.decoder_embed_path:
            decoder_pretrained_embedding = utils.load_embedding(args.decoder_embed_path, tgt_dict)

        # Construct the encoder
        encoder = LSTMEncoder(dictionary=src_dict,
                              embed_dim=args.encoder_embed_dim,
                              hidden_size=args.encoder_hidden_size,
                              num_layers=args.encoder_num_layers,
                              bidirectional=args.encoder_bidirectional,
                              dropout_in=args.encoder_dropout_in,
                              dropout_out=args.encoder_dropout_out,
                              pretrained_embedding=encoder_pretrained_embedding)

        # Construct the decoder
        decoder = LSTMDecoder(dictionary=tgt_dict,
                              embed_dim=args.decoder_embed_dim,
                              hidden_size=args.decoder_hidden_size,
                              num_layers=args.decoder_num_layers,
                              dropout_in=args.decoder_dropout_in,
                              dropout_out=args.decoder_dropout_out,
                              pretrained_embedding=decoder_pretrained_embedding,
                              use_attention=bool(eval(args.decoder_use_attention)),
                              use_lexical_model=bool(eval(args.decoder_use_lexical_model)))
        return cls(encoder, decoder)


class LSTMEncoder(Seq2SeqEncoder):
    """ Defines the encoder class. """

    def __init__(self,
                 dictionary,
                 embed_dim=64,
                 hidden_size=64,
                 num_layers=1,
                 bidirectional=True,
                 dropout_in=0.25,
                 dropout_out=0.25,
                 pretrained_embedding=None):

        super().__init__(dictionary)
        self.dictionary = dictionary
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size   # 64
        self.output_dim = 2 * hidden_size if bidirectional else hidden_size   # 128

        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)

        dropout_lstm = dropout_out if num_layers > 1 else 0.
        self.lstm = nn.LSTM(input_size=embed_dim,       #  The number of expected features in the input x, 也是input_size
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_lstm,
                            bidirectional=bool(bidirectional))

    def forward(self, src_tokens, src_lengths):
        """ Performs a single forward pass through the instantiated encoder sub-network. """  # 通过实例化的编码器子网络执行一次向前传递
        # Embed tokens and apply dropout
        batch_size, src_time_steps = src_tokens.size()
        src_embeddings = self.embedding(src_tokens)   # self.embedding 已经是tensor类型
        _src_embeddings = F.dropout(src_embeddings, p=self.dropout_in, training=self.training)

        # Transpose batch: [batch_size, src_time_steps, num_features] -> [src_time_steps, batch_size, num_features/input_size] 不是batch first
        src_embeddings = _src_embeddings.transpose(0, 1)

        # Pack embedded tokens into a PackedSequence
        # 输入是长度不同的embedded tokens 返回的是打包成一条后。
        packed_source_embeddings = nn.utils.rnn.pack_padded_sequence(src_embeddings, src_lengths.data.tolist())

        # Pass source input through the recurrent layer(s)
        if self.bidirectional:
            state_size = 2 * self.num_layers, batch_size, self.hidden_size
        else:
            state_size = self.num_layers, batch_size, self.hidden_size

        # Returns a Tensor of size size filled with 0.
        hidden_initial = src_embeddings.new_zeros(*state_size)
        context_initial = src_embeddings.new_zeros(*state_size)

        packed_outputs, (final_hidden_states, final_cell_states) = self.lstm(packed_source_embeddings,
                                                                             (hidden_initial, context_initial))

        # h_n of shape (num_layers , batch, hidden_size): tensor containing the hidden state for t = seq_len.

        # Unpack LSTM outputs and optionally apply dropout (dropout currently disabled)
        # return (padded sequence, Tensor containing the list of lengths of each sequence)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=0.)
        lstm_output = F.dropout(lstm_output, p=self.dropout_out, training=self.training)
        #lstm_output_size = lstm_output.size()   # [src_time_steps, batch_size, output_dims]
        assert list(lstm_output.size()) == [src_time_steps, batch_size, self.output_dim]  # sanity check

        '''
        ___QUESTION-1-DESCRIBE-A-START___
        Describe what happens when self.bidirectional is set to True. 
        What is the difference between final_hidden_states and final_cell_states?
        '''
        if self.bidirectional:
            def combine_directions(outs):
                # 在给定维度上对输入的张量序列seq 进行连接操作
                # size of outs  (num_layers * num_directions, batch, hidden_size)  （2， 10， 64）
                # outs[0:2:2]   (1,10,64)
                # outs[1:2:2]   (1,10,64)
                aa = outs.size(0)
                return torch.cat([outs[0: outs.size(0): 2], outs[1: outs.size(0): 2]], dim=2)
            final_hidden_states = combine_directions(final_hidden_states)
            final_cell_states = combine_directions(final_cell_states)
        '''___QUESTION-1-DESCRIBE-A-END___'''

        # Generate mask zeroing-out padded positions in encoder inputs  在编码器输入中生成掩码置零填充位置
        aa = self.dictionary.pad_idx   # 找到padding的id ，这里为0
        #  src_mask(size) = src_tokens(size) = [batch_size,src_time_steps] 10*11
        src_mask = src_tokens.eq(aa) # src_tokens 是储存着id的tensor，在这里将tensor的值变为布尔型，其中原本为0的地方的值为true


        return {'src_embeddings': _src_embeddings.transpose(0, 1),
                'src_out': (lstm_output, final_hidden_states, final_cell_states),
                'src_mask': src_mask if src_mask.any() else None}


class AttentionLayer(nn.Module):
    """ Defines the attention layer class. Uses Luong's global attention with the general scoring function. """
    def __init__(self, input_dims, output_dims):
        super().__init__()
        # Scoring method is 'general'
        self.src_projection = nn.Linear(input_dims, output_dims, bias=False)
        self.context_plus_hidden_projection = nn.Linear(input_dims + output_dims, output_dims, bias=False)

    def forward(self, tgt_input, encoder_out, src_mask):
        # tgt_input has shape = [batch_size, decoder_input_dims]   # 在这里 input_dimension 和 output_dimension 都是128
        # encoder_out has shape = [batch_size, src_time_steps, output_dims]   # 每一个t都会有一个输出 encoder_out 就是 lstm_out
        # src_mask has shape = [batch_size,src_time_steps]   10*11

        # Get attention scores
        # [batch_size, src_time_steps, output_dims]   # encoder_out_size = encoder_out.size()
        encoder_out = encoder_out.transpose(1, 0)


        # [batch_size, 1, src_time_steps]
        attn_scores = self.score(tgt_input, encoder_out)

        '''
        ___QUESTION-1-DESCRIBE-B-START___
        Describe how the attention context vector is calculated. Why do we need to apply a mask to the attention scores?
        '''
        if src_mask is not None: #如果不为空
            src_mask = src_mask.unsqueeze(dim=1) # 对数据维度进行填充 原本（n,）->(n,1)  [batch_size,1,src_time_steps] 10*1*11
            attn_scores.masked_fill_(src_mask, float('-inf'))
            # Fills elements of self tensor with value float('-inf') where mask src_mask is True (padding)

        attn_weights = F.softmax(attn_scores, dim=-1)   #  [batch_size,1,src_time_steps
        attn_context = torch.bmm(attn_weights, encoder_out).squeeze(dim=1)  #  [batch_size,output_dims]  就是 C
        context_plus_hidden = torch.cat([tgt_input, attn_context], dim=1)   #  [batch_size,encoder_output_dims+decoder_inpur_dims]
        attn_out = torch.tanh(self.context_plus_hidden_projection(context_plus_hidden))
        '''___QUESTION-1-DESCRIBE-B-END___'''

        return attn_out, attn_weights.squeeze(dim=1)
        # attn_out 是输入到下一个lstm model的hidden state，attn_weights是对于encoder每个hidden state的probability

    def score(self, tgt_input, encoder_out):
        """ Computes attention scores. """

        '''
        ___QUESTION-1-DESCRIBE-C-START___
        How are attention scores calculated? What role does matrix multiplication (i.e. torch.bmm()) play 
        in aligning encoder and decoder representations?
        '''

        # self.src_projection = nn.Linear(input_dims, output_dims, bias=False)   y = xw^T+b
        # encoder_out  = [batch_size, src_time_steps, output_dims]
        # tgt_input has shape = [batch_size, input_dims]
        sizev =  tgt_input.size()
        # projected_encoder_out_size = [batch_size, src_time_steps, output_dims]
        projected_encoder_out = self.src_projection(encoder_out)
        projected_encoder_out = projected_encoder_out.transpose(2, 1)  # [batch_size, output_dims, src_time_steps]
        tgt_input = tgt_input.unsqueeze(dim=1)   # [batch_size, 1, input_dims]
        attn_scores = torch.bmm(tgt_input, projected_encoder_out)   #  [batch_size, 1, src_time_steps]
        '''___QUESTION-1-DESCRIBE-C-END___'''

        return attn_scores


class LSTMDecoder(Seq2SeqDecoder):
    """ Defines the decoder class. """

    def __init__(self,
                 dictionary,
                 embed_dim=64,
                 hidden_size=128,   # 64*2
                 num_layers=1,
                 dropout_in=0.25,
                 dropout_out=0.25,
                 pretrained_embedding=None,
                 use_attention=True,
                 use_lexical_model=False):

        super().__init__(dictionary)

        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)

        # Define decoder layers and modules
        self.attention = AttentionLayer(hidden_size, hidden_size) if use_attention else None

        # Si= LSTMdecoder(Si-1,hi-1)
        self.layers = nn.ModuleList([nn.LSTMCell(input_size=hidden_size + embed_dim if layer == 0 else hidden_size,hidden_size=hidden_size)
            for layer in range(num_layers)])

        self.final_projection = nn.Linear(hidden_size, len(dictionary))

        self.use_lexical_model = use_lexical_model

        if self.use_lexical_model:
            # __QUESTION-5: Add parts of decoder architecture corresponding to the LEXICAL MODEL here
            # TODO: --------------------------------------------------------------------- CUT
              self.feed_forward_layer_1 = nn.Linear(embed_dim, embed_dim, bias=False)
              self.feed_forward_layer_2 = nn.Linear(embed_dim, len(dictionary))
            # TODO: --------------------------------------------------------------------- /CUT

    def forward(self, tgt_inputs, encoder_out, incremental_state=None):
        """ Performs the forward pass through the instantiated model. """  #实例化模块前向传播
        # Optionally, feed decoder input token-by-token

        # encoder_out 是个字典，包含encoder的输入src_embeddings，encoder的mask src_mask 和 encoder的输出  src_out, src_hidden_states, src_cell_states


        if incremental_state is not None:
            tgt_inputs = tgt_inputs[:, -1:]   #  最后一个time_steps 的值

        # __QUESTION-5 : Following code is to assist with the LEXICAL MODEL implementation
        # Recover encoder input
        src_embeddings = encoder_out['src_embeddings']
        src_out, src_hidden_states, src_cell_states = encoder_out['src_out']
        src_mask = encoder_out['src_mask']
        src_time_steps = src_out.size(0)

        # src_hidden_states 和 src_cell_states   (num_layers, batch, hidden_size*2)
        # src_out = lstm_out  [src_time_steps, batch_size, output_dims]

        # Embed target tokens and apply dropout
        batch_size, tgt_time_steps = tgt_inputs.size()
        tgt_embeddings = self.embedding(tgt_inputs)
        tgt_embeddings = F.dropout(tgt_embeddings, p=self.dropout_in, training=self.training)

        # Transpose batch: [batch_size, tgt_time_steps, num_features] -> [tgt_time_steps, batch_size, num_features]
        tgt_embeddings = tgt_embeddings.transpose(0, 1)

        # Initialize previous states (or retrieve from cache during incremental generation)
        '''
        ___QUESTION-1-DESCRIBE-D-START___
        Describe how the decoder state is initialized. When is cached_state缓存状态 == None? What role does input_feed play?
        '''
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')   # 递增的状态
        if cached_state is not None:
            tgt_hidden_states, tgt_cell_states, input_feed = cached_state
        else:
            # 这三个量的形状都是 batch_size * self.hidden_size
            tgt_hidden_states = [torch.zeros(tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            tgt_cell_states = [torch.zeros(tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            input_feed = tgt_embeddings.data.new(batch_size, self.hidden_size).zero_()
            input_feed_size = input_feed.size()
            # 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容。
            # zero_() 用0 填充该tensor
        '''___QUESTION-1-DESCRIBE-D-END___'''

        # Initialize attention output node
        attn_weights = tgt_embeddings.data.new(batch_size, tgt_time_steps, src_time_steps).zero_()
        rnn_outputs = []

        # __QUESTION-5 : Following code is to assist with the LEXICAL MODEL implementation
        # Cache lexical context vectors per translation time-step
        lexical_contexts = []

        for j in range(tgt_time_steps):   # 处理每个批次中，j时刻的值
            # Concatenate the current token embedding with output from previous time step (i.e. 'input feeding')
            lstm_input = torch.cat([tgt_embeddings[j, :, :], input_feed], dim=1)

            for layer_id, rnn_layer in enumerate(self.layers):    # 把 self.layers 这个列表中的layers一个个取出来，with id and instance
                # Pass target input through the recurrent layer(s)
                # tgt_hidden_states[layer_id] == 等价于 tgt_hidden_states[layer_id,:]
                tgt_hidden_states[layer_id], tgt_cell_states[layer_id] = \
                    rnn_layer(lstm_input, (tgt_hidden_states[layer_id], tgt_cell_states[layer_id]))

                # Current hidden state becomes input to the subsequent layer if we several layers for one lstm block
                # ; apply dropout
                lstm_input = F.dropout(tgt_hidden_states[layer_id], p=self.dropout_out, training=self.training)

            '''
            ___QUESTION-1-DESCRIBE-E-START___
            How is attention integrated into the decoder? Why is the attention function given the previous 
            target state as one of its inputs? What is the purpose of the dropout layer?
            '''
            if self.attention is None:
                input_feed = tgt_hidden_states[-1]
            else:
                input_feed, step_attn_weights = self.attention(tgt_hidden_states[-1], src_out, src_mask)
                attn_weights[:, j, :] = step_attn_weights

                if self.use_lexical_model:
                    # __QUESTION-5: Compute and collect LEXICAL MODEL context vectors here
                   # [src_time_steps, batch_size,  input_size]
                    # TODO: --------------------------------------------------------------------- CUT
                    step_attn_weights = torch.unsqueeze(step_attn_weights, 1) #[batch_size, 1, src_time_steps]
                    #[src_time_steps, batch_size, input_size]  -> [batch_size, src_time_steps, input_size]
                    _src_embeddings_size = src_embeddings.transpose(0, 1)

                    lexical_context = torch.bmm(step_attn_weights,_src_embeddings_size) # [batch_size, 1, input_size]  10*1*64
                    lexical_context = torch.tanh(lexical_context)     #f^l_t
                    lexical_contexts.append(lexical_context)
                    # TODO: --------------------------------------------------------------------- /CUT

            input_feed = F.dropout(input_feed, p=self.dropout_out, training=self.training)
            rnn_outputs.append(input_feed)
            '''___QUESTION-1-DESCRIBE-E-END___'''

        # Cache previous states (only used during incremental, auto-regressive generation)  # 仅在增量、自回归生成过程中使用
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', (tgt_hidden_states, tgt_cell_states, input_feed))

        # Collect outputs across time steps
        decoder_output = torch.cat(rnn_outputs, dim=0).view(tgt_time_steps, batch_size, self.hidden_size)

        # Transpose batch back: [tgt_time_steps, batch_size, num_features] -> [batch_size, tgt_time_steps, num_features]
        decoder_output = decoder_output.transpose(0, 1)  # [10,11,128]


        # Final projection
        decoder_output = self.final_projection(decoder_output)   # [10,11,4420]
        # 11*4420 代表每一个time_step的输出，所对应的score

        if self.use_lexical_model:
            # __QUESTION-5: Incorporate the LEXICAL MODEL into the prediction of target tokens here
            # TODO: --------------------------------------------------------------------- CUT

            weighted_embeddings = torch.cat(lexical_contexts, 1)   # [batch_size,tgt_time_steps,input_size(embed dimension)]   10#11*64
            weighted_embeddings_size = weighted_embeddings.size()
            lexical_hidden = torch.tanh(self.feed_forward_layer_1(weighted_embeddings)) + weighted_embeddings
            decoder_output =decoder_output+ self.feed_forward_layer_2(lexical_hidden)
            # TODO: --------------------------------------------------------------------- /CUT

        return decoder_output, attn_weights


@register_model_architecture('lstm', 'lstm')
def base_architecture(args):
    # getattr(对象，参数，若无则返回默认值) 函数用于返回一个对象属性值。

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 64)
    args.encoder_num_layers = getattr(args, 'encoder_num_layers', 2)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', 'True')
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.25)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.25)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 128)
    args.decoder_num_layers = getattr(args, 'decoder_num_layers', 2)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.25)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.25)
    args.decoder_use_attention = getattr(args, 'decoder_use_attention', 'True')
    args.decoder_use_lexical_model = getattr(args, 'decoder_use_lexical_model', 'False')
