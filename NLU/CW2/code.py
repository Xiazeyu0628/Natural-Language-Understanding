def forward(self,
            query,
            key,
            value,
            key_padding_mask=None,
            attn_mask=None,
            need_weights=True):
    # Get size features
    tgt_time_steps, batch_size, embed_dim = query.size()
    key_size = key.size()  # has same size tgt_time_steps*batch_size*embed_dim
    value_size = value.size()
    assert self.embed_dim == embed_dim
    '''
    ___QUESTION-7-MULTIHEAD-ATTENTION-START
    Implement Multi-Head attention  according to Section 3.2.2 of https://arxiv.org/pdf/1706.03762.pdf.
    Note that you will have to handle edge cases for best model performance. Consider what behaviour should
    be expected if attn_mask or key_padding_mask are given?
    '''
    # TODO: REPLACE THESE LINES WITH YOUR IMPLEMENTATION ------------------------ CUT
    # the key and value may not have same third dimensionality with query especially in encoder-attention in decoder
    # the third dimensionality for key and value is source time step, while for query is tgt_time_steps
    # in self-attention, the key，query，value have same  first dimensionality.
    k = self.k_proj(key).contiguous().view(-1, batch_size, self.num_heads, self.head_embed_size)
    q = self.q_proj(query).contiguous().view(tgt_time_steps, batch_size, self.num_heads, self.head_embed_size)
    v = self.v_proj(value).contiguous().view(-1, batch_size, self.num_heads, self.head_embed_size)
    # linear projection and divide them into num_heads chunks
    k = k.transpose(0, 2).contiguous().view(self.num_heads * batch_size, -1, self.head_embed_size)
    q = q.transpose(0, 2).contiguous().view(self.num_heads * batch_size, tgt_time_steps, self.head_embed_size)
    v = v.transpose(0, 2).contiguous().view(self.num_heads * batch_size, -1, self.head_embed_size)

    attn_weights = torch.bmm(q, k.transpose(1, 2)) / self.head_scaling  # scaling the dimension

    if key_padding_mask is not None:
        # the size of key_padding_mask is [1, batch_size, 1, src_time_steps]
        key_padding_mask = key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=0)

        # the size of key_padding_mask now is [num_heads, batch_size, tgt_time_steps, src_time_steps]
        key_padding_mask_ = key_padding_mask.repeat(self.num_heads, 1, tgt_time_steps, 1)

        # change the size of attn_weights to [num_heads, batch_size, tgt_time_steps, src_time_steps]
        attn_weights = attn_weights.contiguous().view(self.num_heads, batch_size, tgt_time_steps, -1)

        # we have the same dimensionaility for key_padding_mask_ and attn_weights, now we can add the mask
        attn_weights.masked_fill_(key_padding_mask_ == True, float(-1e10))

        # reshape
        attn_weights = attn_weights.contiguous().view(self.num_heads * batch_size, tgt_time_steps, -1)
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(dim=0)  # [1，tgt_time_steps, src_time_steps]
        # now the attn_mask is of dimension [self.num_heads * batch_size, tgt_time_steps, src_time_steps]
        attn_mask_ = attn_mask.repeat(self.num_heads * batch_size, 1, 1)
        # add the mask
        attn_weights.masked_fill_(attn_mask == float("-inf"), float(-1e10))

    attn_weights = F.softmax(attn_weights, dim=-1)
    # F.dropout(attn_weights,p=self.attention_dropout,training=self.training)
    attn = torch.bmm(attn_weights, v)
    attn = attn.contiguous().view(self.num_heads, batch_size, tgt_time_steps, self.head_embed_size)
    attn = attn.transpose(0, 2)
    attn = attn.contiguous().view(tgt_time_steps, batch_size, self.num_heads * self.head_embed_size)
    attn_weights = attn_weights.contiguous().view(self.num_heads, batch_size, tgt_time_steps,-1) if need_weights else None
    # TODO: --------------------------------------------------------------------- CUT
    '''
    ___QUESTION-7-MULTIHEAD-ATTENTION-END
    '''
    return attn, attn_weights