import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import math
from utils import L2_norm, cosine_distance
from transformers import BertModel
from utils.data_utils import pad_tensor

class TextEncoder(nn.Module):
    r"""Initializes a NLP embedding block.
     :param input_size:
     :param nhead:
     :param dim_feedforward:
     :param dropout:
     :param activation:
     参数没有初始化
     """

    def __init__(self, input_size=768, out_size = 300):
        super(TextEncoder, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.norm = nn.LayerNorm(self.out_size)
        self.linear = nn.Linear(self.out_size, 1)
        self.linear_ = nn.Linear(self.input_size, self.out_size)

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        # self.lstm = nn.LSTM()

    def forward(self, t1, word_seq, key_padding_mask, lam=1):
        """
        Function to compute forward pass of the ImageEncoder TextEncoder
        Args:
            t1: (N,L,D) Padded Tensor. L is the length. D is dimension after bert
            token_length: (N) list of length
            word_seq:(N,tensor) list of tensor
            key_padding_mask: (N,L1) Tensor. L1 is the np length. True means mask
        Returns:
            t1: (N,L1,D). The embedding of each word or np. D is dimension after bert.
            score: (N,L1,D). The importance of each word or np. For convenience, expand the tensor (N,L1，D) to compute
            the caption embedding.
        """
        # (batch_size, sequence_length, hidden_size)
        t1 = self.bert_model(**t1)[0]
        # (batch_size, hidden_size) may averaging or pooling
        cls_token = t1[:,0,:]
        # concatenate to np and words
        t1 = t1[:, 1:-1, :]
        captions = []
        for i in range(t1.size(0)):
            # [X,L,H] X is the number of np and word
            captions.append(torch.stack([torch.mean(t1[i][tup[0]:tup[1], :], dim=0) for tup in word_seq[i]]))

        # (N,L,D)
        t1 = pad_sequence(captions, batch_first=True).cuda()
        t1 = self.norm(self.linear_(t1))
        # get each word importance
        # (N,L)
        # score = self.linear(t1).squeeze()
        score = self.linear(t1).squeeze().masked_fill_(key_padding_mask, float("-Inf"))
        # N,L,L distance (0,2)
        # (N, L, D)
        score = nn.Softmax(dim=1)(score*lam).unsqueeze(2).repeat((1, 1, self.out_size))

        return t1, score
