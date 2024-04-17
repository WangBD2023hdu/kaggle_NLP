import math
import torch.nn as nn
import torch
from model.text_encoding import TextEncoder
import utils.gat as tg_conv
import torch.nn.functional as F
from model.CMGCN import GraphConvolution


class GCNClassifer(nn.Module):
    """
    Our model for Image Repurpose Task
    """

    def __init__(self, txt_input_size=768, txt_out_size=300,
                 txt_gat_layer=2, txt_gat_drop=0.2, txt_gat_head=5, max_length=100):
        super(GCNClassifer, self).__init__()
        self.txt_input_size = txt_input_size
        self.txt_out_size = txt_out_size

        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop = txt_gat_drop
        self.txt_gat_head = txt_gat_head
        self.max_length = max_length

        self.txt_encoder = TextEncoder(input_size=self.txt_input_size, out_size=self.txt_out_size)
        """
        Args:
            input_size: 输入的维度
            txt_gat_layer: GCN层数
            txt_gat_drop: # Disable
            txt_gat_head: # Disable
        """ 

        self.gclayer = nn.ModuleList([GraphConvolution(self.txt_input_size, self.txt_input_size) for i in range(self.txt_gat_layer)])

        # for token compute the importance of each token
        self.linear1 = nn.Linear(200, 1)
        # for np compute the importance of each np
        self.linear2 = nn.Linear(self.max_length, 2)
        self.norm = nn.LayerNorm(self.txt_input_size)
        self.relu1 = nn.ReLU()
        self.lstm = nn.LSTM(input_size=200, hidden_size=300, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, encoded_cap, word_spans, mask_batch1, edge_cap1):
        text, score = self.txt_encoder(encoded_cap, word_spans, mask_batch1)

        out, (_, _) = self.lstm(text)
        print(out.size())
        # for gat in self.gclayer:
        #     text = self.norm(torch.stack(
        #         [(self.relu1(gat(data[0], data[1].cuda()))) for data in zip(text, edge_cap1)]))
        text = self.norm(self.linear1(text))
        text = text.squeeze()
        text = self.norm(self.linear2(text))

        return text
