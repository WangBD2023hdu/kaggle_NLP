import math
import torch.nn as nn
import torch
from model.text_encoding import TextEncoder
# import utils.gat as tg_conv
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
        self.linear1 = nn.Linear(4, 2)
        # for np compute the importance of each np
        self.linear2 = nn.Linear(150, 1)
        self.norm = nn.LayerNorm(150)
        self.relu1 = nn.Tanh()
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=self.txt_out_size, hidden_size=150, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(self.txt_gat_drop)
    def forward(self, encoded_cap, word_spans, mask_batch1, edge_cap1):
        text, score = self.txt_encoder(encoded_cap, word_spans, mask_batch1)

        out, (h_n, c_n) = self.lstm(text)
        h_n = h_n.permute(1, 0, 2)
        h_n = h_n.squeeze()
        # h_n = self.norm(h_n)
        y = self.dropout(self.tanh(self.linear2(h_n)))

        y = y.squeeze()
        y = F.softmax(self.linear1(y), dim=1)
        # # for gat in self.gclayer:
        # #     text = self.norm(torch.stack(
        # #         [(self.relu1(gat(data[0], data[1].cuda()))) for data in zip(text, edge_cap1)]))
        # text = self.norm(self.linear1(text))
        # text = text.squeeze()
        # text = self.norm(self.linear2(text))

        return y
