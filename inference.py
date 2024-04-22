import argparse
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import json
import re
import torch
from utils.data_utils import seed_everything
from model.Classifer import GCNClassifer
from text_process import single_sample_token_dependency
from transformers import AutoTokenizer
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 加载模型
seed_everything(42)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--mode', type=str, default='train',
                    help="mode, {'" + "train" + "',     '" + "eval" + "'}")
parser.add_argument('-p', '--path', type=str, default='saved_model path',
                    help="path, relative path to save model}")
parser.add_argument('-s', '--save', type=str, default='saved model',
                    help="path, path to saved model}")
parser.add_argument('-o', '--para', type=str, default='parameter.json',
                    help="path, path to json file keeping parameter}")
args = parser.parse_args()
with open(args.para) as f:
    parameter = json.load(f)

annotation_files = parameter['annotation_files']

def construct_mask_text(seq_len, max_length):
    """

    Args:
        seq_len1(N): list of number of words in a caption without padding in a minibatch
        max_length: the dimension one of shape of embedding of captions of a batch

    Returns:
        mask(N,max_length): Boolean Tensor
    """
    # the realistic max length of sequence
    max_len = max(seq_len)
    if max_len <= max_length:
        mask = torch.stack(
            [torch.cat([torch.zeros(len, dtype=bool), torch.ones(max_length - len, dtype=bool)]) for len in seq_len])
    else:
        mask = torch.stack(
            [torch.cat([torch.zeros(len, dtype=bool),
                        torch.ones(max_length - len, dtype=bool)]) if len <= max_length else torch.zeros(max_length,
                                                                                                         dtype=bool) for
             len in seq_len])

    return mask
model = GCNClassifer(txt_input_size=parameter["txt_input_dim"], txt_out_size=parameter["txt_out_size"],
                                txt_gat_layer=parameter["txt_gat_layer"], txt_gat_drop=parameter["txt_gat_drop"],
                                txt_gat_head=parameter["txt_gat_head"])

model.load_state_dict(torch.load("best_model/kaggle-17.pt", map_location=device), False)
model.eval()
model.to(device=device)

test = pd.read_csv(os.path.join("text", "test.csv"))
res = []
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
for index in range(test.shape[0]):
    twitters = test.iloc[index]["text"].split(" ")
    encoded_cap = tokenizer(twitters, is_split_into_words=True, return_tensors="pt")
    word_spans = []
    word_len = []
    token_lens = [1]
    for index_encode, len_token in enumerate(token_lens):
        word_span_ = []
        for i in range(len_token):
            word_span = encoded_cap[index_encode].word_to_tokens(i)
            if word_span is not None:
                # delete [CLS]
                word_span_.append([word_span[0] - 1, word_span[1] - 1])
        word_spans.append(word_span_)
        word_len.append(len(word_span_))
    max_len1 = max(word_len)
    # mask矩阵是相对于word token的  key_padding_mask for computing the importance of each word in txt_encoder and
    # interaction modules
    mask_batch1 = construct_mask_text(word_len, max_len1)
    encoded_cap = {k: v.to(device) for k, v in encoded_cap.items()}
    print(mask_batch1)
    y = model(encoded_cap=encoded_cap, word_spans=word_spans, mask_batch1=mask_batch1,
              edge_cap1=None)

    res.append([test.iloc[index]["id"], y.max(dim=0)[1]])
result = pd.DataFrame(res)
result.columns = ['id', 'target']
result.to_csv("submission.csv", index=False)





