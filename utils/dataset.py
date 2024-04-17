import torch
import pandas as pd
from torch.utils.data import Dataset
import json

class BaseSet(Dataset):
    def __init__(self, type="train", max_length=100, text_path=None):
        """
        # TODO 
        完成依存分析的集成
        Args:
            type: "train","val","test"
            max_length: the max_lenth for bert embedding
            text_path: path to annotation file
        """
        self.type = type  # train, val, test
        self.max_length = max_length
        self.text_path = text_path
        with open(self.text_path) as f:
            self.dataset = json.load(f)

    def __getitem__(self, index):
        """
        Args:
            index:

        Returns:
            text_emb: (token_len, 758). Tensor
            text_seq: (word_len). List.
            dep: List.
            word_len: Int.
            token_len: Int
            label: Int
            chunk_index: li

        """
        sample = self.dataset[index]

        # for val and test dataset, the sample[2] is hashtag label
        if self.type == "train":
            label = sample[2]
            text = sample[3]
        else:
            # label =sample[2] hashtag label
            label = sample[3]
            text = sample[4]

        twitter = text["token_cap"]
        dep = text["token_dep"]
        return twitter, dep, label

    def __len__(self):
        """
            Returns length of the dataset
        """
        return len(self.dataset)

