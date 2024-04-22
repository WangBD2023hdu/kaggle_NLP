import torch
import pandas as pd
from torch.utils.data import Dataset
import json
import random
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
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
        self.dataset = pd.read_excel(self.text_path)
        self.label_eye = torch.eye(2)

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
        if self.type == "train" or self.type == "val":
            label = int(self.dataset.iloc[index]["label"])
            label = self.label_eye[label]
            twitter = eval(self.dataset.iloc[index]['token_cap'])
            twitter = [word for word in twitter+["UNK"] if word not in stop_words]
            dep = eval(self.dataset.iloc[index]["token_dep"])
        else:
            # label =sample[2] hashtag label
            label = self.label_eye[random.randint(0, 1)]
            twitter = eval(self.dataset.iloc[index]['token_cap'])
            twitter = [word for word in twitter+['UNK'] if word not in stop_words]
            dep = eval(self.dataset.iloc[index]["token_dep"])

        return twitter, dep, label

    def __len__(self):
        """
            Returns length of the dataset
        """
        return len(self.dataset)

