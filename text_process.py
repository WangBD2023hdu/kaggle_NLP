import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

# 根日志器默认日志级别为WARNING，这里将其重置，以保证debug、info级别的日志也能输出
logging.basicConfig(filename="deal.log", filemode="w", level=logging.NOTSET)

logging.info("loading en_core_web_trf")
nlp = spacy.load("en_core_web_trf")
logging.info("nlp = spacy.load(en_core_web_trf) complention")


def re_organize_caption(tokens, noun_phrases):
    # tokens_smaple token+np
    # token_map token_map[a] a 是
    tokens_sample = []
    chunk_index = 0
    chunk_len = len(noun_phrases)
    i = 0
    token_map = []
    while (i < len(tokens)):
        if chunk_index < chunk_len:
            if i < noun_phrases[chunk_index][1]:
                tokens_sample.append(tokens[i][0])
                token_map.append(len(tokens_sample) - 1)
                i = i + 1
            else:
                tokens_sample.append(noun_phrases[chunk_index][0])
                for a in range(i, noun_phrases[chunk_index][2]):
                    token_map.append(len(tokens_sample) - 1)
                i = noun_phrases[chunk_index][2]
                chunk_index = chunk_index + 1
        else:
            tokens_sample.append(tokens[i][0])
            token_map.append(len(tokens_sample) - 1)
            i = i + 1
    return tokens_sample, token_map


# 按照dataloder的方法， 实现单个文本生成token_cap token_dep

def single_sample_token_dependency(text):
    dataset_article = {}
    dataset_article_a = {}
    doc = nlp(text)
    dataset_article["token_caption"] = [(token.text.lower(), token.i, token.head.i, token.is_punct) for token in
                                        doc]
    dataset_article["chunk"] = [(chunk.text.lower(), chunk.start, chunk.end) for chunk in doc.noun_chunks]
    token_sample, token_map = re_organize_caption(dataset_article["token_caption"], dataset_article["chunk"])
    dependency = [(token_map[t[1]], token_map[t[2]]) for t in dataset_article["token_caption"] if
                  (not (token_map[t[1]]) == token_map[t[2]]) and
                  (not t[3])]

    dataset_article_a["chunk_cap"] = token_sample
    # dataset_article_a["token_cap"] = [t[0] for t in dataset_article["token_caption"]]
    #
    # dataset_article_a["token_dep"] = [(t[1], t[2]) for t in dataset_article["token_caption"] if
    #                                   (not t[1] == t[2]) and (not t[3]) and t[0] != " "
    #                                   and dataset_article["token_caption"][t[2]][0] != " "]

    token_cap = [t[0] for t in dataset_article["token_caption"]]

    token_dep = [(t[1], t[2]) for t in dataset_article["token_caption"] if
                                      (not t[1] == t[2]) and (not t[3]) and t[0] != " "
                                      and dataset_article["token_caption"][t[2]][0] != " "]
    return token_cap, token_dep

