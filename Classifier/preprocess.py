import os
import re
import pickle
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from omegaconf import OmegaConf

def normalize(document):
    document = list(document)
    for j in range(len(document)):
        if document[j] in '0123456789':
            document[j] = '*'
    return ('').join(document)

def getname(filepath):
    return filepath.split('/')[-1].split('.')[0]

def main(cfg):
    train_csv = pd.read_csv(cfg.csv_path)
    train_csv['document'] = train_csv['document'].apply(normalize)

    labels = np.array(list(train_csv['class']))

    char_to_id = {}
    id_to_char = {}
    len_char = cfg.char.in_len
    targetchars = " abcdefghijklmnopqrstuvwxyz-\'!%&()*,./:;?@[\]`{|}+<=>^û#$^~\""
    bigchars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    char_data = np.zeros([len(train_csv), len(targetchars)+2, len_char])
    document = bigchars
    for index, character in enumerate(targetchars):
        char_to_id[character] = index + 1
        id_to_char[index + 1] = character
    for j in range(len(train_csv)):
        document = train_csv['document'][j]
        for i in range(min(len_char, len(document))):
            thischar = document[i]
            if thischar in bigchars:
                char_data[j, 0, i] = 1
                char_data[j, char_to_id[thischar.lower()], i] = 1
            elif thischar in targetchars:
                char_data[j, char_to_id[thischar], i] = 1
            else:
                char_data[j, -1] = 1

    data_dict = {}
    data_dict['chat_to_idx'] = char_to_id
    data_dict['idx_to_char'] = id_to_char
    data_dict['char_data'] = char_data
    data_dict['labels'] = labels

    with open(f'data/char-{getname(cfg.csv_path)}-{len_char}.pkl', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sentences = []
    freq_dict = {}
    for script in train_csv['document']:
        sentence = re.sub("[^A-Za-z\*\$\-]+", ' ', script).lower().split()
        sentences.append(sentence)
        for word in set(sentence):
            if word in freq_dict:
                freq_dict[word] += 1
            else:
                freq_dict[word] = 1

    freq = {}
    for word in freq_dict:
        n = freq_dict[word]
        if n in freq:
            freq[n]+=1
        else:
            freq[n]=1
    del freq[1]
    del freq[2]

    x,y = [math.log(t[0], 2) for t in freq.items()], [math.log(t[1],2) for t in freq.items()]
    plt.scatter(x, y)
    plt.savefig(os.path.join(cfg.token.img_dir, 'token_distribution'))

    words = set()
    for word in freq_dict:
        if freq_dict[word]>=100:
            words.add(word)
    sentences_sig = []
    for sentence in sentences:
        sentences_sig.append([word for word in sentence if word in words])
    w2v = Word2Vec(min_count=1, vector_size = 32, negative=20)
    w2v.build_vocab(sentences_sig)
    w2v.train(sentences_sig, total_examples=w2v.corpus_count, epochs=cfg.token.w2v_epochs)
    with open(os.path.join(cfg.token.w2v_dir, f'w2v-{getname(cfg.csv_path)}-{cfg.token.w2v_epochs}.pkl'), 'wb') as handle:
        pickle.dump(w2v, handle, protocol=pickle.HIGHEST_PROTOCOL)

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences_sig)]
    d2v = Doc2Vec(min_count=1, vector_size = 32, negative=20)
    d2v.build_vocab(documents)
    d2v.train(documents, total_examples=d2v.corpus_count, epochs=cfg.token.w2v_epochs)
    with open(os.path.join(cfg.token.w2v_dir, f'd2v-{getname(cfg.csv_path)}-{cfg.token.w2v_epochs}.pkl'), 'wb') as handle:
        pickle.dump(d2v, handle, protocol=pickle.HIGHEST_PROTOCOL)

    token_len = cfg.token.in_len
    token_data = np.zeros([len(sentences_sig), token_len])
    for idx, sentence in enumerate(sentences_sig):
        s = [w2v.wv.key_to_index[word] for word in sentence][:token_len]
        token_data[idx, :len(s)] = np.array(s)
    data_dict = {}
    data_dict['token_to_idx'] = w2v.wv.key_to_index
    data_dict['idx_to_token'] = w2v.wv.index_to_key
    data_dict['token_data'] = token_data
    data_dict['labels'] = labels
    with open(f'data/token-{getname(cfg.csv_path)}-{cfg.token.in_len}.pkl', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    cfg = OmegaConf.load('config/preprocess.yaml')
    main(cfg)