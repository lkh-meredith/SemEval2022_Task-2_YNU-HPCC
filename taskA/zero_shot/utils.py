import re
import os
import sys
import pandas as pd
import numpy as np
import random
import torch


def set_seed(seed: int):
    """
    Modified from : https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_utils.py
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenise_idiom( phrase ) :
    # word=re.sub("[?]",'',phrase)
    s='ID' + re.sub( r'[\s|-]', '', phrase).lower() + 'ID'
    return s

def get_devGold(dev_location):
    df_label=pd.read_csv(dev_location)
    sim=list()
    for i in range(len(df_label)):
        sim.append(float(df_label['sim'][i]))
    return sim

def prepare_data(location,type) :
    data = pd.read_csv(location,encoding='ISO-8859-1')
    sentence1s = list()
    labels = list()
    if type=='train' or type=='dev':
        for index in range(len(data)) :
            if data['sentence1'][index] is not None:
                sentence1 = str(data['sentence1'][index])
                label=data['label'][index]
                labels.append(label)
                sentence1s.append(sentence1)

        return sentence1s,labels

    elif type=='eval' or type=='test':
        labels=None
        for index in range(len(data)):
            if data['sentence1'][index] is not None:
                sentence1 = str(data['sentence1'][index])
                sentence1s.append(sentence1)
                # if mwe != 'None':
                #     replaced = sentence1.replace(mwe, tokenise_idiom(mwe))  # ,flags=re.I
                #     assert replaced != sentence1
                #     sentence1 = replaced
                # replaced_sentence1.append(sentence1)
        return sentence1s,labels
