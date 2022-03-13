import torch
from torch.utils.data import Dataset
import csv
import gzip
import xml.dom.minidom
import os

def load_ENdata(path):
    train_samples,train_label = [],[]
    dev_samples,dev_label = [],[]
    test_samples,test_label = [],[]
    with gzip.open(path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'dev':
                score = float(row['score']) / 5.0
                dev_samples.extend([row['sentence1'], row['sentence2']])
                dev_label.extend([score, score])
            elif row['split'] == 'test':
                score = float(row['score']) / 5.0
                test_samples.extend([row['sentence1'], row['sentence2']])
                test_label.extend([score, score])
            else:
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                train_samples.extend([row['sentence1'], row['sentence2']])
                train_label.extend([score, score])


    return train_label,train_samples,dev_label,dev_samples,test_label,test_samples

def load_PTdata(path):
    train_samples, train_label = [], []
    dev_samples, dev_label = [], []
    test_samples, test_label = [], []

    for split in ['train', 'validation', 'test']:
        if split == 'validation':
            dom = xml.dom.minidom.parse(os.path.join(path, 'assin2-dev.xml'))
            root = dom.documentElement
            pairs = root.getElementsByTagName('pair')
            for pair in pairs:
                score = float(pair.getAttribute('similarity')) / 5.0  # Normalize score to range 0 ... 1
                dev_samples.extend([pair.getElementsByTagName('t')[0].childNodes[0].data,
                                      pair.getElementsByTagName('h')[0].childNodes[0].data])
                dev_label.extend([score, score])
                # inp_example = InputExample(texts=[pair.getElementsByTagName('t')[0].childNodes[0].data,
                #                                   pair.getElementsByTagName('h')[0].childNodes[0].data], label=label)
                # dev_samples.append(inp_example)
        elif split == 'test':
            dom = xml.dom.minidom.parse(os.path.join(path, 'assin2-test.xml'))
            root = dom.documentElement
            pairs = root.getElementsByTagName('pair')
            for pair in pairs:
                score = float(pair.getAttribute('similarity')) / 5.0  # Normalize score to range 0 ... 1
                test_samples.extend([pair.getElementsByTagName('t')[0].childNodes[0].data,
                                      pair.getElementsByTagName('h')[0].childNodes[0].data])
                test_label.extend([score, score])
                # inp_example = InputExample(texts=[pair.getElementsByTagName('t')[0].childNodes[0].data,
                #                                   pair.getElementsByTagName('h')[0].childNodes[0].data], label=label)
                # test_samples.append(inp_example)

        elif split == 'train':
            dom = xml.dom.minidom.parse(os.path.join(path, 'assin2-train-only.xml'))
            root = dom.documentElement
            pairs = root.getElementsByTagName('pair')
            for pair in pairs:
                # if float(pair.getAttribute('similarity'))>=4.0:
                #     inp_example1 = InputExample(texts=[pair.getElementsByTagName('t')[0].childNodes[0].data, pair.getElementsByTagName('h')[0].childNodes[0].data])
                #     train_samples1.append(inp_example1)
                # else:
                score = float(pair.getAttribute('similarity')) / 5.0  # Normalize score to range 0 ... 1
                train_samples.extend([pair.getElementsByTagName('t')[0].childNodes[0].data,pair.getElementsByTagName('h')[0].childNodes[0].data])
                train_label.extend([score, score])

    return train_label, train_samples, dev_label, dev_samples, test_label, test_samples


class CustomDataset(Dataset):
    def __init__(self, sentence, label, tokenizer):
        self.sentence = sentence
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            text=self.sentence[index],
            add_special_tokens=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': self.label[index]
        }

def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids

def collate_fn(batch):
    # 按batch进行padding获取当前batch中最大长度
    max_len = 128#max([len(d['input_ids']) for d in batch])

    # 定一个全局的max_len
    # max_len = 128

    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    for item in batch:
        input_ids.append(pad_to_maxlen(item['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(item['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(item['token_type_ids'], max_len=max_len))
        labels.append(item['label'])

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    all_label_ids = torch.tensor(labels, dtype=torch.float)
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids