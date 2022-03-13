import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append("D:/Meredith/TaskA")
import csv
import math
from torch.utils.data                 import DataLoader
from sklearn.metrics.pairwise         import paired_cosine_distances
# from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.readers import InputExample
from transformers import AutoTokenizer,AutoModel
import numpy as np
from typing import Union, Tuple, List, Iterable, Dict, Callable
from utils import prepare_data,set_seed
from Utils.utils import evaluate_submission,insert_to_submission_file,write_csv
from torch import nn,Tensor
import torch
import pandas as pd
import argparse

def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def collate_fn(batch):
    # input_ids, attention_mask, token_type_ids = [], [], []
    sentence_feature=[]
    for index in range(2):
        inputs = []
        attentions = []
        tokens = []
        for i in range(len(batch)):
            input = tokenizer.encode_plus(
                text=batch[i][index],
                add_special_tokens=True,
                return_token_type_ids=True,
                max_length=128,
                truncation=True
            )

            inputs.append(pad_to_maxlen(input['input_ids'],max_len=128))
            attentions.append(pad_to_maxlen(input['attention_mask'],max_len=128))
            tokens.append(pad_to_maxlen(input["token_type_ids"],max_len=128))

        dic={'input_ids':torch.tensor(inputs).cuda(),'attention_mask': torch.tensor(attentions).cuda(),'token_type_ids': torch.tensor(tokens).cuda()}
        sentence_feature.append(dic)

    # all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    # all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    # all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)

    return sentence_feature
    # return {
    #         'input_ids': input_ids,
    #         'attention_mask': attention_mask,
    #         'token_type_ids': token_type_ids
    #     }

class Loss(nn.Module):
    """
    This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
    model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?
    :param loss_fct: Optional: Custom pytorch loss function. If not set, uses nn.CrossEntropyLoss()

    """
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = False,
                 concatenation_sent_multiplication: bool = False,
                 loss_fct: Callable = nn.CrossEntropyLoss()):
        super(Loss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        # logger.info("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)
        features = self.dropout(features)
        output = self.classifier(features)

        if labels is not None:#train
            loss = self.loss_fct(output, labels)#torch.squeeze(output).to(torch.float32)
            return loss
        else:#dev、eval
            probs = torch.softmax(output, dim=-1)
            y_pred = torch.argmax(probs, dim=-1)
            # loss = self.loss_fct(output, labels)#torch.squeeze(output).to(torch.float32)
            return y_pred

def evaluate(loss,dataloader):
    y_preds = []
    for _,data in enumerate(dataloader):
        y_pred = loss(data,labels=None)
        y_preds += y_pred.tolist()
    return y_preds


if __name__=="__main__":
    model_path = '../model/bert-base-multilingual-cased'
    parser=argparse.ArgumentParser()
    parser.add_argument("--epoches",default=1,type=int)
    parser.add_argument("--batch_size",default=32,type=int)
    args=parser.parse_args()
    train_batch_size = args.batch_size
    num_epochs = args.epoches
    set_seed(4)

    # 加载数据集
    dataset_path = os.path.join('..', 'data', 'OneShot')
    train_path = os.path.join(dataset_path, 'train.csv')
    dev_path=os.path.join(dataset_path,'dev.csv')
    eval_path=os.path.join(dataset_path,'eval.csv')

    train_data,train_MWEs,train_label = prepare_data(train_path,'train')
    dev_data,dev_MWEs,dev_label=prepare_data(dev_path,'dev')
    eval_data,eval_MWEs,_=prepare_data(eval_path,'eval')
    model_save_path=os.path.join('..', 'model', 'OneShot','SentTransModel')

    train_samples,dev_samples,eval_samples=list(),list(),list()
    for index in range(len(train_data)):
        inp_example = InputExample(texts=[train_data[index], train_MWEs[index]], label=train_label[index])
        train_samples.append(inp_example)
    for index in range(len(dev_data)):
        inp_example =[dev_data[index], dev_MWEs[index]]
        dev_samples.append(inp_example)
    for index in range(len(eval_data)):
        inp_example =[eval_data[index], eval_MWEs[index]]
        eval_samples.append(inp_example)

    sentencemodel_save_path = '../model/OneShot/SentTransModel/bert-base-multilingual-cased_' + str(train_batch_size) + '_' + str(num_epochs)

    word_embedding_model = models.Transformer(model_path)  # ,cache_dir=Nonemax_seq_length=256,'bert-base-multilingual-cased'
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        max_length=128,
        force_download=True
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model._first_module().tokenizer = tokenizer

    # model.save(sentencemodel_save_path)

    train_dataloader = DataLoader(dataset=train_samples, shuffle=True, batch_size=train_batch_size)
    dev_dataloader=DataLoader(dataset=dev_samples,shuffle=False,batch_size=16,collate_fn=collate_fn)
    eval_dataloader=DataLoader(dataset=eval_samples,shuffle=False,batch_size=16,collate_fn=collate_fn)

    loss = Loss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)

    # Configure the training.
    warmup_steps = math.ceil((len(train_dataloader)) * num_epochs * 0.1)  # 10% of train data for warm-up
    print("Warmup-steps: {}".format(warmup_steps), flush=True)

    # Train the model
    model.fit(train_objectives=[(train_dataloader,loss)],
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=sentencemodel_save_path
              )

    model.eval()
    dev_pred = evaluate(loss,dev_dataloader)
    pred = pd.DataFrame({'prediction': dev_pred})
    dev_prediction_format_file = '../data/OneShot/dev_predict.csv'
    pred.to_csv(dev_prediction_format_file, index=False)
    dev_submission_format_file = '../data/dev_submission_format.csv'
    submission_data = insert_to_submission_file(dev_submission_format_file, dev_path, dev_prediction_format_file,
                                                'one_shot')
    results_file = os.path.join('..', 'submission', 'OneShot','SentTransModel',
                                'dev.combined_results-' + str(num_epochs) + '_' + str(train_batch_size) + '.csv')
    # submission_data.to_csv(results_file,index=False)
    write_csv(submission_data, results_file)

    ## Evaluate development set.
    dev_gold = '../SubTaskA_data/Data/dev_gold.csv'
    results = evaluate_submission(results_file, dev_gold)
    ## Make results printable.
    for result in results:
        print(result)

    results_file = os.path.join(model_save_path,
                                'RESULTS_TABLE-dev_oneshot-' + str(num_epochs) + '_' + str(train_batch_size) + '.csv')
    write_csv(results, results_file)


    eval_preds = evaluate(loss,eval_dataloader)
    eval_pred = pd.DataFrame({'prediction': eval_preds})
    eval_prediction_format_file = '../data/OneShot/eval_predict.csv'
    eval_pred.to_csv(eval_prediction_format_file, index=False)
    eval_submission_format_file = '../data/eval_submission_format.csv'
    submission_data = insert_to_submission_file(eval_submission_format_file, eval_path, eval_prediction_format_file,
                                                'one_shot')
    results_file = os.path.join('..', 'submission', 'OneShot','SentTransModel',
                                'eval.combined_results-' + str(num_epochs) + '_' + str(train_batch_size) + '.csv')
    write_csv(submission_data, results_file)
