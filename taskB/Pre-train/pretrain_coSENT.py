import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append("D:/Meredith/TaskB")
import csv
import gzip
import xml.dom.minidom
import math
from datetime import datetime
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import paired_cosine_distances
# from datasets import load_dataset
from transformers import AutoTokenizer,AutoModel,BertConfig,BertModel
import numpy as np

from transformers.models.bert import BertTokenizer
from transformers import AdamW,get_linear_schedule_with_warmup

# from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from typing import Union, Tuple, List, Iterable, Dict, Callable
from data_CoSENT import load_ENdata,load_PTdata,CustomDataset,collate_fn,pad_to_maxlen

from utils import evaluate,get_similarity,prepare_data,get_devSimilarity,write_csv,evaluate_submission,insert_to_submission1,writeList_csv,set_seed
from torch import nn

# from transformers import InputExample
from torch.utils.data import Dataset
import random
import torch
import argparse

#本部分来自https://github.com/shawroad/CoSENT_Pytorch
def calc_loss(y_true,y_pred):
    #取出真实标签
    y_true=y_true[::2]
    #对输出的句子向量进行l2归一化 后面只需要对应位相乘就可以得到cos值
    norms=(y_pred**2).sum(axis=1,keepdims=True)**0.5
    y_pred=y_pred/norms
    #奇偶向量相乘
    y_pred=torch.sum(y_pred[::2]*y_pred[1::2],dim=1)*20
    y_1=y_pred[:,None]
    y_2=y_pred[None,:]
    y_pred=y_1-y_2#两两之间的余弦差值
    #矩阵中的第i行第j列 表示的是第i个余弦值-第j个余弦值
    y_true=y_true[:,None]<y_true[None,:]#取出正负样例的差值
    y_true=y_true.float()

    y_pred=y_pred-(1-y_true)*1e12#排除不需要计算的部分
    y_pred=y_pred.view(-1)
    if torch.cuda.is_available():
        y_pred=torch.cat((torch.tensor([0]).float().cuda(),y_pred),dim=0)#相当于在log中+1
    else:
        y_pred=torch.cat((torch.tensor([0]).float(),y_pred),dim=0)

    return torch.logsumexp(y_pred,dim=0)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.config=BertConfig.from_pretrained('../model/model_with_MWE/bert-base-multilingual-cased/config.json')
        self.bert=BertModel.from_pretrained('../model/model_with_MWE/bert-base-multilingual-cased',config=self.config)#,config=self.config

    def forword(self,input_ids,attention_mask,encoder_type='cls'):#
        #param encoder_type:"first-last-avg","last-avg","cls""pooler(cls+dense)"
        output=self.bert(input_ids,attention_mask,output_hidden_states=True)
        if encoder_type=='first-last-avg':#第一层和最后一层的隐藏层取出并经过平均池化
            first=output.hidden_states[1]#hidden_states,第一个是embeddings,第二个元素才是第一层的hidden_state
            last=output.hidden_states[-1]
            seq_length=first.size(1)#序列长度
            first_avg=torch.avg_pool1d(first.transpose(1,2),kernel_size=seq_length).squeeze(-1)#batch,hid_size
            last_avg=torch.avg_pool1d(last.transpose(1,2),kernel_size=seq_length).squeeze(-1)#batch,hid_size
            final_encoding=torch.avg_pool1d(torch.cat([first_avg.unsqueeze(1),last_avg.unsqueeze(1)],dim=1).transpose(1,2),kernel_size=2).squeeze(-1)
            return final_encoding

        if encoder_type=='last-avg':
            sequence_output=output.last_hidden_state
            seq_length=sequence_output.size(1)
            fineal_encoding=torch.avg_pool1d(sequence_output.transpose(1,2),kernel_size=seq_length).squeeze(-1)
            return fineal_encoding

        if encoder_type=='cls':#pooler output是取[CLS]标记处对应的向量后面接个全连接再接tanh激活后的输出。
            sequence_output=output.last_hidden_state
            cls=sequence_output[:,0]#[b,d]
            return cls

        if encoder_type=='pooler':
            pooler_out_put=output.pooler_output
            return pooler_out_put



if __name__=="__main__":
    model_path = '../model/model_with_MWE/bert-base-multilingual-cased'
    parser=argparse.ArgumentParser()
    # parser.add_argument("--triplet_margin",default=1.0,type=float)
    parser.add_argument("--epoches",default=10,type=int)
    parser.add_argument("--batch_size",default=16,type=int)
    parser.add_argument("--encoder_type",default='last-avg',type=str)
    set_seed(4)
    args=parser.parse_args()
    train_batch_size = args.batch_size
    num_epochs = args.epoches
    encoder_type=args.encoder_type
    # sentencemodel_save_path= '../model/SentenceTransformer/MySentTransformers/MySentTransformer_bert-base-multilingual-cased_'+str(train_batch_size)+'_'+str(num_epochs)

    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 加载数据集
    sts_dataset_path = os.path.join('.', 'sts', 'stsbenchmark.tsv.gz')
    assin2_dataset_path = os.path.join('.', 'assin2')
    # if not os.path.exists(sts_dataset_path):
    #     util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)
    train_labelEN,train_sentenceEN ,_,_,test_labelEN,test_sentenceEN = load_ENdata(sts_dataset_path)
    train_labelPT,train_sentencePT ,_,_,test_labelPT,test_sentencePT = load_PTdata(assin2_dataset_path)
    train_sentence=train_sentenceEN+train_sentencePT
    train_label=train_labelEN+train_labelPT

    # test_sentence=test_sentenceEN+test_sentencePT
    # test_label=test_labelEN+test_labelPT
    # # test_data=[]
    # for index in range(len(test_samples)):
    #     test_samples[index].append(test_label[index])

    train_dataset = CustomDataset(sentence=train_sentence, label=train_label, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=train_batch_size,
                                  collate_fn=collate_fn, num_workers=1)

    #外部数据集
    dev_location = os.path.join('..', 'data', 'dev.csv')
    dev_gold = os.path.join('..', 'data', 'dev.gold.csv')
    eval_location = os.path.join('..', 'data', 'eval.csv')
    dev_formated_file_location = '../submission/dev.submission_format_pretrain.csv'
    eval_formated_file_location = '../submission/eval.submission_format_pretrain.csv'

    total_steps = len(train_dataloader) * num_epochs
    gradient_accumulation_steps=1
    num_train_optimization_steps = int(
        len(train_dataset) / train_batch_size / gradient_accumulation_steps) * num_epochs

    model = Model()

    if torch.cuda.is_available():
        model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.1 * total_steps,
                                                num_training_steps=total_steps)

    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Batch size = %d" % train_batch_size)
    print("  Num steps = %d" % num_train_optimization_steps)
    for epoch in range(num_epochs):
        model.train()
        train_label, train_predict = [], []
        epoch_loss = 0

        for step, batch in enumerate(train_dataloader):
            # for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, segment_ids, label_ids = batch
            if torch.cuda.is_available():
                input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
                label_ids = label_ids.cuda()
            output = model.forword(input_ids=input_ids,attention_mask=input_mask,encoder_type=encoder_type)
            loss = calc_loss(label_ids, output)
            loss.backward()
            epoch_loss += loss
            print("epoch:{}, 正在迭代:{}/{}, Average Loss:{:10f}".format(epoch, step, len(train_dataloader), epoch_loss/(step+1)))  # 在进度条前面定义一段文字
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        #内部验证集和内部测试集
        # model.eval()
        # corr = evaluate(test_sentence,model,tokenizer,encoder_type)
        # s = 'Epoch:{} | corr: {:10f}'.format(epoch, corr)
        # print(s)
        # logs_path = os.path.join('..', 'submission', 'pretrain','coSENT',encoder_type,str(epoch) + '_' + str(train_batch_size) + '_logs.txt')
        # with open(logs_path, 'a+') as f:
        #     s += '\n'
        #     f.write(s)

        #外部验证集和测试集
        model.eval()
        dev_sim, devId_index = get_devSimilarity(dev_location, model,tokenizer,encoder_type)
        evalSen1, evalSen2, evalId_index = prepare_data(eval_location)
        eval_sim = get_similarity(evalSen1, evalSen2,model,tokenizer,encoder_type)
        model_save_path = '../model/modelSave/pretrain_CoSENT/'+encoder_type
        ## Create submission file on the development set.
        submission_data = insert_to_submission1(dev_sim, devId_index, dev_formated_file_location)
        results_file = os.path.join('..', 'submission', 'pretrain','CoSENT',encoder_type,'dev.combined_results-' + str(epoch) + '_' + str(train_batch_size) + '.csv')
        # submission_data.to_csv(results_file,index=False)
        write_csv(submission_data, results_file)

        ## Evaluate development set.
        results = evaluate_submission(results_file, dev_gold, 'pre_train')

        ## Make results printable.
        for result in results:
            for result_index in range(2, 5):
                result[result_index] = 'Did Not Attempt' if result[result_index] is None else result[result_index]

        for row in results:
            print('\t'.join([str(i) for i in row]))

        results_file = os.path.join(model_save_path,
                                    'RESULTS_TABLE-dev.pretrain-' + str(epoch) + '_' + str(
                                        train_batch_size) + '.csv')
        writeList_csv(results, results_file)

        ## Generate combined output for this epoch.'_'+str(args.triplet_margin)+
        submission_data = insert_to_submission1(eval_sim, evalId_index, eval_formated_file_location)
        results_file = os.path.join('..', 'submission', 'pretrain','CoSENT',encoder_type,
                                    'eval.combined_results-' + str(epoch) + '_' + str(train_batch_size) + '.csv')
        # submission_data.to_csv(results_file,index=False)
        write_csv(submission_data, results_file)

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(model_save_path, "bert-base-multilingual-cased_epoch_{}_batch_{}.bin".format(epoch,train_batch_size))
        torch.save(model_to_save.state_dict(), output_model_file)
