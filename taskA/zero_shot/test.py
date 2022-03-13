import torch
import os
# import sys
# sys.path.append("/mnt")
import pandas as pd
from torch.utils.data import DataLoader
from train_textCNN import evaluate,CustomDataset,collate_fn
from zero_shot.utils import prepare_data,set_seed
from Utils.utils import _score,evaluate_submission,insert_to_submission_file,write_csv
from torch import nn
from transformers import AutoTokenizer,AutoModel
from math import sqrt

# class TextCNN(nn.Module):
#     def __init__(self):
#         super(TextCNN, self).__init__()
#         self.kernel_list=[2,3,4]
#         self.num_filter_total = 3 * len(self.kernel_list)
#         self.Weight = nn.Linear(self.num_filter_total, 2, bias=False)
#         self.bias = nn.Parameter(torch.ones([2]))
#         self.filter_list = nn.ModuleList([
#             nn.Conv2d(1, 3, kernel_size=(size, 768)) for size in self.kernel_list
#         ])
#
#     def forward(self, x):
#         # x: [bs, seq, hidden]
#         x = x.unsqueeze(1) # [bs, channel=1, seq, hidden]
#
#         pooled_outputs = []
#         for i, conv in enumerate(self.filter_list):
#             h = torch.nn.fuctional.relu(conv(x)) # [bs, channel=1, seq-kernel_size+1, 1]
#             mp = nn.MaxPool2d(#                 kernel_size = (12 -self.kernel_list[i] +1, 1)
#             )
#             # mp: [bs, channel=3, w, h]
#             pooled = mp(h).permute(0, 3, 2, 1) # [bs, h=1, w=1, channel=3]
#             pooled_outputs.append(pooled)
#
#         h_pool = torch.cat(pooled_outputs, len(self.kernel_list)) # [bs, h=1, w=1, channel=3 * 3]
#         h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])
#
#         # out = torch.nn.functional.dropout(input=h_pool_flat, p=0.1)
#         output = self.Weight(h_pool_flat) + self.bias # [bs, n_class]
#
#         return output
#
# # model
# class Bert_Blend_CNN(nn.Module):
#     def __init__(self):
#         super(Bert_Blend_CNN, self).__init__()
#         self.bert = AutoModel.from_pretrained('../model/xlm-roberta-base', output_hidden_states=True, return_dict=True)
#         # self.linear = nn.Linear(768, 2)
#         self.textcnn = TextCNN()
#
#     def forward(self, X):
#         input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # 返回一个output字典
#         # 取每一层encode出来的向量
#         # outputs.pooler_output: [bs, hidden_size]
#         hidden_states = outputs.hidden_states # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
#         cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1) # [bs, 1, hidden]
#         # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textcnn的输入
#         for i in range(2, 13):
#             cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
#         # cls_embeddings: [bs, encode_layer=12, hidden]
#         logits = self.textcnn(cls_embeddings)
#         return logits

# class Linear(nn.Module):
#     def __init__(self):
#         super(Linear, self).__init__()
#         self.linear1 = nn.Linear(12, 1, bias=True)
#         self.linear2 = nn.Linear(768,2)
#
#     def forward(self, x):
#         # x: [bs, seq, hidden]
#         x=x.transpose(1,2)
#         x1=self.linear1(x)
#         x1=x1.transpose(1,2)
#         # x1=torch.nn.functional.dropout(input=x1, p=0.1)
#         x2=self.linear2(x1)
#         output=x2.transpose(1,2).squeeze(2)
#
#         return output
#
# # model
# class Bert_Linear(nn.Module):
#     def __init__(self):
#         super(Bert_Linear, self).__init__()
#         self.bert = AutoModel.from_pretrained('../model/xlm-roberta-base', output_hidden_states=True, return_dict=True)
#         # self.linear = nn.Linear(768, 2)
#         self.linear = Linear()
#
#     def forward(self, X):
#         input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # 返回一个output字典
#         # 取每一层encode出来的向量
#         # outputs.pooler_output: [bs, hidden_size]
#         hidden_states = outputs.hidden_states # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
#         cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1) # [bs, 1, hidden]
#         # 将每一层的第一个token(cls向量)提取出来，拼在一起当作cnn的输入
#         for i in range(2, 13):
#             cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
#         # cls_embeddings: [bs, encode_layer=12, hidden]
#         output = self.linear(cls_embeddings)
#         return output
# class SelfAttention(nn.Module):
#     dim_in: int
#     dim_k: int
#     dim_v: int
#
#     def __init__(self, dim_in, dim_k, dim_v):
#         super(SelfAttention, self).__init__()
#         self.dim_in = dim_in
#         self.dim_k = dim_k
#         self.dim_v = dim_v
#         self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
#         self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
#         self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
#         self._norm_fact = 1 / sqrt(dim_k)
#
#         # self.linear1=nn.Linear(12,1)
#         self.maxpooling=nn.MaxPool1d(kernel_size = 12)
#         self.linear=nn.Linear(dim_v,2)
#
#     def forward(self, x):
#         # x: batch, seq, dim_in
#         batch, n, dim_in = x.shape
#         assert dim_in == self.dim_in
#
#         q = self.linear_q(x)  # batch, n, dim_k
#         k = self.linear_k(x)  # batch, n, dim_k
#         v = self.linear_v(x)  # batch, n, dim_v
#
#         dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
#         dist = torch.softmax(dist, dim=-1)  # batch, n, n
#
#         att = torch.bmm(dist, v).transpose(1, 2) #bs,12,dim_v, # batch, dim_v ,seq
#         output=self.maxpooling(att).transpose(1, 2)
#
#         # att_list=[]
#         # for index in range(len(att[0])):
#         #     line=att[:,index,:]
#         #     att_list.append(line.unsqueeze(1))
#         #
#         # output=torch.cat(att_list,2)
#         output= self.linear(output)
#
#         return output.transpose(1,2).squeeze(2)

# model
# class Bert_Attention(nn.Module):
#     def __init__(self):
#         super(Bert_Attention, self).__init__()
#         self.bert = AutoModel.from_pretrained('../model/xlm-roberta-base', output_hidden_states=True, return_dict=True)
#         # self.linear = nn.Linear(768, 2)
#         # self.multiHead_selfAttention = MultiHeadSelfAttention(768,512,512)
#         self.selfAttention=SelfAttention(768,512,16)
#
#     def forward(self, X):
#         input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # 返回一个output字典
#         # 取每一层encode出来的向量
#         # outputs.pooler_output: [bs, hidden_size]
#         hidden_states = outputs.hidden_states # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
#         cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1) # [bs, 1, hidden]
#         # 将每一层的第一个token(cls向量)提取出来，拼在一起当作cnn的输入
#         # cls_embeddings=[]
#         for i in range(2, 13):
#             # cls_embeddings.append(hidden_states[i][:, 0, :].unsqueeze(1))
#             cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
#         # cls_embeddings = torch.cat(cls_embeddings, dim=1)
#         # cls_embeddings: [bs, encode_layer=12, hidden]
#         output = self.selfAttention(cls_embeddings)
#         return output

# class Bert_Linear(nn.Module):
#     def __init__(self):
#         super(Bert_Linear, self).__init__()
#         self.bert = AutoModel.from_pretrained('../model/xlm-roberta-base', output_hidden_states=True, return_dict=True)
#         self.linear = nn.Linear(768, 2)
#         # self.linear = Linear()
#
#     def forward(self, X):
#         input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # 返回一个output字典
#         # 取每一层encode出来的向量
#         # outputs.pooler_output: [bs, hidden_size]
#         hidden_states = outputs.last_hidden_state # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
#         cls_embeddings = hidden_states[:, 0, :] # [bs, hidden]
#         # # 将每一层的第一个token(cls向量)提取出来，拼在一起当作输入
#         # for i in range(2, 13):
#         #     cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
#         # cls_embeddings: [bs, encode_layer=12, hidden]
#         output = self.linear(cls_embeddings)
#         return output

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.linear1 = nn.Linear(12, 1, bias=True)
        self.linear2 = nn.Linear(768,2)

    def forward(self, x):
        # x: [bs, seq, hidden]
        x=x.transpose(1,2)
        x1=self.linear1(x)
        x1=x1.transpose(1,2)
        # x1=torch.nn.functional.dropout(input=x1, p=0.1)
        x2=self.linear2(x1)
        output=x2.transpose(1,2).squeeze(2)

        return output

class Bert_Linear(nn.Module):
    def __init__(self):
        super(Bert_Linear, self).__init__()
        self.bert = AutoModel.from_pretrained('../model/bert-base-multilingual-cased', output_hidden_states=True, return_dict=True)
        print(self.bert)
        # self.linear = nn.Linear(768, 2)
        self.linear = Linear()

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # 返回一个output字典
        # 取每一层encode出来的向量
        # outputs.pooler_output: [bs, hidden_size]
        hidden_states = outputs.hidden_states # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1) # [bs, 1, hidden]
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作输入
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [bs, encode_layer=12, hidden]
        output = self.linear(cls_embeddings)
        return output


if __name__=="__main__":
    model_file='../model/ZeroShot/bert_Linear_12cls/mBERT-base_epoch_7_batch_32.pkl'
    # model=Model()
    # model.load_state_dict(torch.load(model_file))
    model = torch.load(model_file)

    # model = torch.load(model_file)
    tokenizer = AutoTokenizer.from_pretrained('../model/bert-base-multilingual-cased')

    test_path = os.path.join('..', 'data', 'ZeroShot', 'test.csv')
    test_data,label=prepare_data(test_path,'test')

    test_dataset = CustomDataset(sentence=test_data, label=None, tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=16,
                                 collate_fn=collate_fn, num_workers=1)

    model.eval()
    test_preds = evaluate(model, test_dataloader)
    # print("Eval:precision:{},f1 score:{},auc_score:{}".format(accuracy, precision, f1))
    # eval_preds=eval_preds.cpu()
    pred = pd.DataFrame({'prediction': test_preds})
    test_prediction_format_file = '../data/ZeroShot/test_predict.csv'
    pred.to_csv(test_prediction_format_file, index=False)
    test_submission_format_file = '../data/test_submission_format.csv'
    submission_data = insert_to_submission_file(test_submission_format_file, test_path, test_prediction_format_file,
                                                'zero_shot')
    results_file = os.path.join('..', 'submission', 'ZeroShot', 'bert_Linear_12cls',
                                'test.combined_results-' + str(7) + '_' + str(32) + '.csv')
    write_csv(submission_data, results_file)