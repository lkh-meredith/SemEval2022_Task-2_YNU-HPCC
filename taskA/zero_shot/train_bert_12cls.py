import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append("D:/Meredith/TaskA(1)")
from torch.utils.data import DataLoader
from transformers.models.bert import BertTokenizer
from transformers import AutoTokenizer,AutoModel,BertConfig,BertModel, AutoModelForMaskedLM
from zero_shot.utils import prepare_data,set_seed
from Utils.utils import _score,evaluate_submission,insert_to_submission_file,write_csv
from torch import nn
from transformers import AdamW,get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, f1_score
from torch.utils.data import Dataset
import pandas as pd
import transformers
transformers.logging.set_verbosity_error()
import torch
import argparse

def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids

def collate_fn(batch):
    max_len = 128#max([len(d['input_ids']) for d in batch])
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    for item in batch:
        input_ids.append(pad_to_maxlen(item['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(item['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(item['token_type_ids'], max_len=max_len))
        if item['label'] is not None:
            labels.append(item['label'])

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    if labels is not None:
        all_label_ids = torch.tensor(labels, dtype=torch.long)
    else:
        all_label_ids=None
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids

class CustomDataset(Dataset):
    def __init__(self, sentence,tokenizer,label=None):
        self.sentence = sentence
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            text=self.sentence[index],
            add_special_tokens=True,
            return_token_type_ids=True,
            max_length=128,
            truncation=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': self.label[index] if self.label is not None else None
        }


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

# model
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


def evaluate(model,dataloader):
    y_preds = []
    for batch in dataloader:
        outputs= model([batch[0].cuda(),batch[1].cuda(),batch[2].cuda()])
        probs = torch.softmax(outputs, dim=-1)
        y_pred = torch.argmax(probs, dim=-1)
        y_preds += y_pred.tolist()
    return y_preds

if __name__=="__main__":
    model_path = '../model/bert-base-multilingual-cased'

    parser=argparse.ArgumentParser()
    parser.add_argument("--epoches",default=10,type=int)
    parser.add_argument("--batch_size",default=32,type=int)
    args=parser.parse_args()
    train_batch_size = args.batch_size
    num_epochs = args.epoches
    set_seed(4)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载数据集
    dataset_path = os.path.join('..', 'data', 'ZeroShot')
    train_path = os.path.join(dataset_path, 'train.csv')
    dev_path=os.path.join(dataset_path,'dev.csv')
    # eval_path=os.path.join(dataset_path,'eval.csv')
    test_path=os.path.join(dataset_path,'test.csv')

    train_data,train_label = prepare_data(train_path,'train')
    dev_data,dev_label=prepare_data(dev_path,'dev')
    test_data,_=prepare_data(test_path,'eval')
    model_save_path=os.path.join('..', 'model', 'ZeroShot','bert_Linear_12cls')
    #两个编码器
    train_dataset = CustomDataset(sentence=train_data,label=train_label, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=train_batch_size,
                                  collate_fn=collate_fn, num_workers=1,drop_last=True)
    dev_dataset=CustomDataset(sentence=dev_data,label=dev_label,tokenizer=tokenizer)
    dev_dataloader=DataLoader(dataset=dev_dataset,shuffle=False,batch_size=8,
                              collate_fn=collate_fn, num_workers=1)
    test_dataset = CustomDataset(sentence=test_data,label=None,tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset=test_dataset,shuffle=False, batch_size=8,
                                 collate_fn=collate_fn, num_workers=1)

    total_steps = len(train_dataloader) * num_epochs
    gradient_accumulation_steps = 2
    num_train_optimization_steps = int(len(train_dataset) / train_batch_size / gradient_accumulation_steps) * num_epochs

    model = Bert_Linear()

    if torch.cuda.is_available():
        model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Batch size = %d" % train_batch_size)
    print("  Num steps = %d" % num_train_optimization_steps)
    best_dev_result = 0.0
    # best_predict = list()
    best_epoch = 0
    output_model_file=None
    for epoch in range(num_epochs):
        model.train()
        # train_label, train_predict = [], []
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            logits= model([batch[0].cuda(),batch[1].cuda(),batch[2].cuda()])
            loss = loss_fn(logits, batch[3].cuda())
            loss.backward()
            epoch_loss += loss
            print("epoch:{}, 正在迭代:{}/{},Average Loss:{:10f}".format(epoch, step, len(train_dataloader), epoch_loss / (step + 1)))  # 在进度条前面定义一段文字


            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.eval()
        dev_preds=evaluate(model,dev_dataloader)
        # print("epoch:{},precision:{},f1 score:{}".format(epoch+1, accuracy, precision, f1))

        pred=pd.DataFrame({'prediction':dev_preds})
        dev_prediction_format_file='../data/ZeroShot/dev_predict.csv'
        pred.to_csv(dev_prediction_format_file,index=False)
        dev_submission_format_file='../data/dev_submission_format.csv'
        submission_data = insert_to_submission_file( dev_submission_format_file, dev_path, dev_prediction_format_file, 'zero_shot')
        results_file = os.path.join('..', 'submission', 'ZeroShot','bert_Linear_12cls','dev.combined_results-' + str(epoch+1) + '_' + str(train_batch_size) + '.csv')
        # submission_data.to_csv(results_file,index=False)
        write_csv(submission_data, results_file)

        ## Evaluate development set.
        dev_gold='../SubTaskA_data/Data/dev_gold.csv'
        results = evaluate_submission(results_file, dev_gold)
        ## Make results printable.
        for result in results:
            print(result)

        results_file = os.path.join(model_save_path,
                                    'RESULTS_TABLE-dev_zeroshot-' +str(epoch+1)  + '_' + str(train_batch_size) + '.csv')
        write_csv(results, results_file)

        if best_dev_result < results[3][2]:#记录[EN,PT] F1 score最好的模型
            best_dev_result = results[3][2]
            best_epoch = epoch+1
            # best_predict=y_preds
            # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(model_save_path,"xlm-roberta-base_epoch_{}_batch_{}.pkl".format(best_epoch,train_batch_size))
            torch.save(model, output_model_file)

    model = torch.load(output_model_file)
    model.eval()
    test_preds=evaluate(model,test_dataloader)
    # print("Eval:precision:{},f1 score:{},auc_score:{}".format(accuracy, precision, f1))
    # eval_preds=eval_preds.cpu()
    test_pred = pd.DataFrame({'prediction': test_preds})
    test_prediction_format_file = '../data/ZeroShot/test_predict.csv'
    test_pred.to_csv(test_prediction_format_file,index=False)
    test_submission_format_file = '../data/test_submission_format.csv'
    submission_data = insert_to_submission_file(test_submission_format_file, test_path, test_prediction_format_file, 'zero_shot')
    results_file = os.path.join('..', 'submission', 'ZeroShot','bert_Linear_12cls','test.combined_results-' + str(best_epoch) + '_' + str(train_batch_size) + '.csv')
    write_csv(submission_data, results_file)


