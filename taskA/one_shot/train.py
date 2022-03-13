import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append("D:/Meredith/TaskA")
from torch.utils.data import DataLoader
from transformers.models.bert import BertTokenizer
from transformers import AutoTokenizer,AutoModel,BertConfig,BertModel, AutoModelForMaskedLM
from utils import prepare_data,set_seed
from Utils.utils import _score,evaluate_submission,insert_to_submission_file,write_csv
from torch import nn
from transformers import AdamW,get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
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
    def __init__(self, sentence,MWE,tokenizer,label=None):
        self.sentence = sentence
        self.MWE=MWE
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            text=self.sentence[index],
            text_pair=self.MWE[index],
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

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.config=BertConfig.from_pretrained('../model/xlm-roberta-base/config.json')
        # self.bert=BertModel.from_pretrained('../model/bert-base-multilingual-cased/')
        self.bert = AutoModelForMaskedLM.from_pretrained('../model/xlm-roberta-base')
        # self.batch_normalization = nn.BatchNorm1d(num_features=768)
        # self.weight=nn.Linear(in_features=train_batch_size*2,out_features=train_batch_size,bias=True)
        self.classifier = nn.Linear(in_features=768, out_features=2, bias=True)

        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)
        self.dropout = nn.Dropout(p=0.1)
        self.loss_fct = nn.CrossEntropyLoss()

    def forword(self,input_ids,attention_mask,encoder_type,labels):
        self.output = self.bert(input_ids, attention_mask, output_hidden_states=True)
        #param encoder_type:"first-last-avg","last-avg","cls""pooler(cls+dense)"
        if encoder_type=='first-last-avg':
            first=self.output.hidden_states[1]#hidden_states,第一个是embeddings,第二个元素才是第一层的hidden_state
            last=self.output.hidden_states[-1]
            seq_length=first.size(1)#序列长度
            first_avg=torch.avg_pool1d(first.transpose(1,2),kernel_size=seq_length).squeeze(-1)#batch,hid_size
            last_avg=torch.avg_pool1d(last.transpose(1,2),kernel_size=seq_length).squeeze(-1)#batch,hid_size
            # first_last=torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=0).squeeze(1).T
            # final_encoding=self.weight(first_last).T#学习两层的参数权重
            final_encoding=torch.avg_pool1d(torch.cat([first_avg.unsqueeze(1),last_avg.unsqueeze(1)],dim=1).transpose(1,2),kernel_size=2).squeeze(-1)
            features = self.dropout(final_encoding)

            probs = self.classifier(features)#学习权重
            if labels is not None:
                loss = self.loss_fct(probs, labels)  # torch.squeeze(output).to(torch.float32)
            else:
                loss=None
            probs = torch.softmax(probs, dim=-1)
            y_pred = torch.argmax(probs, dim=-1)
            return loss,probs[:, 1],y_pred
            # return final_encoding

        if encoder_type=='last-avg':
            # sequence_output=self.output.last_hidden_state
            sequence_output=self.output.hidden_states[-1]
            seq_length=sequence_output.size(1)
            final_encoding=torch.avg_pool1d(sequence_output.transpose(1,2),kernel_size=seq_length).squeeze(-1)
            features = self.dropout(final_encoding)
            # normalized = self.batch_normalization(features)
            probs = self.classifier(features)
            if labels is not None:
                loss = self.loss_fct(probs, labels)  # torch.squeeze(output).to(torch.float32)
            else:
                loss = None
            probs = torch.softmax(probs, dim=-1)
            y_pred = torch.argmax(probs, dim=-1)
            return loss, probs[:, 1], y_pred
            # return fineal_encoding

        if encoder_type=='cls':
            # sequence_output=self.output.last_hidden_state
            sequence_output = self.output.hidden_states[-1]
            cls=sequence_output[:,0]#[b,d]
            features = self.dropout(cls)
            # normalized = self.batch_normalization(features)
            probs = self.classifier(features)
            if labels is not None:
                loss = self.loss_fct(probs, labels)  # torch.squeeze(output).to(torch.float32)
            else:
                loss = None
            probs = torch.softmax(probs, dim=-1)
            y_pred = torch.argmax(probs, dim=-1)
            return loss, probs[:, 1], y_pred

        if encoder_type=='pooler':
            pooler_out_put=self.output.pooler_output
            features = self.dropout(pooler_out_put)
            # normalized = self.batch_normalization(features)
            probs = self.classifier(features)
            if labels is not None:
                loss = self.loss_fct(probs, labels)  # torch.squeeze(output).to(torch.float32)
            else:
                loss = None
            probs = torch.softmax(probs, dim=-1)
            y_pred = torch.argmax(probs, dim=-1)
            return loss, probs[:, 1], y_pred

def evaluate(model,dataloader,type='dev',encoder_type='first-last-avg'):
    y_preds = []
    # y_probs = []
    ys = []
    if type=='dev':
        for dev_batch in dataloader:
            input_ids, input_mask, segment_ids, label_ids = dev_batch
            input_ids = input_ids.cuda()
            # segment_ids = segment_ids.cuda()
            input_mask = input_mask.cuda()
            label_ids = label_ids.cuda()

            probs,_, y_pred = model.forword(input_ids=input_ids, attention_mask=input_mask, encoder_type=encoder_type,labels=label_ids)
            labels = label_ids.cpu().numpy().tolist()
            y_preds += y_pred.tolist()
            # y_probs += probs
            ys += labels

        accuracy = accuracy_score(ys, y_preds)
        precision = precision_score(ys, y_preds)
        f1 = f1_score(ys, y_preds,average='macro')
        # auc_score = roc_auc_score(ys, y_probs)
        # print("epoch:{},precision:{},f1 score:{},auc_score:{}".format(epoch,accuracy, precision, f1, auc_score))
        return accuracy,precision,f1,y_preds

    elif type=='eval':
        for eval_batch in dataloader:
            input_ids, input_mask, segment_ids,_= eval_batch
            input_ids = input_ids.cuda()
            # segment_ids = segment_ids.cuda()
            input_mask = input_mask.cuda()
            _, _, y_pred = model.forword(input_ids=input_ids, attention_mask=input_mask, encoder_type=encoder_type,labels=None)
            y_preds += y_pred.tolist()
        return _,_,_,y_preds



if __name__=="__main__":
    model_path = '../model/bert-base-multilingual-cased'

    parser=argparse.ArgumentParser()
    parser.add_argument("--epoches",default=10,type=int)
    parser.add_argument("--batch_size",default=4,type=int)
    parser.add_argument("--encoder_type",default='cls',type=str)
    args=parser.parse_args()
    train_batch_size = args.batch_size
    num_epochs = args.epoches
    encoder_type=args.encoder_type
    set_seed(4)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载数据集
    dataset_path = os.path.join('..', 'data', 'OneShot')
    train_path = os.path.join(dataset_path, 'train.csv')
    dev_path=os.path.join(dataset_path,'dev.csv')
    eval_path=os.path.join(dataset_path,'eval.csv')

    train_data,train_MWEs,train_label = prepare_data(train_path,'train')
    dev_data,dev_MWEs,dev_label=prepare_data(dev_path,'dev')
    eval_data,eval_MWEs,_=prepare_data(eval_path,'eval')
    model_save_path=os.path.join('..', 'model', 'OneShot','mBERT',encoder_type)
    #两个编码器
    train_dataset = CustomDataset(sentence=train_data,MWE=train_MWEs,label=train_label, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=train_batch_size,
                                  collate_fn=collate_fn, num_workers=1,drop_last=True)
    dev_dataset=CustomDataset(sentence=dev_data,MWE=dev_MWEs,label=dev_label,tokenizer=tokenizer)
    dev_dataloader=DataLoader(dataset=dev_dataset,shuffle=False,batch_size=8,
                              collate_fn=collate_fn, num_workers=1)
    eval_dataset = CustomDataset(sentence=eval_data,label=None,MWE=eval_MWEs,tokenizer=tokenizer)
    eval_dataloader = DataLoader(dataset=eval_dataset,shuffle=False, batch_size=8,
                                collate_fn=collate_fn, num_workers=1)

    total_steps = len(train_dataloader) * num_epochs
    gradient_accumulation_steps = 2
    num_train_optimization_steps = int(len(train_dataset) / train_batch_size / gradient_accumulation_steps) * num_epochs

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

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)

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
            # for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, segment_ids, label_ids = batch
            if torch.cuda.is_available():
                input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
                label_ids = label_ids.cuda()
            loss,_, _ = model.forword(input_ids=input_ids, attention_mask=input_mask, encoder_type=encoder_type,labels=label_ids)
            loss.backward()
            epoch_loss += loss
            print("epoch:{}, 正在迭代:{}/{}, Loss:{:10f}".format(epoch, step, len(train_dataloader), loss))  # 在进度条前面定义一段文字
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)


            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.eval()
        accuracy,precision,f1,y_preds=evaluate(model,dev_dataloader,type='dev',encoder_type=encoder_type)
        # print("epoch:{},precision:{},f1 score:{}".format(epoch+1, accuracy, precision, f1))

        pred=pd.DataFrame({'prediction':y_preds})
        dev_prediction_format_file='../data/OneShot/dev_predict.csv'
        pred.to_csv(dev_prediction_format_file,index=False)
        dev_submission_format_file='../data/dev_submission_format.csv'
        submission_data = insert_to_submission_file( dev_submission_format_file, dev_path, dev_prediction_format_file, 'one_shot')
        results_file = os.path.join('..', 'submission', 'OneShot','mBERT',encoder_type,'dev.combined_results-' + str(epoch+1) + '_' + str(train_batch_size) + '.csv')
        # submission_data.to_csv(results_file,index=False)
        write_csv(submission_data, results_file)

        ## Evaluate development set.
        dev_gold='../SubTaskA_data/Data/dev_gold.csv'
        results = evaluate_submission(results_file, dev_gold)
        ## Make results printable.
        for result in results:
            print(result)

        results_file = os.path.join(model_save_path,
                                    'RESULTS_TABLE-dev_oneshot-' +str(epoch+1)  + '_' + str(train_batch_size) + '.csv')
        write_csv(results, results_file)

        if best_dev_result < results[6][2]:#记录[EN,PT] F1 score最好的模型
            best_dev_result = results[6][2]
            best_epoch = epoch+1
            # best_predict=y_preds
            # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(model_save_path,"xlm-roberta-base_epoch_{}_batch_{}.pkl".format(best_epoch,train_batch_size))
            torch.save(model, output_model_file)

    model = torch.load(output_model_file)
    model.eval()
    _,_,_,eval_preds=evaluate(model,eval_dataloader,type='eval',encoder_type=encoder_type)
    # print("Eval:precision:{},f1 score:{},auc_score:{}".format(accuracy, precision, f1))
    # eval_preds=eval_preds.cpu()
    eval_pred = pd.DataFrame({'prediction': eval_preds})
    eval_prediction_format_file = '../data/OneShot/eval_predict.csv'
    eval_pred.to_csv(eval_prediction_format_file,index=False)
    eval_submission_format_file = '../data/eval_submission_format.csv'
    submission_data = insert_to_submission_file(eval_submission_format_file, eval_path, eval_prediction_format_file, 'one_shot')
    results_file = os.path.join('..', 'submission', 'OneShot','mBERT',encoder_type,'eval.combined_results-' + str(best_epoch) + '_' + str(train_batch_size) + '.csv')
    write_csv(submission_data, results_file)














