from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from openprompt import trainer
import os
import sys
sys.path.append("D:/Meredith/TaskA")
import csv
import math
# from datasets import load_dataset
from transformers import AutoTokenizer,AutoModel
import numpy as np
from typing import Union, Tuple, List, Iterable, Dict, Callable
from utils import prepare_data,set_seed
from Utils.utils import evaluate_submission,insert_to_submission_file,write_csv
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn,Tensor
import torch
import pandas as pd
import argparse



if __name__=="__main__":
    model_path = '../model/bert-base-multilingual-cased'
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", model_path)

    parser=argparse.ArgumentParser()
    parser.add_argument("--epoches",default=10,type=int)
    parser.add_argument("--batch_size",default=8,type=int)
    args=parser.parse_args()
    batch_size = args.batch_size
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

    model_save_path=os.path.join('..', 'model', 'OneShot','Prompt')

    train_samples, dev_samples, eval_samples = list(), list(), list()
    for index in range(len(train_data)):
        inp_example = InputExample(guid = index,text_a=train_data[index], text_b=train_MWEs[index], label=train_label[index])
        train_samples.append(inp_example)
    for index in range(len(dev_data)):
        inp_example =InputExample(guid = index,text_a=dev_data[index], text_b=dev_MWEs[index], label=dev_label[index])
        dev_samples.append(inp_example)
    for index in range(len(eval_data)):
        inp_example =InputExample(guid = index,text_a=eval_data[index], text_b=eval_MWEs[index])
        eval_samples.append(inp_example)


    promptTemplate = ManualTemplate(
        text=' Question: Is {"placeholder":"text_b"} a MWE in {"placeholder":"text_a"}? {"mask"}.',
        tokenizer=tokenizer,
    )

    # view wrapped example
    wrapped_example = promptTemplate.wrap_one_example(train_samples[0])
    print(wrapped_example)

    classes = ["0","1"]

    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            "0": "No",
            "1": "Yes",
        },
        tokenizer=tokenizer,
    )

    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )

    train_dataloader = PromptDataLoader(
        dataset=train_samples,
        batch_size=batch_size,
        max_seq_length = 128,
        shuffle=True,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
    )

    dev_dataloader = PromptDataLoader(
        dataset=dev_samples,
        batch_size=batch_size,
        shuffle=False,
        max_seq_length = 128,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
    )

    eval_dataloader = PromptDataLoader(
        dataset=eval_samples,
        batch_size=batch_size,
        max_seq_length=128,
        shuffle=False,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
    )

    # trainer.ClassificationRunner(model=promptModel,train_dataloader=train_dataloader,valid_dataloader=dev_dataloader,test_dataloader=eval_dataloader,
    #                              loss_function=nn.CrossEntropyLoss())#LM-BFF
    if torch.cuda.is_available():
        promptModel = promptModel.cuda()

    loss_function = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in promptModel.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in promptModel.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

    best_dev_result=0.0
    best_epoch=0
    for epoch in range(num_epochs):
        promptModel.train()
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            logits = promptModel(inputs)
            labels = inputs['label']
            loss = loss_function(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            # if step % 100 == 1:
            print("Epoch {}, 正在迭代:{}/{}, average loss: {}".format(epoch + 1, step, len(train_dataloader),tot_loss / (step + 1)), flush=True)

        promptModel.eval()
        dev_pred=[]
        with torch.no_grad():
            for batch in dev_dataloader:
                if torch.cuda.is_available():
                    batch = batch.cuda()
                logits = promptModel(batch)
                dev_pred.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        pred = pd.DataFrame({'prediction': dev_pred})
        dev_prediction_format_file = '../data/OneShot/dev_predict.csv'
        pred.to_csv(dev_prediction_format_file, index=False)
        dev_submission_format_file = '../data/dev_submission_format.csv'
        submission_data = insert_to_submission_file(dev_submission_format_file, dev_path, dev_prediction_format_file,
                                                    'one_shot')
        results_file = os.path.join('..', 'submission', 'OneShot','Prompt',
                                    'dev.combined_results-' + str(epoch + 1) + '_' + str(batch_size) + '.csv')
        # submission_data.to_csv(results_file,index=False)
        write_csv(submission_data, results_file)

        ## Evaluate development set.
        dev_gold = '../SubTaskA_data/Data/dev_gold.csv'
        results = evaluate_submission(results_file, dev_gold)
        ## Make results printable.
        for result in results:
            print(result)

        results_file = os.path.join(model_save_path,
                                    'RESULTS_TABLE-dev_oneshot-' + str(epoch + 1) + '_' + str(batch_size) + '.csv')
        write_csv(results, results_file)

        if best_dev_result < results[6][2]:  # 记录[EN,PT] F1 score最好的模型
            best_dev_result = results[6][2]
            best_epoch = epoch + 1
            output_model_file = os.path.join(model_save_path,
                                             "bert-base-multilingual-cased_epoch_{}_batch_{}.pkl".format(best_epoch,batch_size))
            torch.save(promptModel, output_model_file)

    model = torch.load(output_model_file)
    model.eval()
    eval_pred = []
    with torch.no_grad():
        for batch in eval_dataloader:
            if torch.cuda.is_available():
                batch = batch.cuda()
            logits = promptModel(batch)
            # labels = inputs['label']
            eval_pred.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    # print("Eval:precision:{},f1 score:{},auc_score:{}".format(accuracy, precision, f1))
    # eval_preds=eval_preds.cpu()
    eval_pred = pd.DataFrame({'prediction': eval_pred})
    eval_prediction_format_file = '../data/OneShot/eval_predict.csv'
    eval_pred.to_csv(eval_prediction_format_file, index=False)
    eval_submission_format_file = '../data/eval_submission_format.csv'
    submission_data = insert_to_submission_file(eval_submission_format_file, eval_path, eval_prediction_format_file,
                                                'one_shot')
    results_file = os.path.join('..', 'submission', 'OneShot', 'Prompt',
                                'eval.combined_results-' + str(best_epoch) + '_' + str(batch_size) + '.csv')
    write_csv(submission_data, results_file)