import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from train import evaluate,CustomDataset,collate_fn,Model
from utils import prepare_data,set_seed
from Utils.utils import _score,evaluate_submission,insert_to_submission_file,write_csv
from transformers import AutoTokenizer

if __name__=="__main__":
    model_file='../model/OneShot/mBERT/cls/mBERT-base_epoch_7_batch_4.pkl'
    # model=Model()
    # model.load_state_dict(torch.load(model_file))
    model = torch.load(model_file)

    # model = torch.load(model_file)
    tokenizer = AutoTokenizer.from_pretrained('../model/bert-base-multilingual-cased')

    test_path = os.path.join('..', 'data', 'OneShot', 'test.csv')
    test_data,test_MWEs,_=prepare_data(test_path,'eval')
    test_dataset = CustomDataset(sentence=test_data, label=None, MWE=test_MWEs, tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=8,
                                 collate_fn=collate_fn, num_workers=1)
    model.eval()
    _, _, _, test_preds = evaluate(model, test_dataloader, type='eval', encoder_type='cls')
    # print("Eval:precision:{},f1 score:{},auc_score:{}".format(accuracy, precision, f1))
    # eval_preds=eval_preds.cpu()
    pred = pd.DataFrame({'prediction': test_preds})
    test_prediction_format_file = '../data/OneShot/test_predict.csv'
    pred.to_csv(test_prediction_format_file, index=False)
    test_submission_format_file = '../data/test_submission_format.csv'
    submission_data = insert_to_submission_file(test_submission_format_file, test_path, test_prediction_format_file,
                                                'one_shot')
    results_file = os.path.join('..', 'submission', 'OneShot','mBERT', 'cls',
                                'test.combined_results-' + str(9) + '_' + str(4) + '.csv')
    write_csv(submission_data, results_file)