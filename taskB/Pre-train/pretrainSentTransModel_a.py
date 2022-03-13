import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append("D:/Meredith/TaskB")
import csv
import gzip
import xml.dom.minidom
import math
from datetime                         import datetime
from torch.utils.data                 import DataLoader
from sklearn.metrics.pairwise         import paired_cosine_distances
# from datasets import load_dataset
from sentence_transformers  import SentenceTransformer,  LoggingHandler, losses, util, models
from sentence_transformers.readers import InputExample
from transformers import AutoTokenizer,AutoModel
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from typing import Union, Tuple, List, Iterable, Dict, Callable
from Utils.utils import get_similarity,prepare_data,get_devSimilarity,write_csv,evaluate_submission,insert_to_submission1,writeList_csv,set_seed
from torch import nn,Tensor
import torch
import argparse

sts_dataset_path = os.path.join('.', 'sts', 'stsbenchmark.tsv.gz')
assin2_dataset_path=os.path.join('.','assin2')
# if not os.path.exists(sts_dataset_path):
#     util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

train_samples = []
dev_samples = []
test_samples = []

with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0
            inp_example=InputExample(texts=[row['sentence1'], row['sentence2']],label=score)
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            score = float(row['score']) / 5.0
            inp_example=InputExample(texts=[row['sentence1'], row['sentence2']],label=score)
            test_samples.append(inp_example)
        else:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            inp_example=InputExample(texts=[row['sentence1'], row['sentence2']],label=score)
            train_samples.append(inp_example)


for split in [ 'train', 'validation', 'test' ] :
    # dataset = load_dataset( "assin2")
    # for elem in dataset :
    ## {'entailment_judgment': 1, 'hypothesis': 'Uma criança está segurando uma pistola de água', 'premise': 'Uma criança risonha está segurando uma pistola de água e sendo espirrada com água', 'relatedness_score': 4.5, 'sentence_pair_id': 1}
    # inp_example = InputExample(texts=[elem['hypothesis'], elem['premise']])

    if split == 'validation':
        dom = xml.dom.minidom.parse(os.path.join(assin2_dataset_path,'assin2-dev.xml'))
        root = dom.documentElement
        pairs=root.getElementsByTagName('pair')
        for pair in pairs:
            score = float( pair.getAttribute('similarity') ) / 5.0 # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[pair.getElementsByTagName('t')[0].childNodes[0].data, pair.getElementsByTagName('h')[0].childNodes[0].data], label=score)
            dev_samples.append(inp_example)
    elif split == 'test':
        dom = xml.dom.minidom.parse(os.path.join(assin2_dataset_path,'assin2-test.xml'))
        root = dom.documentElement
        pairs=root.getElementsByTagName('pair')
        for pair in pairs:
            score = float( pair.getAttribute('similarity') ) / 5.0 # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[pair.getElementsByTagName('t')[0].childNodes[0].data, pair.getElementsByTagName('h')[0].childNodes[0].data], label=score)
            test_samples.append(inp_example)

    elif split == 'train' :
        dom = xml.dom.minidom.parse(os.path.join(assin2_dataset_path,'assin2-train-only.xml'))
        root = dom.documentElement
        pairs=root.getElementsByTagName('pair')
        for pair in pairs:
            # if float(pair.getAttribute('similarity'))>=4.0:
            #     inp_example1 = InputExample(texts=[pair.getElementsByTagName('t')[0].childNodes[0].data, pair.getElementsByTagName('h')[0].childNodes[0].data])
            #     train_samples1.append(inp_example1)
            # else:
            score = float( pair.getAttribute('similarity') ) / 5.0 # Normalize score to range 0 ... 1

            inp_example=InputExample(texts=[pair.getElementsByTagName('t')[0].childNodes[0].data, pair.getElementsByTagName('h')[0].childNodes[0].data],label=score)
            train_samples.append(inp_example)
            # train_samples.append(inp_example)
    else :
        raise Exception( "Unknown split. Should be one of ['train', 'test', 'validation']." )

if __name__=="__main__":
    model_path = '../model/model_with_MWE/bert-base-multilingual-cased'
    parser=argparse.ArgumentParser()
    # parser.add_argument("--triplet_margin",default=1.0,type=float)
    parser.add_argument("--epoches",default=1,type=int)
    parser.add_argument("--batch_size",default=16,type=int)
    args=parser.parse_args()
    train_batch_size = args.batch_size
    num_epochs = args.epoches
    set_seed(4)
    sentencemodel_save_path= '../model/SentenceTransformer/MySentTransformersa/MySBERTa_bert-base-multilingual-cased_'+str(train_batch_size)+'_'+str(num_epochs)

    word_embedding_model = models.Transformer(model_path)#,cache_dir=Nonemax_seq_length=256,'bert-base-multilingual-cased'

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast = False ,
        max_length = 128 ,
        force_download = True
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model._first_module().tokenizer = tokenizer

    model.save(sentencemodel_save_path)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    train_loss = losses.CosineSimilarityLoss(model=model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples)

    # Configure the training.
    warmup_steps = math.ceil((len(train_dataloader)) * num_epochs  * 0.1) #10% of train data for warm-up
    print("Warmup-steps: {}".format(warmup_steps), flush=True)

    # Train the model

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=sentencemodel_save_path
              )

    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples)
    test_evaluator(model, output_path=sentencemodel_save_path)

    data_location=os.path.join('..','data')
    dev_location = os.path.join( data_location,'dev.csv')
    dev_gold = os.path.join(data_location,'dev.gold.csv')
    eval_location = os.path.join( data_location, 'eval.csv')
    dev_formated_file_location='../submission/dev.submission_format_pretrain.csv'
    eval_formated_file_location='../submission/eval.submission_format_pretrain.csv'

    dev_sim,devId_index  = get_devSimilarity( dev_location , sentencemodel_save_path)
    evalSen1,evalSen2,evalId_index=prepare_data(eval_location)
    eval_sim = get_similarity(sentencemodel_save_path,evalSen1,evalSen2)

    ## Create submission file on the development set.
    submission_data = insert_to_submission1( dev_sim, devId_index, dev_formated_file_location )
    results_file = os.path.join('..','submission','pretrain','MySBERTa','dev.combined_results-'+ str( num_epochs ) +'_'+str(train_batch_size)+'.csv' )
    # submission_data.to_csv(results_file,index=False)
    write_csv(submission_data,results_file)

    ## Evaluate development set.
    results = evaluate_submission( results_file,dev_gold,'pre_train')

    ## Make results printable.
    for result in results :
        for result_index in range( 2, 5 ) :
            result[result_index] = 'Did Not Attempt' if result[result_index] is None else result[ result_index ]

    for row in results :
        print( '\t'.join( [str(i) for i in row ] ) )

    results_file = os.path.join(sentencemodel_save_path, 'RESULTS_TABLE-dev.pretrain-'+ str(num_epochs)+'_'+str(train_batch_size)+'.csv' )
    writeList_csv(results,results_file)

    ## Generate combined output for this epoch.'_'+str(args.triplet_margin)+
    submission_data = insert_to_submission1(eval_sim, evalId_index, eval_formated_file_location )
    results_file = os.path.join( '..','submission','pretrain','MySBERTa','eval.combined_results-'+ str(num_epochs) +'_'+str(train_batch_size)+'.csv' )
    # submission_data.to_csv(results_file,index=False)
    write_csv(submission_data,results_file)