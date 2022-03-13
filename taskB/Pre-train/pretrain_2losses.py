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


class Loss1(nn.Module):
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
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 loss_fct: Callable = nn.CrossEntropyLoss()):
        super(Loss1, self).__init__()
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
        # if concatenation_sent_multiplication:
        #     num_vectors_concatenated += 1
        # logger.info("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        self.dropout = nn.Dropout(p=0.1)
        # self.batch_normalization = nn.BatchNorm1d(num_features=num_vectors_concatenated * sentence_embedding_dimension)
        self.classifier = nn.Linear(in_features=num_vectors_concatenated * sentence_embedding_dimension,
                                    out_features=num_labels, bias=True)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)
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

        # output = self.classifier(features)
        # probs = torch.softmax(output, dim=-1)

        if labels is not None:
            loss = self.loss_fct(output, labels)  # torch.squeeze(output).to(torch.float32)
            # probs = torch.softmax(output, dim=-1)
            # y_pred = torch.argmax(probs, dim=-1)
            # loss = self.loss_fct(output, labels)#torch.squeeze(output).to(torch.float32)
            # return loss
            return loss#, probs[:, 1], y_pred
        else:
            return reps, output

train_samples = []
# train_samples2 = []
dev_samples = []
test_samples = []

with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0
            label=1
            if score<0.5:
                label=0
            inp_example=InputExample(texts=[row['sentence1'], row['sentence2']],label=label)
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            score = float(row['score']) / 5.0
            label=1
            if score<0.5:
                label=0
            inp_example=InputExample(texts=[row['sentence1'], row['sentence2']],label=label)
            test_samples.append(inp_example)
        else:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            label=1
            if score<0.5:
                label=0
            inp_example=InputExample(texts=[row['sentence1'], row['sentence2']],label=label)
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
            label=1
            if score<0.5:
                label=0
            inp_example = InputExample(texts=[pair.getElementsByTagName('t')[0].childNodes[0].data, pair.getElementsByTagName('h')[0].childNodes[0].data], label=label)
            dev_samples.append(inp_example)
    elif split == 'test':
        dom = xml.dom.minidom.parse(os.path.join(assin2_dataset_path,'assin2-test.xml'))
        root = dom.documentElement
        pairs=root.getElementsByTagName('pair')
        for pair in pairs:
            score = float( pair.getAttribute('similarity') ) / 5.0 # Normalize score to range 0 ... 1
            label=1
            if score<0.5:
                label=0
            inp_example = InputExample(texts=[pair.getElementsByTagName('t')[0].childNodes[0].data, pair.getElementsByTagName('h')[0].childNodes[0].data], label=label)
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
            label=1
            if score<0.5:
                label=0
            inp_example=InputExample(texts=[pair.getElementsByTagName('t')[0].childNodes[0].data, pair.getElementsByTagName('h')[0].childNodes[0].data],label=label)
            train_samples.append(inp_example)
            # train_samples.append(inp_example)
    else :
        raise Exception( "Unknown split. Should be one of ['train', 'test', 'validation']." )

if __name__=="__main__":
    model_path = '../model/model_with_MWE/bert-base-multilingual-cased'
    parser=argparse.ArgumentParser()
    # parser.add_argument("--triplet_margin",default=1.0,type=float)
    parser.add_argument("--epoches",default=1,type=int)
    parser.add_argument("--batch_size",default=128,type=int)
    args=parser.parse_args()
    train_batch_size = args.batch_size
    num_epochs = args.epoches
    set_seed(4)
    sentencemodel_save_path= '../model/SentenceTransformer/MySentTransformers/MySentTransformer_bert-base-multilingual-cased_'+str(train_batch_size)+'_'+str(num_epochs)

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
    train_loss = Loss1(model=model,sentence_embedding_dimension=model.get_sentence_embedding_dimension(),num_labels=2)

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
    results_file = os.path.join('..','submission','pretrain','dev.combined_results-'+ str( num_epochs ) +'_'+str(train_batch_size)+'.csv' )
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
    results_file = os.path.join( '..','submission','pretrain','eval.combined_results-'+ str(num_epochs) +'_'+str(train_batch_size)+'.csv' )
    # submission_data.to_csv(results_file,index=False)
    write_csv(submission_data,results_file)
