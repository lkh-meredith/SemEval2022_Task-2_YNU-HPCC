import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import pandas as pd
import sys
sys.path.append("D:/Meredith/TaskB")
# sys.path.append("/mnt")
import numpy as np
import time
import datetime
import torch
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, evaluation,models
from transformers import AutoTokenizer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from Utils.utils import get_similarity,get_devSimilarity,prepare_data,write_csv,evaluate_submission,insert_to_submission1,writeList_csv
import argparse

def create_and_eval_subtask_b_fine_tune(
        args,
        model_path,
        seed,
        data_location,
        dev_formated_file_location,
        eval_formated_file_location,
        modelOut_location,
        file_outpath
        # epoch=None
) :
    random.seed(seed)
    dev_location = os.path.join( data_location,'dev.csv')
    dev_gold = os.path.join(data_location,'dev.gold.csv')
    # dev_location = os.path.join( data_location,str(language)+'dev.csv')
    # dev_gold = os.path.join(data_location,str(language)+'dev_gold.csv')
    eval_location = os.path.join( data_location, 'eval.csv')
    train_location = os.path.join(data_location,'train_data.csv')

    ## Training Dataloader
    train_samples1,train_samples2,dev_samples = list(),list(),list()
    train_data=pd.read_csv(train_location,encoding='ISO-8859-1')#
    sentence_1,sentence_2,index_id=prepare_data(train_location)
    for i in range(len(train_data)) :
        # inp_example1,inp_example2=None,None
        if train_data['sim'][i]=="1":
            inp_example1 = InputExample(texts=[sentence_1[i], sentence_2[i]])
            train_samples1.append(inp_example1)
        else:
            inp_example2 = InputExample(texts=[sentence_1[i], train_data['alternative_1'][i], train_data['sentence_2'][i]])#anchor,positive,negative
            train_samples2.append(inp_example2)
        # score = float(train_data['sim'][i])
        # inp_example = InputExample(texts=[train_data['sentence_1'][i], train_data['sentence_2'][i]], label=score)

    ## Params
    # train_batch_size = 4
    train_batch_size = args.batch_size
    epoch=args.epoches

    #mBERT
    # word_embedding_model = models.Transformer(model_path)
    # # Apply mean pooling to get one fixed sized sentence vector
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
    #                                pooling_mode_mean_tokens=True,
    #                                pooling_mode_cls_token=False,
    #                                pooling_mode_max_tokens=False)
    #
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_path,
    #     use_fast=False,
    #     max_length=512,
    #     force_download=True
    # )
    #
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # model._first_module().tokenizer = tokenizer
    # model.save(sentencemodel_save_path)


    model = SentenceTransformer(model_path)
    # print(model)
    # train_dataset = SentencesDataset(train_samples, model)
    train_dataloader1 = DataLoader(train_samples1, shuffle=True, batch_size=train_batch_size)
    train_dataloader2=DataLoader(train_samples2,shuffle=True,batch_size=train_batch_size)
    train_loss1 = losses.MultipleNegativesRankingLoss(model=model,scale=1)
    train_loss2 = losses.TripletLoss(model=model,triplet_margin=args.triplet_margin)#,distance_metric=losses.TripletDistanceMetric.COSINE


    warmup_steps = math.ceil((len(train_dataloader1)+len(train_dataloader2)) * epoch  * 0.1) #10% of train data for warm-up
    print("Warmup-steps: {}".format(warmup_steps), flush=True)

    model_save_path = os.path.join( modelOut_location,str( seed ), str( epoch ),str(args.triplet_margin)+'_'+str(args.batch_size))
    model.fit(train_objectives=[(train_dataloader1, train_loss1),( train_dataloader2,train_loss2)],
              evaluator=None,#devEvaluator,
              epochs=epoch,
              evaluation_steps=0,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              use_amp=True#自适应精度
              )

    dev_sim,devId_index=get_devSimilarity(dev_location,model_save_path)
    evalSen1,evalSen2,evalId_index=prepare_data(eval_location)
    eval_sim = get_similarity(model_save_path,evalSen1,evalSen2)

    ## Create submission file on the development set.
    submission_data = insert_to_submission1( dev_sim, devId_index, dev_formated_file_location )
    results_file = os.path.join(file_outpath, 'dev.combined_results-' +str(seed)+ '_' + str( epoch ) +'_'+ str(args.triplet_margin)+'_'+str(args.batch_size)+'.csv' )
    # submission_data.to_csv(results_file,index=False)
    write_csv(submission_data,results_file)

    ## Evaluate development set.
    results = evaluate_submission( results_file,dev_gold,'fine_tune')

    ## Make results printable.
    for result in results :
        for result_index in range( 2, 5 ) :
            result[result_index] = 'Did Not Attempt' if result[result_index] is None else result[ result_index ]

    for row in results :
        print( '\t'.join( [str(i) for i in row ] ) )

    results_file = os.path.join( model_save_path, 'RESULTS_TABLE-dev.fineTune_' + str(seed) +'_'+ str(epoch) +'_'+str(args.triplet_margin)+'_'+str(args.batch_size)+'.csv' )
    writeList_csv(results,results_file)

    ## Generate combined output for this epoch.
    submission_data = insert_to_submission1(eval_sim, evalId_index, eval_formated_file_location )
    results_file = os.path.join( file_outpath, 'eval.combined_results-' + str( seed ) + '_' + str( epoch ) +'_'+str(args.triplet_margin)+'_'+str(args.batch_size)+'.csv' )
    # submission_data.to_csv(results_file,index=False)
    write_csv(submission_data,results_file)

    ## Outside if
    return results

if __name__=="__main__":
    # current_path=os.path.abspath(__file__)
    model_path=os.path.join('..','model','model_with_MWE','distiluse-base-multilingual-cased-v1')
    # model_path = os.path.join('..', 'model', 'model_with_MWE', 'bert-base-multilingual-cased')
    dataLocation=os.path.join('..','data')
    dev_formated_file_location='../submission/dev.submission_format_fineTune.csv'
    eval_formated_file_location='../submission/eval.submission_format_fineTune.csv'
    modelOut_location=os.path.join('..','model','modelSave','combined_with_losses1','mBERT-MySentTransformers')#'../modelSave'
    file_outpath=os.path.join('..','submission','combined_with_losses1','mBERT-MySentTransformers')#'../submission'

    parser=argparse.ArgumentParser()
    parser.add_argument("--triplet_margin",default=0.1,type=float)
    parser.add_argument("--epoches",default=1,type=int)
    parser.add_argument("--batch_size",default=16,type=int)
    args=parser.parse_args()

    results=create_and_eval_subtask_b_fine_tune(args,model_path=model_path,seed=4,data_location=dataLocation,
                                                dev_formated_file_location=dev_formated_file_location,
                                                eval_formated_file_location=eval_formated_file_location,
                                                modelOut_location=modelOut_location,file_outpath=file_outpath,
                                                )#epoch=3
    df=pd.DataFrame(data=results[1:],columns=results[0])
    print(df)
