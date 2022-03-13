import os
import re
import math
import pandas as pd
import random
import numpy as np
import time
import datetime
import torch
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from Utils.utils import get_similarity,prepare_data,write_csv,evaluate_submission,insert_to_submission1,writeList_csv,get_devSimilarity
import argparse

def create_and_eval_subtask_b_fine_tune(
        model_path,
        seed,
        data_location,
        dev_formated_file_location,
        eval_formated_file_location,
        modelOut_location,
        outpath,
        epoch=None
) :
    random.seed(seed)
    dev_location = os.path.join( data_location,'dev.csv')
    dev_gold = os.path.join(data_location,'dev.gold.csv')
    # dev_location = os.path.join( data_location,str(language)+'dev.csv')
    # dev_gold = os.path.join(data_location,str(language)+'dev_gold.csv')
    eval_location = os.path.join( data_location, 'eval.csv')
    train_location = os.path.join(data_location,'train_data1.csv')

    ## Training Dataloader
    train_samples,dev_samples = list(),list()
    train_data=pd.read_csv(train_location)

    for i in range(len(train_data)) :
        score = float(train_data['sim'][i])
        inp_example = InputExample(texts=[train_data['sentence_1'][i], train_data['sentence_2'][i]], label=score)
        train_samples.append(inp_example)

    ## Params
    train_batch_size = 16
    model = SentenceTransformer( model_path )
    # print(model)
    # train_dataset = SentencesDataset(train_samples, model)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    # devSen1,devSen2,_=prepare_data(dev_location)
    # dev_sims = get_devGold(dev_location)
    # devEvaluator=evaluation.EmbeddingSimilarityEvaluator(devSen1,devSen2,dev_sims)

    dev_sim = eval_sim = results = None
    if epoch is None :
        ## Going to test all epochs - notice we can't use the default evaluator.
        for epoch in range(1, 11) :#
            warmup_steps = math.ceil(len(train_dataloader) * epoch  * 0.1) #10% of train data for warm-up
            print("Warmup-steps: {}".format(warmup_steps), flush=True)

            model_save_path = os.path.join( modelOut_location, str( seed ), str( epoch ) )
            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=None,
                      epochs=1,
                      evaluation_steps=0,
                      warmup_steps=warmup_steps,
                      output_path=model_save_path,
                      show_progress_bar=True,
                      use_amp=True,
                      # optimizer_params={'lr':1e-10}
                      )

            # devSen1,devSen2,devID=prepare_data(dev_location)
            dev_sim,devId_index=get_devSimilarity(dev_location,model_save_path)
            evalSen1,evalSen2,evalId_index=prepare_data(eval_location)
            eval_sim = get_similarity(model_save_path,evalSen1,evalSen2)

            ## Create submission file on the development set.
            submission_data = insert_to_submission1( dev_sim,devId_index, dev_formated_file_location )
            results_file = os.path.join(outpath, 'dev.combined_results-' + str(seed)+ '_' + str( epoch ) + '.csv' )
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

            results_file = os.path.join(model_save_path, 'RESULTS_TABLE-dev.fineTune_' + str(seed) +'_'+ str(epoch) + '.csv' )
            writeList_csv(results,results_file)

            ## Generate combined output for this epoch.
            submission_data = insert_to_submission1(eval_sim,evalId_index,eval_formated_file_location )
            results_file = os.path.join( outpath, 'eval.combined_results-' + str( seed ) + '_' + str( epoch ) + '.csv' )
            # submission_data.to_csv(results_file,index=False)
            write_csv(submission_data,results_file)

    else :
        ## We already know the best epoch and so will use it.
        warmup_steps = math.ceil(len(train_dataloader) * epoch  * 0.1) #10% of train data for warm-up
        print("Warmup-steps: {}".format(warmup_steps), flush=True)

        model_save_path = os.path.join( modelOut_location,str( seed ), str( epoch ) )
        model.fit(train_objectives=[(train_dataloader, train_loss)],
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
        results_file = os.path.join(outpath, 'dev.combined_results-' +str(seed)+ '_' + str( epoch ) + '.csv' )
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

        results_file = os.path.join( model_save_path, 'RESULTS_TABLE-dev.fineTune_' + str(seed) +'_'+ str(epoch) + '.csv' )
        writeList_csv(results,results_file)

        ## Generate combined output for this epoch.
        submission_data = insert_to_submission1(eval_sim, evalId_index, eval_formated_file_location )
        results_file = os.path.join( outpath, 'eval.combined_results-' + str( seed ) + '_' + str( epoch ) + '.csv' )
        # submission_data.to_csv(results_file,index=False)
        write_csv(submission_data,results_file)

    ## Outside if
    return results

if __name__=="__main__":
    # current_path=os.path.abspath(__file__)
    # model_pathEN= os.path.join('..','modelSave','EN','distiluse-base-multilingual-cased-v1')#'../modelSave/bert-base-cased'
    # model_pathPT=os.path.join('..','modelSave','PT','distiluse-base-multilingual-cased-v1')#'../modelSave/bert-base-multilingual-cased'
    # model_dict={'EN':model_pathEN,'PT':model_pathPT}
    model_path = os.path.join('..', 'model', 'model_with_MWE', 'distiluse-base-multilingual-cased-v1')
    dataLocation = os.path.join('..', 'data')
    dev_formated_file_location = '../submission/dev.submission_format_fineTune.csv'
    eval_formated_file_location = '../submission/eval.submission_format_fineTune.csv'
    modelOut_location = os.path.join('..', 'model', 'modelSave', 'baseline_of_fine-tune')  # '../modelSave'
    file_outpath = os.path.join('..', 'submission', 'baseline_of_fine-tune')

    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--epoches", default=1, type=int)
    # parser.add_argument("--batch_size", default=16, type=int)
    # args = parser.parse_args()

    results=create_and_eval_subtask_b_fine_tune(model_path=model_path,seed=4,data_location=dataLocation,
                                        dev_formated_file_location=dev_formated_file_location,
                                        eval_formated_file_location=eval_formated_file_location,
                                        modelOut_location=modelOut_location,outpath=file_outpath
                                        )#epoch=3
    df=pd.DataFrame(data=results[1:],columns=results[0])
    print(df)