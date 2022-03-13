import re
import os
import sys
import pandas as pd
import numpy as np
import random
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances
import csv
from scipy.stats import pearsonr, spearmanr

#create baseline of create train data
def get_alter_sentence(data_location) :
    df=pd.read_csv(data_location,encoding='ISO-8859-1')
    alter1 = list()
    alter2 = list()
    for i in range(len(df)):
        if df['sim'][i] == 'None':
            alter1.append(df['alternative_1'][i])
            alter2.append(df['alternative_2'][i])
    return alter1, alter2


def out_train_file(data_location,similarity):
    df=pd.read_csv(data_location,encoding='ISO-8859-1')
    k=0
    sentence1,sentence2=list(),list()
    MWE=list()
    alter_sim=list()
    for i in range(len(df)):
        sentence1.append(df['sentence_1'][i])
        sentence2.append(df['sentence_2'][i])
        MWE.append(df['MWE1'][i])
        if(df['sim'][i]=='None'):
            alter_sim.append(similarity[k])
            k+=1
        else:
            alter_sim.append(df['sim'][i])
    dataframe = pd.DataFrame({'MWE1':MWE,'sentence_1':sentence1,'sentence_2':sentence2,'sim':alter_sim})
    return dataframe


def load_csv( path ) :
    header = None
    data = list()
    with open( path, encoding='utf-8') as csvfile:
        reader = csv.reader( csvfile )
        for row in reader :
            if header is None :
                header = row
                continue
            data.append( row )
    return header, data

def writeList_csv(data, location):
    with open( location, 'a', encoding='utf-8',newline='') as csvfile:
        writer = csv.writer( csvfile )
        if os.path.getsize(location)==0:#文件为空则添加表头
            for i in range(len(data)):
                writer.writerow(data[i])
        else:
            for i in range(len(data)-1):
                writer.writerow(data[i+1])
    print( "Wrote {}".format(location))
    return

def write_csv( data, location ) :
    with open( location, 'a', encoding='utf-8',newline='') as csvfile:
        writer = csv.writer( csvfile )
        if os.path.getsize(location)==0:#文件为空则添加表头
            writer.writerow([column for column in data])
        for i in data.values:
            writer.writerow(i)
    print( "Wrote {}".format( location ))
    return

def set_seed(seed: int):
    """
    Modified from : https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_utils.py
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available

    # torch.backends.cudnn.benchmark = False


def tokenise_idiom( phrase ) :
    # word=re.sub("[?]",'',phrase)
    s='ID' + re.sub( r'[\s|-]', '', phrase).lower() + 'ID'
    return s

def get_devGold(dev_location):
    df_label=pd.read_csv(dev_location)
    sim=list()
    for i in range(len(df_label)):
        sim.append(float(df_label['sim'][i]))
    return sim

def get_similarity(model_location, sent1, sent2) :

    model = SentenceTransformer(model_location)
    sent1_embeddings = model.encode( sent1, show_progress_bar=True, convert_to_numpy=True)##值为nan
    sent2_embeddings = model.encode( sent2, show_progress_bar=True, convert_to_numpy=True)

    sent1_sent2 = 1 - (paired_cosine_distances(sent1_embeddings, sent2_embeddings))#余弦距离越小则余弦分数越大，两个句子越相近

    return sent1_sent2#.astype(np.float32)

def get_devSimilarity(data_location,model_location) :#finetune时调用
    df_data=pd.read_csv(data_location,encoding='ISO-8859-1')
    # df_label=pd.read_csv(label_location,keep_default_na=False)
    sentence2=list()
    sentence1_MWE=list()
    id_MWE={}
    index_id={}
    # id_index={}
    for i in range(len(df_data)):
        index_id[df_data['ID'][i]] = i#id->index
        # id_index[i]=df_data['ID'][i]#index->id
        id_MWE[df_data['ID'][i]]=df_data['MWE1'][i]
        sent1=df_data['sentence_1'][i]
        if df_data['MWE1'][i] !='None':
            replaced = sent1.replace( df_data['MWE1'][i], tokenise_idiom(df_data['MWE1'][i]))
            assert replaced != sent1
            sent1 = replaced

        sentence1_MWE.append(sent1)
        # sentence1.append(df_data['sentence_1'][i])
        sentence2.append(df_data['sentence_2'][i])

    sim = get_similarity(model_location,sentence1_MWE,sentence2) #similarity of all sentence pairs
    print(len(sim))
    return sim,index_id

def prepare_data(location) :
    data = pd.read_csv(location,encoding='ISO-8859-1')
    replaced_sentence1 = list()
    index_id={}
    sentences2 = data['sentence_2']
    # print(data.keys()[0])
    if 'ID' in data.keys():
        for index in range(len(data)) :
            sentence1 = str(data['sentence_1'][index])
            mwe = str(data[ 'MWE1'][index])
            index_id[data['ID'][index]] = index#id->index

            if mwe != 'None' :
                # IDMWEID=
                replaced = sentence1.replace( mwe, tokenise_idiom (mwe))#,flags=re.I
                # if replaced==sentence1:
                    # print('replaced:',replaced)
                    # print('sentence1:',sentence1)
                assert replaced != sentence1
                sentence1 = replaced

            replaced_sentence1.append( sentence1 )
    else:
        for index in range(len(data)) :
            sentence1 = data['sentence_1'][index]
            mwe = data[ 'MWE1'][index]

            if mwe != 'None' :
                replaced = re.sub( mwe, tokenise_idiom (mwe), sentence1, flags=re.I)
                assert replaced != sentence1
                sentence1 = replaced

            replaced_sentence1.append( sentence1 )

    return replaced_sentence1, sentences2, index_id

def insert_to_submission(sims,language,submissionLocation) :#,dataLocation ,
    dataFromSubssion=pd.read_csv(submissionLocation,encoding='ISO-8859-1')#submission file
    # dataInsert=pd.read_csv(dataLocation,encoding='ISO-8859-1')#insert_file
    #复制一份language满足条件的submmision
    ID,Language,Setting,Sim=list(),list(),list(),list()
    for i in range(len(dataFromSubssion)):
        if dataFromSubssion['Language'][i]==language:
            ID.append(dataFromSubssion['ID'][i])
            Language.append(dataFromSubssion['Language'][i])
            Setting.append(dataFromSubssion['Setting'][i])
            Sim.append(dataFromSubssion['Sim'][i])
    data=pd.DataFrame({'ID':ID,'Language':Language,'Setting':Setting,'Sim':Sim})
    #插入sim
    for i in range(len(data)):
        # if data['Language'][i]==language:
        data.loc[i,'Sim']=sims[i]
    return data

def insert_to_submission1(sims,id_index,submissionLocation ) :#language,dataLocation ,
    dataFromSubssion=pd.read_csv(submissionLocation,encoding='ISO-8859-1')#submission file
    data=dataFromSubssion.copy()
    assert len(sims)==len(data)
    # if id_index is not None:# dev file
    # assert len(dataInsert)==len(id_index)
    for i in range(len(data)):
        index=id_index[data.loc[i,'ID']]#插入数据ID的索引
        # data.loc[index,'Language']=dataInsert.loc[index,'Language']
        # if data['Language'][index] == language and data['Setting'][index] == settings:
        data.loc[index,'Sim']=sims[index]
    return data

def insert_to_submission3(sims,language,id_index,submissionLocation) :#,dataLocation ,
    dataFromSubssion=pd.read_csv(submissionLocation,encoding='ISO-8859-1')#submission file
    # dataInsert=pd.read_csv(dataLocation,encoding='ISO-8859-1')#insert_file
    #复制一份language满足条件的submmision
    ID,Language,Setting,Sim=list(),list(),list(),list()
    for i in range(len(dataFromSubssion)):
        if dataFromSubssion['Language'][i]==language:
            ID.append(dataFromSubssion['ID'][i])
            Language.append(dataFromSubssion['Language'][i])
            Setting.append(dataFromSubssion['Setting'][i])
            Sim.append(dataFromSubssion['Sim'][i])
    data=pd.DataFrame({'ID':ID,'Language':Language,'Setting':Setting,'Sim':Sim})
    #插入sim
    for i in range(len(data)):
        index=id_index[data.loc[i,'ID']]#插入数据ID的索引
        data.loc[index,'Sim']=sims[index]
    return data

def _score( submission_data, submission_headers, gold_data, gold_headers, language,settings)  :

    # ['ID', 'Language', 'Setting', 'Sim']
    # ['ID', 'DataID', 'Language', 'sim', 'otherID']

    filtered_submission_data = [ i for i in submission_data if i[ submission_headers.index( 'Language' ) ] in language ]#and i[ submission_headers.index( 'Setting' ) ] in settings
    if any( [(i[submission_headers.index('Sim')] == '') for i in filtered_submission_data ] ) :
        return None, None, None

    filtered_submission_dict = dict()
    for elem in filtered_submission_data :
        filtered_submission_dict[ int(elem[submission_headers.index('ID')])] = elem[ submission_headers.index( 'Sim' ) ]

    ## Generate gold
    if len( settings ) > 1 :
        raise Exception( "This script does not work for multiple Settings (Submission IDs not unique)" )
    else :
        gold_data = [ i for i in gold_data if i[ gold_headers.index( 'Language' ) ] in language ]

    gold_labels_all = list()
    predictions_all = list()

    gold_labels_sts = list()
    predictions_sts = list()

    gold_labels_no_sts = list()
    predictions_no_sts = list()

    for elem in gold_data :
        this_sim = elem[ gold_headers.index( 'sim' ) ]
        if this_sim == '':
            this_sim = filtered_submission_dict[ int(float(elem[gold_headers.index('otherID')]))]
        this_sim = float( this_sim )
        this_prediction = float( filtered_submission_dict[ int(elem[gold_headers.index('ID')])])

        gold_labels_all.append( this_sim )#对应句子的预测
        predictions_all.append( this_prediction )#原句句的预测

        if elem[ gold_headers.index( 'DataID' ) ].split( '.' )[2] == 'sts':#
            gold_labels_sts.append( this_sim )
            predictions_sts.append( this_prediction )
        else :
            gold_labels_no_sts.append(this_sim)
            predictions_no_sts.append(this_prediction)

    corel_all, pvalue = spearmanr(gold_labels_all , predictions_all )
    corel_sts, pvalue = spearmanr(gold_labels_sts , predictions_sts )
    corel_no_sts, pvalue = spearmanr(gold_labels_no_sts, predictions_no_sts)
    return ( corel_all, corel_sts, corel_no_sts )


def evaluate_submission( submission_file, gold_labels,settings) :#fine_tune,评估dev集
    submission_headers, submission_data = load_csv( submission_file )
    gold_headers , gold_data = load_csv( gold_labels )

    if submission_headers != ['ID', 'Language', 'Setting', 'Sim'] :
        print( "ERROR: Incorrect submission format", file=sys.stderr )
        sys.exit()
    if gold_headers != ['ID', 'DataID', 'Language', 'sim', 'otherID']:
        print( "ERROR: Incorrect gold labels data format (did you use the correct file?)", file=sys.stderr )
        sys.exit()

    submission_ids = [ int( i[0] ) for i in submission_data]
    gold_ids = [ int( i[0] ) for i in gold_data] + [ int( float(i[-1])) for i in gold_data if i[-1] != '' ]

    for id in submission_ids :
        if not id in gold_ids :
            print( "ERROR: Submission file contains IDs that gold data does not - this could be because you submitted the wrong results (dev results instead of evaluation results) or because your submission file is corrupted", file=sys.stderr )
            sys.exit()

    output = [ [ 'Settings', 'Languages', "Spearman Rank ALL", "Spearman Rank Idiom Data", "Spearman Rank STS Data" ] ]
    for language in [
        # [ [ 'EN' ]      , [ 'pre_train' ] ],
        # [ [ 'PT' ]      , [ 'pre_train' ] ],
        # [ [ 'EN', 'PT' ], [ 'pre_train' ] ],
        [ 'EN' ] ,
        [ 'PT' ],
        ['EN', 'PT'],
    ] :
        corel_all, corel_sts, corel_no_sts = _score( submission_data, submission_headers, gold_data, gold_headers, language,[settings])#[setting]
        this_entry = [ settings, language, corel_all, corel_no_sts, corel_sts ]#所有句子、正面或相关联的句子对、已有相似度
        output.append( this_entry )
    return output

if __name__ == '__main__':
    ALLdata='../data/train_data.csv'
    model_location='../model/model_with_MWE/distiluse-base-multilingual-cased-v1'
    out_location='../data/train_data1.csv'

    alter1,alter2=get_alter_sentence(ALLdata)
    sim_ALL=get_similarity(model_location,alter1, alter2)

    out_train_file(ALLdata,sim_ALL).to_csv(out_location)
