from transformers import AutoTokenizer,AutoModelForMaskedLM,AutoModel
from Utils.utils import tokenise_idiom
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

def updateVocab(model_checkpoint,idioms,outdir):#,outdir
    # model = AutoModel.from_pretrained(model_checkpoint)
    model = SentenceTransformer(model_checkpoint)
    print(model)
    # model = AutoModel.from_pretrained(model_checkpoint)#simCSE
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, truncation=True,do_lower_case=False)#, use_fast=False
    old_len = len( tokenizer )
    num_added_toks = tokenizer.add_tokens( idioms )
    print( "Old tokenizer length was {}. Added {} new tokens. New length is {}.".format( old_len, num_added_toks, len( tokenizer ) )  )
    model.resize_token_embeddings(len(tokenizer))

    model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)

    # word_embedding_model = model._first_module()
    # word_embedding_model.tokenizer.add_tokens(idioms)
    # word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
    # print(word_embedding_model.tokenizer('This is a IDhighlifeID'))
    #
    # # word_embedding_model.save(outdir)
    print("success saving")

def extract_idiom(data_location):
    idioms = list()
    data = pd.read_csv(data_location,encoding='ISO-8859-1')
    for index in range(len(data)) :
        idioms.append(data['MWE1'][index])

    idioms = list(set(idioms))
    if 'None' in idioms:
        idioms.remove('None')
    print( "Found a total of {} idioms".format(len(idioms)))

    idioms = [ tokenise_idiom( i ) for i in idioms ]
    return idioms

if __name__=="__main__":

    #sentenceTranceformer
    # model = SentenceTransformer("sentence-transformers/bert-base-multilingual-cased")
    model_name='distiluse-base-multilingual-cased-v1'
    # model_name="bert-base-multilingual-cased"
    model_checkpoint = os.path.join('..','model','SentenceTransformer',model_name)
    dataLocation=os.path.join('..','data')

    # # download model
    # model = AutoModelForMaskedLM.from_pretrained(model_name)
    # model.save_pretrained( model_checkpoint, _use_new_zipfile_serialization=False)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, truncation=True)
    # tokenizer.save_pretrained(model_checkpoint)
    # print( "Wrote to: ", model_checkpoint)

    #extract idiom from train, dev, eval data
    idiomFromTrain=extract_idiom(os.path.join(dataLocation,'train_data.csv'))
    idiomFromDev=extract_idiom(os.path.join(dataLocation,'dev.csv'))
    idiomFromEval=extract_idiom(os.path.join(dataLocation,'eval.csv'))
    idiomFromTest=extract_idiom(os.path.join(dataLocation,'test.csv'))

    idioms=idiomFromTrain+idiomFromDev+idiomFromEval+idiomFromTest
    savePath=os.path.join('..','model','model_with_MWE',model_name+'_dense')#添加MWE后的model
    updateVocab(model_checkpoint,idioms,savePath)#,savePath

