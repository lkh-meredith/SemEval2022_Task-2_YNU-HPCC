from transformers.models.bert import BertTokenizer
from utils import evaluate,prepare_data,get_similarity,get_devSimilarity,write_csv,evaluate_submission,insert_to_submission1,writeList_csv,set_seed
# from Utils.utils import get_similarity
from pretrain_coSENT import Model
import torch
import os

# model_path='../model/SentenceTransformer/MySentTransformersa/MySBERTa_bert-base-multilingual-cased_16_1'
model_path='../model/modelSave/pretrain_CoSENT/last-avg/bert-base-multilingual-cased_epoch_7_batch_16.bin'
#CoSENT
model=Model()
model.load_state_dict(torch.load(model_path))
tokenizer = BertTokenizer.from_pretrained('../model/model_with_MWE/bert-base-multilingual-cased')

test_location = os.path.join('..', 'data', 'test.csv')
test_formated_file_location = '../submission/test_submission_format_pretrain.csv'

model.eval()
evalSen1, evalSen2, evalId_index = prepare_data(test_location)

# eval_sim = get_similarity(model_path, evalSen1, evalSen2)

eval_sim = get_similarity(evalSen1, evalSen2, model.cuda(), tokenizer,'last-avg')

submission_data = insert_to_submission1(eval_sim, evalId_index, test_formated_file_location)
results_file = os.path.join('..','submission','pretrain','coSENT', 'last-avg',
                            'test.combined_results-' + str(7) + '_' + str(16) + '.csv')
write_csv(submission_data, results_file)

# eval_location = os.path.join('..', 'data', 'eval.csv')
# eval_formated_file_location = '../submission/eval.submission_format_pretrain.csv'
#
# model.eval()
# evalSen1, evalSen2, evalId_index = prepare_data(eval_location)
# eval_sim = get_similarity(evalSen1, evalSen2, model.cuda(), tokenizer,'first-last-avg')
#
# submission_data = insert_to_submission1(eval_sim, evalId_index, eval_formated_file_location)
# results_file = os.path.join('..', 'submission', 'pretrain', 'coSENT', 'first-last-avg',
#                             'eval.combined_results-' + 'test' + '.csv')
# write_csv(submission_data, results_file)