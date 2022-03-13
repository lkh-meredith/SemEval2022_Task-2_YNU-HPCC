import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from Utils.utils import get_similarity,prepare_data,write_csv,insert_to_submission1


model_path='../model/modelSave/baseline_of_fine-tune/4/1'
test_location = os.path.join('..', 'data', 'test.csv')
test_formated_file_location = '../submission/test_submission_format_fineTune.csv'
evalSen1, evalSen2, evalId_index = prepare_data(test_location)
eval_sim = get_similarity(model_path, evalSen1, evalSen2)

submission_data = insert_to_submission1(eval_sim, evalId_index, test_formated_file_location)
results_file = os.path.join('..', 'submission','baseline_of_fine-tune', 'test.combined_results-' + str(4) + '_' + str(1) + '_'  + str(16) + '.csv')
# submission_data.to_csv(results_file,index=False)
write_csv(submission_data, results_file)