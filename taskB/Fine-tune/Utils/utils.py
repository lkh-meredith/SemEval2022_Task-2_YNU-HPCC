import os
import csv
from pathlib import Path

def load_csv( path, delimiter=',' ) :
  header = None
  data   = list()
  with open( path, encoding='utf-8') as csvfile:
    reader = csv.reader( csvfile, delimiter=delimiter )
    for row in reader :
      if header is None :
        header = row
        continue
      data.append( row )
  return header, data

def write_csv( data, location ) :
  with open( location, 'w', encoding='utf-8',newline='') as csvfile:
    writer = csv.writer( csvfile )
    writer.writerows( data )
  print( "Wrote {}".format( location ) )
  return

def insert_to_submission_file( submission_format_file, input_file, prediction_format_file, setting ) :
    submission_header, submission_content = load_csv( submission_format_file )
    input_header     , input_data         = load_csv( input_file             )
    prediction_header, prediction_data    = load_csv( prediction_format_file, '\t' )

    assert len( input_data ) == len( prediction_data )

    ## submission_header ['ID', 'Language', 'Setting', 'Label']
    ## input_header      ['label', 'sentence1' ]
    ## prediction_header ['index', 'prediction']

    prediction_data = list( reversed( prediction_data ) )

    started_insert  = False
    for elem in submission_content :
        if elem[ submission_header.index( 'Setting' ) ] != setting :
            if started_insert :
                if len( prediction_data ) == 0 :
                    break
                else :
                    raise Exception( "Update should to contiguous ... something wrong." )
            continue
        started_insert = True
        elem[ submission_header.index( 'Label' ) ] = prediction_data.pop()[ prediction_header.index( 'prediction' ) ]

    return [ submission_header ] + submission_content


def _get_train_data(data_location, file_name, include_context, include_idiom):
    file_name = os.path.join(data_location, file_name)

    header, data = load_csv(file_name)

    out_header = ['label', 'sentence1']
    if include_idiom:
        out_header = ['label', 'sentence1', 'sentence2']

    # ['DataID', 'Language', 'MWE', 'Setting', 'Previous', 'Target', 'Next', 'Label']
    out_data = list()
    for elem in data:
        label = elem[header.index('Label')]
        sentence1 = elem[header.index('Target')]
        if include_context:
            sentence1 = ' '.join(
                [elem[header.index('Previous')], elem[header.index('Target')], elem[header.index('Next')]])
        this_row = None
        if not include_idiom:
            this_row = [label, sentence1]
        else:
            sentence2 = elem[header.index('MWE')]
            this_row = [label, sentence1, sentence2]
        out_data.append(this_row)
        assert len(out_header) == len(this_row)
    return [out_header] + out_data


def _get_dev_eval_data(data_location, input_file_name, gold_file_name, include_context, include_idiom):
    input_headers, input_data = load_csv(os.path.join(data_location, input_file_name))
    gold_header = gold_data = None
    if not gold_file_name is None:
        gold_header, gold_data = load_csv(os.path.join(data_location, gold_file_name))
        assert len(input_data) == len(gold_data)

    # ['ID', 'Language', 'MWE', 'Previous', 'Target', 'Next']
    # ['ID', 'DataID', 'Language', 'Label']

    out_header = ['label', 'sentence1']
    if include_idiom:
        out_header = ['label', 'sentence1', 'sentence2']

    out_data = list()
    for index in range(len(input_data)):
        label = 1
        if not gold_file_name is None:
            this_input_id = input_data[index][input_headers.index('ID')]
            this_gold_id = gold_data[index][gold_header.index('ID')]
            assert this_input_id == this_gold_id

            label = gold_data[index][gold_header.index('Label')]

        elem = input_data[index]
        sentence1 = elem[input_headers.index('Target')]
        if include_context:
            sentence1 = ' '.join([elem[input_headers.index('Previous')], elem[input_headers.index('Target')],
                                  elem[input_headers.index('Next')]])
        this_row = None
        if not include_idiom:
            this_row = [label, sentence1]
        else:
            sentence2 = elem[input_headers.index('MWE')]
            this_row = [label, sentence1, sentence2]
        assert len(out_header) == len(this_row)
        out_data.append(this_row)

    return [out_header] + out_data


def create_data(input_location, output_location):
    ## Zero shot data
    train_data = _get_train_data(
        data_location=input_location,
        file_name='train_zero_shot.csv',
        include_context=True,
        include_idiom=False
    )
    write_csv(train_data, os.path.join(output_location, 'ZeroShot', 'train.csv'))

    dev_data = _get_dev_eval_data(
        data_location=input_location,
        input_file_name='dev.csv',
        gold_file_name='dev_gold.csv',
        include_context=True,
        include_idiom=False
    )
    write_csv(dev_data, os.path.join(output_location, 'ZeroShot', 'dev.csv'))

    eval_data = _get_dev_eval_data(
        data_location=input_location,
        input_file_name='eval.csv',
        gold_file_name=None,  ## Don't have gold evaluation file -- submit to CodaLab
        include_context=True,
        include_idiom=False
    )
    write_csv(eval_data, os.path.join(output_location, 'ZeroShot', 'eval.csv'))

    ## OneShot Data (combine both for training)
    train_zero_data = _get_train_data(
        data_location=input_location,
        file_name='train_zero_shot.csv',
        include_context=False,
        include_idiom=True
    )
    train_one_data = _get_train_data(
        data_location=input_location,
        file_name='train_one_shot.csv',
        include_context=False,
        include_idiom=True
    )

    train_one_data_from_test=_get_train_data(
        data_location='../SubTaskA_data/TestData',
        file_name='train_one_shot.csv',
        include_context=False,
        include_idiom=True
    )

    assert train_zero_data[0] == train_one_data[0]==train_one_data_from_test[0] ## Headers
    train_data = train_one_data +train_one_data_from_test+ train_zero_data[1:]
    write_csv(train_data, os.path.join(output_location, 'OneShot', 'train.csv'))

    dev_data = _get_dev_eval_data(
        data_location=input_location,
        input_file_name='dev.csv',
        gold_file_name='dev_gold.csv',
        include_context=False,
        include_idiom=True
    )
    write_csv(dev_data, os.path.join(output_location, 'OneShot', 'dev.csv'))

    eval_data = _get_dev_eval_data(
        data_location=input_location,
        input_file_name='eval.csv',
        gold_file_name=None,
        include_context=False,
        include_idiom=True
    )
    write_csv(eval_data, os.path.join(output_location, 'OneShot', 'eval.csv'))

    return


import sys
import csv
from sklearn.metrics import f1_score

def _score(submission_data, submission_headers, gold_data, gold_headers, languages, settings):
    if len(settings) == 2:
        assert len(languages) == 2
        gold_data = gold_data + gold_data

    filtered_submission_data = [i for i in submission_data if
                                i[submission_headers.index('Language')] in languages and i[
                                    submission_headers.index('Setting')] in settings]
    submission_ids = [int(i[submission_headers.index('ID')]) for i in filtered_submission_data]

    filtered_gold_data = [i for i in gold_data if int(i[gold_headers.index('ID')]) in submission_ids]

    gold_ids = [int(i[0]) for i in filtered_gold_data]

    if submission_ids != gold_ids:
        print("ERROR: IDs in Submission file do not match IDs in gold file!")
        sys.exit()

    y_pred = [i[submission_headers.index('Label')] for i in filtered_submission_data]
    y_true = [i[gold_headers.index('Label')] for i in filtered_gold_data]

    if any([(i == '') for i in y_pred]):
        return None, None, None

    y_pred = [int(i) for i in y_pred]
    y_true = [int(i) for i in y_true]
    f1_macro = f1_score(y_true, y_pred, average='macro')

    return f1_macro


def evaluate_submission(submission_file, gold_labels):
    submission_headers, submission_data = load_csv(submission_file)
    gold_headers, gold_data = load_csv(gold_labels)

    if submission_headers != ['ID', 'Language', 'Setting', 'Label']:
        print("ERROR: Incorrect submission format", file=sys.stderr)
        sys.exit()

    if gold_headers != ['ID', 'DataID', 'Language', 'Label']:
        print("ERROR: Incorrect gold labels data format (did you use the correct file?)", file=sys.stderr)
        sys.exit()

    output = [['Settings', 'Languages', "F1 Score (Macro)"]]

    for languages, settings in [
        [['EN'], ['zero_shot']],
        [['PT'], ['zero_shot']],
        [['EN', 'PT'], ['zero_shot']],

        [['EN'], ['one_shot']],
        [['PT'], ['one_shot']],
        [['EN', 'PT'], ['one_shot']],

    ]:
        f1_macro = _score(submission_data, submission_headers, gold_data, gold_headers, languages, settings)
        this_entry = [','.join(settings), ','.join(languages), f1_macro]
        output.append(this_entry)

    return output

if __name__=="__main__":
    # outpath=os.path.join('..','data')
    # Path(os.path.join(outpath, 'ZeroShot')).mkdir(parents=True, exist_ok=True)
    # Path(os.path.join(outpath, 'OneShot')).mkdir(parents=True, exist_ok=True)
    #
    # create_data('../SubTaskA_data/Data', outpath)

    #test.csv
    #one_shot
    Oneshot_test_data = _get_dev_eval_data(
        data_location='../SubTaskA_data/TestData',
        input_file_name='test.csv',
        gold_file_name=None,  ## Don't have gold evaluation file -- submit to CodaLab
        include_context=False,
        include_idiom=True
    )
    write_csv(Oneshot_test_data, os.path.join('..', 'data', 'OneShot', 'test.csv'))

    # zero_shot
    Zeroshot_test_data = _get_dev_eval_data(
        data_location='../SubTaskA_data/TestData',
        input_file_name='test.csv',
        gold_file_name=None,  ## Don't have gold evaluation file -- submit to CodaLab
        include_context=True,
        include_idiom=False
    )
    write_csv(Zeroshot_test_data, os.path.join('..', 'data', 'ZeroShot', 'test.csv'))