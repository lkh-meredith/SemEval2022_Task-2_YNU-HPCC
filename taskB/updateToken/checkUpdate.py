from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

model_location='../model/model_with_MWE/distiluse-base-multilingual-cased-v1'
# model_location='../model/SentenceTransformer/distiluse-base-multilingual-cased-v1'

tokenizer = AutoTokenizer.from_pretrained(
    model_location ,
    use_fast = False ,
    max_length = 510 ,
    force_download = True,
    add_special_tokens = False,
    do_lower_case=False
)

#test tokenizer
some_pass = False
if tokenizer.tokenize('This is a IDhighlifeID?')[-2] == 'IDhighlifeID' :#/
    print( tokenizer.tokenize('This is a IDhighlifeID?'))
    print(tokenizer.encode('This is a IDhighlifeID?'))
    # some_pass = True

if tokenizer.tokenize('This is a IDbra?odireitoID')[-1] =='IDbra?odireitoID':
    print(tokenizer.tokenize('This is a IDbra?odireitoID?'))
    print(tokenizer.encode('This is a IDbra?odireitoID?'))
# some_pass = False
if tokenizer.tokenize('This is a IDpastoralem?oID')[-1] =='IDpastoralem?oID':
    print(tokenizer.tokenize('This is a IDpastoralem?oID'))
    print(tokenizer.encode('This is a IDpastoralem?oID'))

if tokenizer.tokenize('This is a IDclosecallID')[-1] == 'IDclosecallID':
    print(tokenizer.tokenize( 'This is a IDclosecallID' ) )
    print(tokenizer.encode( 'This is a IDclosecallID'))
    some_pass = True
assert some_pass
print( "All good")
