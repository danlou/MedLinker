import torch as th
from  pytorch_transformers import BertModel, BertTokenizer

PYTT_CONFIG = {}
# PYTT_CONFIG['external'] = False
# PYTT_CONFIG['lower_case'] = False
# PYTT_CONFIG['name'] = 'bert-large-cased'
# PYTT_CONFIG['path'] = None

PYTT_CONFIG['external'] = True
PYTT_CONFIG['lower_case'] = True
PYTT_CONFIG['name'] = 'scibert_scivocab_uncased'
PYTT_CONFIG['path'] = 'models/BERT/scibert_scivocab_uncased'

# PYTT_CONFIG['external'] = True
# PYTT_CONFIG['lower_case'] = True
# PYTT_CONFIG['name'] = 'NCBI_BERT_pubmed_uncased_large'
# PYTT_CONFIG['path'] = '/mnt/E68C68028C67CC1D/projects/umls-contextual-linker/data/NCBI_BERT_pubmed_uncased_large'

if PYTT_CONFIG['external']:
    pytt_tokenizer = BertTokenizer.from_pretrained(PYTT_CONFIG['path'], do_lower_case=PYTT_CONFIG['lower_case'])
    pytt_model = BertModel.from_pretrained(PYTT_CONFIG['path'], output_hidden_states=True, output_attentions=True)
else:
    pytt_tokenizer = BertTokenizer.from_pretrained(PYTT_CONFIG['name'], do_lower_case=PYTT_CONFIG['lower_case'])
    pytt_model = BertModel.from_pretrained(PYTT_CONFIG['name'], output_hidden_states=True, output_attentions=True)

pytt_model.eval()
pytt_model.to('cuda')


def get_num_features(tokens):
    return len(sum([pytt_tokenizer.encode(t) for t in tokens], [])) + 2  # plus CLS and SEP


def toks2vecs(tokens, layers=[-1, -2, -3, -4], subword_op='avg', layer_op='sum', return_tokens=True):
    
    # if PYTT_CONFIG['lower_case']:
    #     tokens = [t.lower() for t in tokens]

    encoding_map = [pytt_tokenizer.encode(t) for t in tokens]
    sent_encodings = sum(encoding_map, [])
    sent_encodings = pytt_tokenizer.encode(pytt_tokenizer.cls_token) + \
                     sent_encodings + \
                     pytt_tokenizer.encode(pytt_tokenizer.sep_token)

    input_ids = th.tensor([sent_encodings]).to('cuda')
    all_hidden_states, all_attentions = pytt_model(input_ids)[-2:]

    all_hidden_states = sum([all_hidden_states[i] for i in layers])
    all_hidden_states = all_hidden_states[0]  # batch size 1
    all_hidden_states = all_hidden_states[1:-1]  # ignoring CLS and SEP

    # align and merge subword embeddings (average)
    tok_embeddings = []
    encoding_idx = 0
    for tok, tok_encodings in zip(tokens, encoding_map):
        tok_embedding = th.zeros(pytt_model.config.hidden_size).to('cuda')
        for tok_encoding in tok_encodings:
            tok_embedding += all_hidden_states[encoding_idx]
            encoding_idx += 1
        tok_embedding = tok_embedding / len(tok_encodings)  # avg of subword embs
        tok_embedding = tok_embedding.detach().cpu().numpy()

        if return_tokens:
            tok_embeddings.append((tok, tok_embedding))
        else:
            tok_embeddings.append(tok_embedding)
    
    return tok_embeddings


if __name__ == '__main__':

    sent = "Hello World !"
    sent_embeddings = toks2vecs(sent.split())
    
