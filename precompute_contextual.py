import json
import logging

import numpy as np
import spacy

from pytt_hf import toks2vecs
from pytt_hf import PYTT_CONFIG

from umls import umls_kb_st21pv as umls_kb


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def read_mm_converted(mm_set_path):

    with open(mm_set_path, 'r') as json_f:
        mm_set = json.load(json_f)

    return list(mm_set['docs'])



if __name__ == '__main__':

    split_label = 'train'
    # split_label = 'dev'
    # train_data = list(iterate_docs_converted('data/MedMentions/mm_converted.%s.json' % split_label))
    mm_docs = read_mm_converted('data/MedMentions/st21pv/custom/mm_converted.%s.json' % split_label)

    all_embeddings_info = []

    for doc_idx, doc in enumerate(mm_docs):

        logging.info('At doc %d - %d embeddings' % (doc_idx, len(all_embeddings_info)))

        for sent_idx, sent in enumerate(doc['sentences']):

            sent['ctx_vecs'] = toks2vecs(sent['tokens'], return_tokens=False)
            assert len(sent['ctx_vecs']) == len(sent['tokens'])

            # for ent_idx, ent in enumerate(sent['entities']):
            for span_idx, span in enumerate(sent['spans']):
                
                span_vecs = [sent['ctx_vecs'][i] for i in range(span['start'], span['end'])]
                span_vec = np.array(span_vecs, dtype=np.float32).mean(axis=0)

                if np.isnan(span_vec.sum()) or span_vec.sum() == 0:
                    continue                
                
                emb_info = (doc_idx, sent_idx, span_idx, span['cui'], span['st'], span_vec)

                all_embeddings_info.append(emb_info)


    logging.info('Writing Embeddings ...')
    vecs_path = 'mm_st21pv.%s.%s.precomputed' % (split_label, PYTT_CONFIG['name'])
    with open(vecs_path, 'w') as vecs_f:
        for (doc_idx, sent_idx, ent_idx, ent_cui, ent_st, ent_vec) in all_embeddings_info:
            ent_vec_str = ' '.join([str(round(v, 6)) for v in ent_vec.tolist()])
            vecs_f.write('%d\t%d\t%d\t%s\t%s\t%s\n' % (doc_idx, sent_idx, ent_idx, ent_cui, ent_st, ent_vec_str))
    logging.info('Written %s' % vecs_path)

