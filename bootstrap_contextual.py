import json
import logging

import numpy as np

from pytt_hf import toks2vecs
from pytt_hf import PYTT_CONFIG

# from umls_utils import sci_nlp
# from umls_utils import cui2ent
# from umls_utils import cui2st

from umls import umls_kb_st21pv as umls_kb


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def iterate_docs_converted(split_path):

    # load json dataset
    with open(split_path, 'r') as json_f:
        dataset = json.load(json_f)

    for doc in dataset['docs']:
        yield doc


def get_ctx_vec(sent_ctx_out, sent_tokens, span_idx_start, span_idx_end, normalize=False):
    span_toks = sent_tokens[span_idx_start:span_idx_end]
    span_ctx_out = sent_ctx_out[span_idx_start:span_idx_end]

    span_ctx_toks = [t for t, v in span_ctx_out]

    # sanity check - prob. unnecessary ...
    assert span_ctx_toks == span_toks

    span_ctx_vecs = [v for t, v in span_ctx_out]
    span_ctx_vec = np.array(span_ctx_vecs).mean(axis=0)
    
    if normalize:
        span_ctx_vec = span_ctx_vec / np.linalg.norm(span_ctx_vec)

    return span_ctx_vec


if __name__ == '__main__':

    # train_docs = list(iterate_docs_converted('data/MedMentions/full/custom/mm_converted.train.json'))
    train_docs = list(iterate_docs_converted('data/MedMentions/st21pv/custom/mm_converted.train.json'))

    skipped_anns = 0
    concept_vecs = {}
    st_ann_vecs = {}  # pooled over all annotations belonging to the same ST

    for doc_idx, doc in enumerate(train_docs):

        logging.info('#Docs:%d #Concepts:%d #Types:%d #Skipped Ann.:%d' % (doc_idx, len(concept_vecs), len(st_ann_vecs), skipped_anns))

        if doc_idx == 10:
            break

        for sent in doc['sentences']:

            sent_ctx_out = toks2vecs(sent['tokens'])

            for ent in sent['spans']:
                ent['cui'] = ent['cui'].lstrip('UMLS:')
                span_ctx_vec = get_ctx_vec(sent_ctx_out, sent['tokens'], ent['start'], ent['end'], normalize=False)

                if np.isnan(span_ctx_vec.sum()) or span_ctx_vec.sum() == 0:
                    continue

                if np.sum(span_ctx_vec) == 0:  # beyond max_seq_len
                    skipped_anns += 1
                    continue

                if ent['cui'] in concept_vecs:
                    concept_vecs[ent['cui']]['vecs_sum'] += span_ctx_vec
                    concept_vecs[ent['cui']]['vecs_num'] += 1
                else:
                    concept_vecs[ent['cui']] = {'vecs_sum': span_ctx_vec, 'vecs_num': 1}

                if ent['st'] in st_ann_vecs:
                    st_ann_vecs[ent['st']]['vecs_sum'] += span_ctx_vec
                    st_ann_vecs[ent['st']]['vecs_num'] += 1
                else:
                    st_ann_vecs[ent['st']] = {'vecs_sum': span_ctx_vec, 'vecs_num': 1}

    logging.info('Skipped %d annotations' % skipped_anns)

    logging.info('Writing Concept Vectors ...')
    # vecs_path = 'mm_full.cuis.%s.vecs' % PYTT_CONFIG['name']
    vecs_path = 'mm_st21pv.cuis.%s.vecs' % PYTT_CONFIG['name']
    with open(vecs_path, 'w') as vecs_f:
        for cui, vecs_info in concept_vecs.items():
            vecs_info['vecs_avg'] = vecs_info['vecs_sum'] / vecs_info['vecs_num']
            vec_str = ' '.join([str(round(v, 6)) for v in vecs_info['vecs_avg'].tolist()])
            vecs_f.write('%s %s\n' % (cui, vec_str))
    logging.info('Written %s' % vecs_path)


    logging.info('Writing ST Vectors (pooled all annotations) ...')
    # vecs_path = 'mm_full.sts_anns.%s.vecs' % PYTT_CONFIG['name']
    vecs_path = 'mm_st21pv.sts_anns.%s.vecs' % PYTT_CONFIG['name']
    with open(vecs_path, 'w') as vecs_f:
        for st, vecs_info in st_ann_vecs.items():
            vecs_info['vecs_avg'] = vecs_info['vecs_sum'] / vecs_info['vecs_num']
            vec_str = ' '.join([str(round(v, 6)) for v in vecs_info['vecs_avg'].tolist()])
            vecs_f.write('%s %s\n' % (st, vec_str))
    logging.info('Written %s' % vecs_path)


    logging.info('Writing ST Vectors (pooled all concepts) ...')
    # computing ST embeddings from precomputed concept embeddings    
    st_cpt_vecs = {}  # pooled over all concept vecs belonging to the same ST
    missing_cuis = set()
    for cui, vecs_info in concept_vecs.items():
        cui_vec = vecs_info['vecs_avg']
        
        try:
            for st in umls_kb.get_sts(cui):
                if st in st_cpt_vecs:
                    st_cpt_vecs[st]['vecs_sum'] += cui_vec
                    st_cpt_vecs[st]['vecs_num'] += 1
                else:
                    st_cpt_vecs[st] = {'vecs_sum': cui_vec, 'vecs_num': 1}
        except KeyError:
            missing_cuis.add(cui)
    if len(missing_cuis) > 0:
        print('WARNING: %d CUIs not covered in umls_kb' % len(missing_cuis))

    # vecs_path = 'mm_full.sts_cpts.%s.vecs' % PYTT_CONFIG['name']
    vecs_path = 'mm_st21pv.sts_cpts.%s.vecs' % PYTT_CONFIG['name']
    with open(vecs_path, 'w') as vecs_f:
        for st, vecs_info in st_cpt_vecs.items():
            vecs_info['vecs_avg'] = vecs_info['vecs_sum'] / vecs_info['vecs_num']
            vec_str = ' '.join([str(round(v, 6)) for v in vecs_info['vecs_avg'].tolist()])
            vecs_f.write('%s %s\n' % (st, vec_str))
    logging.info('Written %s' % vecs_path)
