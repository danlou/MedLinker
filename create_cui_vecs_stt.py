import logging
import numpy as np

from umls import umls_kb_st21pv as umls_kb


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


logging.info('Loading scispacy (lg) ...')
import spacy
scispacy_version = 'en_core_sci_lg'
sci_nlp = spacy.load(scispacy_version, disable=['tagger', 'parser', 'ner'])


def alias2vec(sci_nlp, alias):
    toks_vecs = [t.vector for t in sci_nlp(alias)]
    return np.array(toks_vecs).mean(axis=0)


logging.info('Embedding aliases ...')
cui_vecs = {}
for cui_idx, cui in enumerate(umls_kb.get_all_cuis()):

    # if cui_idx > 0:
    if cui_idx % 1000 == 0:
        logging.info('At #CUI: %d/%d' % (cui_idx, len(umls_kb.umls_data)))

    cui_aliases_vecs = []
    for alias in umls_kb.get_aliases(cui, include_name=True):
        alias_vec = alias2vec(sci_nlp, alias)
        cui_aliases_vecs.append(alias_vec)

    cui_vecs[cui] = np.array(cui_aliases_vecs).mean(axis=0)


logging.info('Writing vecs ...')
vecs_path = '%s.%s.cuis.vecs' % (umls_kb.umls_version, scispacy_version)
with open(vecs_path, 'w') as vecs_f:
    for cui, vec in cui_vecs.items():
        vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
        vecs_f.write('%s %s\n' % (cui, vec_str))
logging.info('Written %s' % vecs_path)











"""

import logging

import numpy as np
import spacy

from scispacy_medmentions_reader import read_full_med_mentions
from scispacy_medmentions_reader import iterate_annotations

from umls_utils import sci_nlp
from umls_utils import cui2ent
from umls_utils import cui2st


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def get_static_vecs(sci_nlp, spans):
    span_vecs = []
    for doc in sci_nlp.pipe(spans, batch_size=1000, disable=['tagger', 'parser', 'ner']):
        span_vecs.append((doc.text, doc.vector))
    return span_vecs


if __name__ == '__main__':

    batch_size = 1000  # batch processing is WIP

    logging.info('Loading MedMentions ...')
    train_examples, _, _ = read_full_med_mentions('data/MedMentions/full/data/')

    skipped_anns = 0
    concept_vecs = {}
    st_ann_vecs = {}  # pooled over all annotations belonging to the same ST
    batch = []

    annotations = list(iterate_annotations(sci_nlp, train_examples))

    logging.info('Processing annotations ...')
    for ann_idx, (ent, sent) in enumerate(annotations):
        batch.append(ent)

        if (len(batch) == batch_size) or (ann_idx == len(annotations) - 1):

            batch_spans = [ent.mention_text for ent in batch]
            batch_vecs = get_static_vecs(sci_nlp, batch_spans)

            for mention_idx, ent in enumerate(batch):
                processed_span, span_vec = batch_vecs[mention_idx]

                if np.isnan(span_vec.sum()) or span_vec.sum() == 0:  # failed due to vocab limitations ?
                    continue

                if ent.umls_id in concept_vecs:
                    concept_vecs[ent.umls_id]['vecs_sum'] += span_vec
                    concept_vecs[ent.umls_id]['vecs_num'] += 1
                else:
                    concept_vecs[ent.umls_id] = {'vecs_sum': span_vec, 'vecs_num': 1}

                if ent.mention_type in st_ann_vecs:
                    st_ann_vecs[ent.mention_type]['vecs_sum'] += span_vec
                    st_ann_vecs[ent.mention_type]['vecs_num'] += 1
                else:
                    st_ann_vecs[ent.mention_type] = {'vecs_sum': span_vec, 'vecs_num': 1}

            batch = []

        if ann_idx % 100 == 0:
            logging.info('#Annotations:%d #Concepts:%d #Types:%d #Skipped Ann.:%d' % (ann_idx, len(concept_vecs), len(st_ann_vecs), skipped_anns))

    logging.info('Skipped %d annotations' % skipped_anns)


    logging.info('Writing Concept Vectors ...')
    vecs_path = 'medmentions.concepts.%s.vecs' % sci_nlp.meta['name']
    with open(vecs_path, 'w') as vecs_f:
        for cui, vecs_info in concept_vecs.items():
            vecs_info['vecs_avg'] = vecs_info['vecs_sum'] / vecs_info['vecs_num']
            vec_str = ' '.join([str(round(v, 6)) for v in vecs_info['vecs_avg'].tolist()])
            vecs_f.write('%s %s\n' % (cui, vec_str))
    logging.info('Written %s' % vecs_path)


    logging.info('Writing ST Vectors (pooled all annotations) ...')
    vecs_path = 'medmentions.sts_anns.%s.vecs' % sci_nlp.meta['name']
    with open(vecs_path, 'w') as vecs_f:
        for st, vecs_info in st_ann_vecs.items():
            vecs_info['vecs_avg'] = vecs_info['vecs_sum'] / vecs_info['vecs_num']
            vec_str = ' '.join([str(round(v, 6)) for v in vecs_info['vecs_avg'].tolist()])
            vecs_f.write('%s %s\n' % (st, vec_str))
    logging.info('Written %s' % vecs_path)


    logging.info('Writing ST Vectors (pooled all concepts) ...')
    # computing ST embeddings from precomputed concept embeddings    
    st_cpt_vecs = {}  # pooled over all concept vecs belonging to the same ST
    for cui, vecs_info in concept_vecs.items():
        cui_vec = vecs_info['vecs_avg']
        st = cui2st(cui)

        if st is None:  # cui not in KB?
            continue
        
        elif st in st_cpt_vecs:
            st_cpt_vecs[st]['vecs_sum'] += cui_vec
            st_cpt_vecs[st]['vecs_num'] += 1
        
        else:
            st_cpt_vecs[st] = {'vecs_sum': cui_vec, 'vecs_num': 1}

    vecs_path = 'medmentions.sts_cpts.%s.vecs' % sci_nlp.meta['name']
    with open(vecs_path, 'w') as vecs_f:
        for st, vecs_info in st_cpt_vecs.items():
            vecs_info['vecs_avg'] = vecs_info['vecs_sum'] / vecs_info['vecs_num']
            vec_str = ' '.join([str(round(v, 6)) for v in vecs_info['vecs_avg'].tolist()])
            vecs_f.write('%s %s\n' % (st, vec_str))
    logging.info('Written %s' % vecs_path)
"""