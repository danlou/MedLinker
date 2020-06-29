import logging
import pickle
import joblib
from functools import partial
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def create_alias_vecs():

    from ext_umls_utils import cui2ent
    from ext_umls_utils import all_cuis

    # preprocessing aliases and labels
    logging.info('Preprocessing aliases ...')
    all_aliases = set()
    alias_mapping = defaultdict(set)

    for cui in all_cuis:
        ent = cui2ent(cui)
        if ent is None:
            continue
        
        # clean strings to ensure we can use tab separators
        cui_aliases = set([a.lower().replace('\t', ' ') for a in ent.aliases])
        cui_aliases.add(ent.canonical_name.lower().replace('\t', ' '))

        for alias in cui_aliases:
            alias_mapping[alias].add(cui)

        all_aliases.update(cui_aliases)
    
    all_aliases = list(all_aliases)

    logging.info('Tokenizing aliases ...')
    sci_nlp_pipe = partial(sci_nlp.pipe, batch_size=10000, disable=['tagger', 'parser', 'ner'])
    
    all_aliases_tokenized = []
    for alias_idx, alias_doc in enumerate(sci_nlp_pipe(all_aliases)):
        all_aliases_tokenized.append([t.text for t in alias_doc][:64])  # some aliases have too many tokens

        if alias_idx % 100000 == 0:
            logging.info('Tokenization at %d/%d' % (alias_idx, len(all_aliases)))
    
    logging.info('Generating alias vecs ...')

    all_aliases_vecs = {}

    # # load precomputed embeddings (for continuing interrupted process ...)
    # with open('umls_aliases_ctx.NCBI_BERT.vecs.p', 'rb') as vecs_pf:
    #     all_aliases_vecs = pickle.load(vecs_pf)

    for alias_idx, alias_toks in enumerate(all_aliases_tokenized):
        if alias_idx % 100000 == 0:
            logging.info('Embeddings at %d/%d' % (alias_idx, len(all_aliases)))
        
        alias = all_aliases[alias_idx]

        try:
            all_aliases_vecs[alias]
            continue
        except KeyError:
            pass

        alias_vecs = [vec for tok, vec in toks2vecs(alias_toks)]
        all_aliases_vecs[alias] = np.array(alias_vecs).mean(axis=0)

    logging.info('Storing alias vecs ...')
    with open('umls_aliases_ctx.NCBI_BERT.vecs', 'w') as vecs_f:
        for alias, vec in all_aliases_vecs.items():
            vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
            vecs_f.write('%s\t%s\n' % (alias, vec_str))

    logging.info('Storing alias mapping ...')
    with open('umls_aliases___.map', 'wb') as f:
        alias_mapping = dict(alias_mapping)
        pickle.dump(alias_mapping, f)


class ST_CLF(object):

    def __init__(self, clf_path, mapping_path):
        self.clf = None
        self.label2idx_mapping = None
        self.idx2label_mapping = None

        self.load(clf_path, mapping_path)

    def load(self, clf_path, mapping_path):
        self.clf = joblib.load(clf_path)
        
        self.label2idx_mapping = joblib.load(mapping_path)

        # with open(mapping_path, 'rb') as f:
        #     self.label2idx_mapping = pickle.load(f)

        self.idx2label_mapping = dict(zip(self.label2idx_mapping.values(), self.label2idx_mapping.keys()))

    def predict(self, query_feats):

        preds = self.clf.predict_proba([query_feats])[0]
        preds = [(self.idx2label_mapping[cls_idx], cls_prob) for cls_idx, cls_prob in enumerate(preds)]
        preds = sorted(preds, key=lambda x:x[1], reverse=True)

        return preds


if __name__ == '__main__':

    # create_alias_vecs()

    # st_clf = ST_CLF('mlp512.sts.NCBI_BERT.v0.joblib', 'mlp512.sts.NCBI_BERT.v0.mapping.p')
    st_clf = ST_CLF('mlp512.sts.NCBI_BERT.v1.model.joblib', 'mlp512.sts.NCBI_BERT.v1.mapping.joblib')
    
