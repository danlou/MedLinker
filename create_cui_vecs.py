import logging
import numpy as np

from pytt_hf import toks2vecs
from pytt_hf import PYTT_CONFIG

from umls import umls_kb_st21pv as umls_kb


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


logging.info('Loading scispacy ...')
import spacy
sci_nlp = spacy.load('en_core_sci_md', disable=['tagger', 'parser', 'ner'])


logging.info('Embedding aliases ...')
cui_vecs = {}
for cui_idx, cui in enumerate(umls_kb.get_all_cuis()):

    # if cui_idx > 0:
    if cui_idx % 1000 == 0:
        logging.info('At #CUI: %d/%d' % (cui_idx, len(umls_kb.umls_data)))

    cui_aliases_vecs = []
    for alias in umls_kb.get_aliases(cui, include_name=True):
        alias_toks = [t.text.lower() for t in sci_nlp(alias)]
        alias_vecs = toks2vecs(alias_toks, return_tokens=False)

        alias_vec = np.array(alias_vecs).mean(axis=0)
        cui_aliases_vecs.append(alias_vec)

    cui_vecs[cui] = np.array(cui_aliases_vecs).mean(axis=0)


logging.info('Writing vecs ...')
vecs_path = '%s.%s.cuis.vecs' % (umls_kb.umls_version, PYTT_CONFIG['name'])
with open(vecs_path, 'w') as vecs_f:
    for cui, vec in cui_vecs.items():
        vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
        vecs_f.write('%s %s\n' % (cui, vec_str))
logging.info('Written %s' % vecs_path)
