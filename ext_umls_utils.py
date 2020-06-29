from functools import lru_cache
from collections import Counter

import spacy
from scispacy.umls_utils import UmlsKnowledgeBase

from scispacy.umls_semantic_type_tree import construct_umls_tree_from_tsv
umls_tree = construct_umls_tree_from_tsv("data/umls_semantic_type_tree.tsv")

sci_nlp = spacy.load('en_core_sci_md')
# sci_nlp = spacy.load('en_core_sci_lg')


st21pv_set = set(['T005', 'T007', 'T017', 'T022', 'T031', 'T033', 'T037', 
                  'T038', 'T058', 'T062', 'T074', 'T082', 'T091', 'T092', 
                  'T097', 'T098', 'T103', 'T168', 'T170', 'T201', 'T204'])


def cui2ent(cui):
    try:
        return umls_kb.cui_to_entity[cui]
    except KeyError:
        return None


@lru_cache(maxsize=None)
def cui2st(cui):
    ent = cui2ent(cui)
    if ent is None:
        return None
    else:
        return max(ent.types, key=lambda x: umls_tree.get_node_from_id(x).level)


umls_kb = UmlsKnowledgeBase()
all_cuis = set(umls_kb.cui_to_entity.keys())

