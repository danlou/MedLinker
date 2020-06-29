import logging
import numpy as np
from functools import lru_cache

from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.dataset_readers.dataset_utils.span_utils import bioul_tags_to_spans

from matcher_exactmatch import WhitespaceTokenizer
from matcher_exactmatch import ExactMatch_UMLS


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class MedNER(object):

    def __init__(self, umls_kb):
        self.umls_kb = umls_kb
        self.contextual_ner = None
        self.contextual_ner_labels = []
        self.exactmatch_ner = None
    
    def load_contextual_ner(self, path, ws_tokenizer=True):
        # 
        logging.info('Loading Contextual NER ...')
        self.contextual_ner = Predictor.from_path(path, cuda_device=0)

        if ws_tokenizer:
            # switch-off tokenizer (expect pretokenized, space-separated strings)
            self.contextual_ner._tokenizer = JustSpacesWordSplitter()

        # load labels (to use logits, wip)
        self.contextual_ner_labels = []
        with open(path+'vocabulary/labels.txt', 'r') as labels_f:
            for line in labels_f:
                self.contextual_ner_labels.append(line.strip())

    def load_exactmatch_ner(self, path):
        # 
        logging.info('Loading ExactMatch NER ...')
        self.exactmatch_ner = ExactMatch_UMLS(self.umls_kb, path)

    def predict_exactmatch(self, tokens):
        # 
        em_results = self.exactmatch_ner.match_cuis(' '.join(tokens))

        em_spans = []
        for (_, s, e, text) in em_results:
            assert tokens[s:e] == text.split()  # sanity check
            em_spans.append((s, e))

        return tokens, em_spans

    def predict_contextual(self, sentence):
        # 
        cx_results = self.contextual_ner.predict(sentence)
        tokens = cx_results['words']

        cx_spans = bioul_tags_to_spans(cx_results['tags'])
        cx_spans = [(s, e + 1) for l, (s, e) in cx_spans]  # consistent with em

        return tokens, cx_spans
    
    @lru_cache(262144)
    def predict(self, sentence):
        #
        if self.contextual_ner is not None:
            return self.predict_contextual(sentence)
        
        elif self.exactmatch_ner is not None:
            tokens = sentence.split()  # exactmatch expects pre-tokenized with ws
            return self.predict_exactmatch(tokens)

        else:
            # TO-DO: raise warning
            pass

        return None, None



if __name__ == '__main__':

    from umls import umls_kb_st21pv as umls_kb

    cx_ner_path = 'models/ContextualNER/mm_st21pv_SCIBERT_uncased/'

    medner = MedNER(umls_kb)
    medner.load_contextual_ner(cx_ner_path)

    s = 'Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity.'

    tokens, spans = medner.predict(s)
    print(tokens, spans)
