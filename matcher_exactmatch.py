import logging
import random
import pickle
from functools import lru_cache

import spacy
from spacy.tokens import Doc
from spacy.matcher import PhraseMatcher


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class WhitespaceTokenizer(object):
    # copied from spacy docs
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


class ExactMatch_UMLS():

    def __init__(self, umls_db, nerfed_nlp_and_matcher_path):
        self.umls_db = umls_db
        self.nlp_nerfed = None
        self.matcher = None

        self.load(nerfed_nlp_and_matcher_path)

    def load(self, nerfed_nlp_and_matcher_path):
        # 
        with open(nerfed_nlp_and_matcher_path, 'rb') as f:
            self.nlp_nerfed, self.matcher = pickle.load(f)

    def hash2string(self, hash_):
        return self.nlp_nerfed.vocab.strings[hash_]

    @lru_cache(262144)
    def match_cuis(self, text, ignore_overlaps=True):
        # 
        doc = self.nlp_nerfed(text.lower())
        tokens = text.split(' ')
        
        matches = self.matcher(doc)
        matches = [(self.hash2string(h), s, e) for (h, s, e) in matches]

        # remove alias indexes from cui_ids
        matches = [(cui_id.split('_')[0], s, e) for (cui_id, s, e) in matches]

        matches = [(cui, s, e, ' '.join(tokens[s:e])) for (cui, s, e) in matches]

        # sort by num. tokens
        matches = sorted(matches, key=lambda x: len(x[-1].split()), reverse=True)

        # 
        if ignore_overlaps:
            matches_no_overlaps = []
            matched_idxs = set()
            for cui, s, e, t in matches:
                match_idxs = set(list(range(s, e)))
                if len(matched_idxs.intersection(match_idxs)) > 0:
                    continue

                matches_no_overlaps.append((cui, s, e, t))
                matched_idxs.update(match_idxs)
            
            matches = matches_no_overlaps
        
        return matches

    def match_sts(self, text, ignore_overlaps=True):
        # 
        matches = []
        for cui, s, e, t in self.match_cuis(text, ignore_overlaps=ignore_overlaps):
            st = self.umls_db.get_sts(cui)[0]  # take 1st STY, no scores to compare
            matches.append((st, s, e, t))

        return matches        


def create_matcher(umls_kb, n_max_tokens=5):
    # 
    from nltk.corpus import stopwords
    en_stopwords = set(stopwords.words('english'))
    fb_punctuation = set('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~')  # string.punctuation except '-'

    # from umls_utils import cui2ent
    # from umls_utils import all_cuis

    logging.info('Loading scispacy (and nerfing it) ...')
    sci_nlp_nerfed = spacy.load('en_core_sci_sm', disable=['tagger', 'parser', 'ner'])
    sci_nlp_nerfed.tokenizer = WhitespaceTokenizer(sci_nlp_nerfed.vocab)  # enforcing ws tokenizer 

    logging.info('Loading and adding UMLS aliases ...')
    sci_matcher = PhraseMatcher(sci_nlp_nerfed.vocab)

    n_added = 0
    for cui_idx, cui in enumerate(umls_kb.get_all_cuis()):

        if cui_idx % 100000 == 0:
            logging.info('at cui #%d/>2.3M, added %d' % (cui_idx, n_added))
        
        # ent = cui2ent(cui)
        # if ent is None:
        #     continue
        
        cui_aliases = set([a.lower() for a in umls_kb.get_aliases(cui, include_name=True)])
        cui_aliases = [' '.join(a.split()) for a in cui_aliases]  # normalizing ws
        # unique_aliases = set([a.lower() for a in ent.aliases])
        # unique_aliases.add(ent.canonical_name.lower())

        for alias_idx, alias in enumerate(cui_aliases):
            
            if alias in en_stopwords:
                continue
            
            elif alias.isnumeric():
                continue

            alias_chars = set(alias)
            if len(alias_chars.intersection(fb_punctuation)) > 0:
                continue

            alias_doc = sci_nlp_nerfed(alias)
            if len(alias_doc) > n_max_tokens:  # gets too big without restrictions
                continue

            cui_id = '%s_%d' % (cui, alias_idx)  # otherwise alias gets overwritten for the same cui
            sci_matcher.add(cui_id, None, alias_doc)
            n_added += 1

    logging.info('Pickling [sci_nlp_nerfed, sci_matcher] ...')

    phrase_matcher_path = 'models/ExactMatchNER/%s.nerfed_nlp_and_matcher.max%d.p' % (umls_kb.umls_version, n_max_tokens)

    with open(phrase_matcher_path, 'wb') as f:
        pickle.dump([sci_nlp_nerfed, sci_matcher], f)

    logging.info('Done')


if __name__ == '__main__':

    # from umls import umls_kb_full as umls_kb
    from umls import umls_kb_st21pv as umls_kb
    
    create_matcher(umls_kb, n_max_tokens=3)

    # phrase_matcher = ExactMatch_UMLS('models/ExactMatchNER/mm_st21pv.nerfed_nlp_and_matcher.max5.p')
    # r = phrase_matcher.match_cuis('1 genus actinokineospora 2')

    pass
