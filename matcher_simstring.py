import logging
import pickle
from collections import defaultdict
from functools import lru_cache

from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher

from nltk.corpus import stopwords
en_stopwords = set(stopwords.words('english'))
fb_punctuation = set('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~')  # string.punctuation except '-'


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class SimString_UMLS(object):

    def __init__(self, umls_db, db_path, cui_mapping_path, alpha=0.5):
        self.db = None
        self.umls_db = umls_db
        self.cui_mapping = None
        self.searcher = None
        self.alpha = alpha

        self.load(db_path, cui_mapping_path)

    def load(self, db_path, cui_mapping_path):

        logging.info('Loading DB ...')
        with open(db_path, 'rb') as db_f:
            self.db = pickle.load(db_f)
        
        logging.info('Loading Mapping ...')
        with open(cui_mapping_path, 'rb') as mapping_f:
            self.cui_mapping = pickle.load(mapping_f)

        logging.info('Creating Searcher ...')
        self.searcher = Searcher(self.db, CosineMeasure())

    @lru_cache(262144)
    def match(self, text):
        results = self.searcher.ranked_search(text, alpha=self.alpha)
        results = [(a, sim) for sim, a in results]  # to be consistent with other matchers
        return results

    def match_cuis(self, text):
        alias_results = self.match(text)

        cui_results = []
        included_cuis = set()
        for alias, sim in alias_results:
            for cui in self.cui_mapping[alias]:
                if cui not in included_cuis:
                    cui_results.append((cui, sim))
                    included_cuis.add(cui)

        return cui_results

    def match_sts(self, text):

        st_results = {}
        for cui, sim in self.match_cuis(text):
            for st in self.umls_db.get_sts(cui):

                if st not in st_results:
                    st_results[st] = sim
                else:
                    st_results[st] = max(sim, st_results[st])
        
        st_results = list(st_results.items())
        st_results = sorted(st_results, key=lambda x: (x[1], x[0]), reverse=True)

        return st_results


def create_umls_ss_db(umls_kb, char_ngram_len=3, n_max_tokens=5):

    logging.info('Loading scispacy ...')
    import spacy
    sci_nlp = spacy.load('en_core_sci_md', disable=['tagger', 'parser', 'ner'])

    simstring_db = DictDatabase(CharacterNgramFeatureExtractor(char_ngram_len))

    # preprocessing aliases and labels
    logging.info('Preprocessing aliases ...')
    alias_mapping = defaultdict(set)

    aliases = []
    for cui in umls_kb.get_all_cuis():

        cui_aliases = set([a.lower() for a in umls_kb.get_aliases(cui, include_name=True)])

        for alias in cui_aliases:

            alias_chars = set(alias)
            if len(alias_chars.intersection(fb_punctuation)) > 0:
                continue

            elif alias in en_stopwords:
                continue
            
            elif alias.isnumeric():
                continue

            alias_doc = sci_nlp(alias)  # use same tokenizer as when splitting medmentions
            if len(alias_doc) > n_max_tokens:  # gets too big without restrictions
                continue

            alias_mapping[alias].add(cui)
            aliases.append(alias)


    logging.info('Adding to DB ...')
    for alias_idx, alias in enumerate(aliases):
        simstring_db.add(alias)
        if alias_idx % 1000000 == 0:
            logging.info('At %d/%d ...' % (alias_idx, len(aliases)))

    # setting paths
    db_path = '%s.aliases.%dgram.%dtoks.db' % (umls_kb.umls_version, char_ngram_len, n_max_tokens)
    map_path = '%s.aliases.%dtoks.map' % (umls_kb.umls_version, n_max_tokens)

    logging.info('Storing DB ...')
    with open(db_path, 'wb') as f:
        pickle.dump(simstring_db, f)

    logging.info('Storing Alias Mapping ...')
    with open(map_path, 'wb') as f:
        alias_mapping = dict(alias_mapping)
        pickle.dump(alias_mapping, f)


if __name__ == '__main__':

    # ngram_db_path = 'models/SimString/mm_st21pv.umls_aliases.3gram.5toks.db'
    # ngram_map_path = 'models/SimString/mm_st21pv.umls_aliases.5toks.map'

    # umls_string_matcher = SimString_UMLS(ngram_db_path, ngram_map_path)
    # r = umls_string_matcher.match('apoptosis')

    from umls import umls_kb_st21pv as umls_kb
    create_umls_ss_db(umls_kb, char_ngram_len=3, n_max_tokens=5)
