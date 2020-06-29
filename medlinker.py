import numpy as np
from sklearn.externals import joblib

from pytt_hf import toks2vecs
from matcher_simstring import SimString_UMLS
from matcher_exactmatch import WhitespaceTokenizer  # ???
from vectorspace import VSM
# from vectorspace import FaissVSM

from matcher_softmax import SoftMax_CLF


def norm(v):
    return v / np.linalg.norm(v)


class MedLinkerDoc(object):

    def __init__(self, text, tokens, spans):
        # 
        self.text = text
        self.tokens = tokens
        self.spans = spans
        self.vectors = []

    def set_contextual_vectors(self):
        # 
        self.vectors = toks2vecs(self.tokens, return_tokens=False)
        assert len(self.vectors) == len(self.tokens)

    def get_span_vector(self, span_start, span_end, normalize=False):
        # 
        span_vecs = [self.vectors[i] for i in range(span_start, span_end)]
        span_vec = np.array(span_vecs).mean(axis=0)

        if normalize:
            span_vec = norm(span_vec)

        return span_vec

    def get_spans(self, include_vectors=True, normalize=False):
        # 
        output_spans = []

        for span_start, span_end in self.spans:
            if include_vectors and len(self.vectors) > 0:
                span_vec = self.get_span_vector(span_start, span_end, normalize=normalize)
                output_spans.append((span_start, span_end, span_vec))
            else:
                output_spans.append((span_start, span_end))

        return output_spans


class MedLinker(object):

    def __init__(self, medner, umls_kb):
        # 
        self.medner = medner
        self.umls_kb = umls_kb

        self.string_matcher = None
        self.exact_matcher = None
        self.sty_clf = None
        self.cui_clf = None

        self.st_vsm = None
        self.cui_vsm = None

        self.cui_validator = None
        self.cui_validator_thresh = None

        self.st_validator = None
        self.st_validator_thresh = None

    def load_sty_clf(self, model_path):
        # 
        self.sty_clf = SoftMax_CLF(threshold=0.5)
        self.sty_clf.load(model_path, model_path.replace('.h5', '.map'))

    def load_cui_clf(self, model_path):
        # 
        self.cui_clf = SoftMax_CLF(threshold=0.5)
        self.cui_clf.load(model_path,  model_path.replace('.h5', '.map'))

    def load_st_VSM(self, st_vsm_path):
        # 
        self.st_vsm = VSM(st_vsm_path)

    def load_cui_VSM(self, cui_vecs_path):
        #
        self.cui_vsm = VSM(cui_vecs_path)

    def load_cui_validator(self, clf_path, validator_thresh=0.5):
        #
        self.cui_validator = joblib.load(clf_path)
        self.cui_validator_thresh = validator_thresh

    def load_st_validator(self, clf_path, validator_thresh=0.5):
        #
        self.st_validator = joblib.load(clf_path)
        self.st_validator_thresh = validator_thresh

    def load_string_matcher(self, ngram_db_path, ngram_map_path):
        # 
        self.string_matcher = SimString_UMLS(self.umls_kb, ngram_db_path, ngram_map_path)

    def load_exact_matcher(self, em_path):
        # TO-DO
        pass

    def predict(self, sentence,
                gold_tokens=None, gold_spans=None,
                predict_cui=True, predict_sty=True,
                require_cui=True, require_sty=False):
        # 
        if (gold_tokens is not None) and (gold_spans is not None):
            tokens, spans = gold_tokens, gold_spans
        else:
            tokens, spans = self.medner.predict(sentence)

        doc = MedLinkerDoc(sentence, tokens, spans)
        doc.set_contextual_vectors()

        r = {'sentence': sentence, 'tokens': tokens, 'spans': []}
        # for span_start, span_end, span_vec in doc.get_spans(include_vectors=True, normalize=True):
        for span_start, span_end, span_vec in doc.get_spans(include_vectors=True, normalize=False):
            span_str = ' '.join(doc.tokens[span_start:span_end])
            span_info = {'start': span_start, 'end': span_end, 'text': span_str, 'st': None, 'cui': None}

            if predict_cui:
                span_cuis = self.match_cui(span_str, span_vec)
                
                if span_cuis is not None:
                    span_info['cui'] = span_cuis[0]
                elif require_cui:
                    continue

            if predict_sty:
                span_sts = self.match_sty(span_str, span_vec)
                
                if span_sts is not None:
                    span_info['st'] = span_sts[0]
                elif require_sty:
                    continue

            r['spans'].append(span_info)

        return r

    def match_sty(self, span_str, span_ctx_vec):
        # 
        matches_str = []
        if self.string_matcher is not None:
            matches_str = self.string_matcher.match_sts(span_str.lower())
        elif self.exact_matcher is not None:
            matches_str = self.exact_matcher.match_sts(span_str.lower())
            matches_str = [(st, 1/(1+idx)) for idx, (st, _, _, _) in enumerate(matches_str)]

        matches_ctx = []
        if self.sty_clf is not None:
            matches_ctx = self.sty_clf.predict(span_ctx_vec)
        elif self.st_vsm is not None:
            span_ctx_vec = norm(span_ctx_vec)
            matches_ctx = self.st_vsm.most_similar(span_ctx_vec, threshold=0.5)

        scores_str, scores_ctx = dict(matches_str), dict(matches_ctx)
        matches = {sty: max(scores_str.get(sty, 0), scores_ctx.get(sty, 0))
                   for sty in scores_str.keys() | scores_ctx.keys()}
        matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)

        if (self.st_validator is not None) and (len(matches) > 0):
            pred_valid = self.validate_st_pred(matches_str, scores_str, 
                                               matches_ctx, scores_ctx,
                                               matches)

            if pred_valid == False:
                matches = {}

        if len(matches) > 0:
            return matches
        else:
            return None

    def match_cui(self, span_str, span_ctx_vec):
        #
        matches_str = []
        if self.string_matcher is not None:
            matches_str = self.string_matcher.match_cuis(span_str.lower())
        elif self.exact_matcher is not None:
            matches_str = self.exact_matcher.match_cuis(span_str.lower())
            matches_str = [(cui, 1/(1+idx)) for idx, (cui, _, _, _) in enumerate(matches_str)]

        matches_ctx = []
        if self.cui_clf is not None:
            matches_ctx = self.cui_clf.predict(span_ctx_vec)
        elif self.cui_vsm is not None:
            span_ctx_vec = norm(span_ctx_vec)
            matches_ctx = self.cui_vsm.most_similar(span_ctx_vec, threshold=0.5)

        scores_str, scores_ctx = dict(matches_str), dict(matches_ctx)
        matches = {cui: max(scores_str.get(cui, 0), scores_ctx.get(cui, 0))
                   for cui in scores_str.keys() | scores_ctx.keys()}
        matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)

        if (self.cui_validator is not None) and (len(matches) > 0):
            pred_valid = self.validate_cui_pred(matches_str, scores_str, 
                                                matches_ctx, scores_ctx,
                                                matches)

            if pred_valid == False:
                matches = {}

        if len(matches) > 0:
            return matches
        else:
            return None

    def validate_cui_pred(self, matches_str, scores_str, matches_vsm, scores_vsm, matches_joint):

        matchers_agree = False
        if len(matches_str) > 0 and len(matches_vsm) > 0:
            if matches_str[0][0] == matches_vsm[0][0]:
                matchers_agree = True
        
        top_match = matches_joint[0][0]

        x = []
        if len(matches_str) > 0:
            x.append(matches_str[0][1])
        else:
            x.append(0)
        if len(matches_vsm) > 0:
            x.append(matches_vsm[0][1])
        else:
            x.append(0)
        x.append(matches_joint[0][1])
        x.append((scores_str.get(top_match, 0) + scores_vsm.get(top_match, 0))/2)
        x.append(int(matchers_agree))

        prob_F, prob_T = self.cui_validator.predict_proba([x])[0]

        if prob_T >= self.cui_validator_thresh:
            return True
        else:
            return False

    def validate_st_pred(self, matches_str, scores_str, matches_vsm, scores_vsm, matches_joint):

        matchers_agree = False
        if len(matches_str) > 0 and len(matches_vsm) > 0:
            if matches_str[0][0] == matches_vsm[0][0]:
                matchers_agree = True
        
        top_match = matches_joint[0][0]

        x = []
        if len(matches_str) > 0:
            x.append(matches_str[0][1])
        else:
            x.append(0)
        if len(matches_vsm) > 0:
            x.append(matches_vsm[0][1])
        else:
            x.append(0)
        x.append(matches_joint[0][1])
        x.append((scores_str.get(top_match, 0) + scores_vsm.get(top_match, 0))/2)
        x.append(int(matchers_agree))

        prob_F, prob_T = self.st_validator.predict_proba([x])[0]

        if prob_T >= self.st_validator_thresh:
            return True
        else:
            return False



if __name__ == '__main__':

    from medner import MedNER
    from umls import umls_kb_st21pv as umls_kb

    cx_ner_path = 'models/ContextualNER/mm_st21pv_SCIBERT_uncased/'
    em_ner_path = 'models/ExactMatchNER/umls.2017AA.active.st21pv.nerfed_nlp_and_matcher.max3.p'
    ngram_db_path = 'models/SimString/umls.2017AA.active.st21pv.aliases.3gram.5toks.db'
    ngram_map_path = 'models/SimString/umls.2017AA.active.st21pv.aliases.5toks.map'
    st_vsm_path = 'models/VSMs/mm_st21pv.sts_anns.scibert_scivocab_uncased.vecs'
    cui_vsm_path = 'models/VSMs/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'
    cui_clf_path = 'models/Classifiers/softmax.cui.h5'
    sty_clf_path = 'models/Classifiers/softmax.sty.h5'
    cui_val_path = 'models/Validators/mm_st21pv.lr_clf_cui.dev.joblib'
    sty_val_path = 'models/Validators/mm_st21pv.lr_clf_sty.dev.joblib'

    print('Loading MedNER ...')
    medner = MedNER(umls_kb)
    medner.load_contextual_ner(cx_ner_path)
    
    print('Loading MedLinker ...')
    medlinker = MedLinker(medner, umls_kb)
    
    medlinker.load_string_matcher(ngram_db_path, ngram_map_path)  # simstring approximate string matching

    medlinker.load_st_VSM(st_vsm_path)
    # medlinker.load_sty_clf(sty_clf_path)
    # medlinker.load_st_validator(sty_val_path, validator_thresh=0.45)
    
    medlinker.load_cui_VSM(cui_vsm_path)
    # medlinker.load_cui_clf(cui_clf_path)
    # medlinker.load_cui_validator(cui_val_path, validator_thresh=0.70)

    s = 'Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity.'
    r = medlinker.predict(s)
    print(r)

