import json
import logging
import numpy as np

from matcher_exactmatch import WhitespaceTokenizer  # ???

from umls import umls_kb_st21pv as umls_kb
from medner import MedNER
from medlinker import MedLinker

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def read_mm_converted(mm_set_path):

    with open(mm_set_path, 'r') as json_f:
        mm_set = json.load(json_f)

    return list(mm_set['docs'])


def calc_metrics(obs):
    # 
    def calc_p(obs):
        try:
            return len(obs['tp'])/(len(obs['tp']) + len(obs['fp']))
        except ZeroDivisionError:
            return 0

    def calc_r(obs):
        try:
            return len(obs['tp'])/(len(obs['tp']) + len(obs['fn']))
        except ZeroDivisionError:
            return 0

    def calc_f1(obs):
        try:
            p = calc_p(obs)
            r = calc_r(obs)
            return 2*((p*r)/(p+r))
        except ZeroDivisionError:
            return 0

    def calc_acc(obs):
        try:
            return len(obs['tp'])/sum([len(v) for v in obs.values()])
        except ZeroDivisionError:
            return 0

    p = calc_p(obs) * 100
    r = calc_r(obs) * 100
    f1 = calc_f1(obs) * 100
    acc = calc_acc(obs) * 100
    
    return p, r, f1, acc

def stringify_obs(obs):
    # 
    obs_counts = {k:len(m) for k, m in obs.items()}
    obs_counts['n'] = sum(obs_counts.values())
    return ' '.join(['%s:%d' % (l.upper(), c) for l, c in obs_counts.items()])


def update_obs(doc_idx, sent_idx, gold_spans, pred_spans, perf_ner, perf_st, perf_cui):

    # 1st pass - register pred matched and unmatched (TP & FP)
    for pred_span in pred_spans:
        pred_start, pred_end = pred_span['start'], pred_span['end']
        pred_info = (doc_idx, sent_idx, pred_start, pred_end)

        matched_ner, matched_st, matched_cui = False, False, False
        for gold_span in gold_spans:
            gold_start, gold_end = gold_span['start'], gold_span['end']
            gold_info = (doc_idx, sent_idx, gold_start, gold_end)

            if (pred_start == gold_start) and (pred_end == gold_end):
                matched_ner = True

                if pred_span['st'] is not None:
                    if pred_span['st'][0] == gold_span['st']:
                        matched_st = True  # matched st & NER

                if pred_span['cui'] is not None:
                    gold_span['cui'] = gold_span['cui'].lstrip('UMLS:')  # maybe fix in dataset...
                    if pred_span['cui'][0] == gold_span['cui']:
                        matched_cui = True  # matched cui & NER
        
        if matched_ner:
            perf_ner['tp'].add(pred_info)
        else:
            perf_ner['fp'].add(pred_info)

        if matched_st:
            perf_st['tp'].add(pred_info)
        else:
            perf_st['fp'].add(pred_info)

        if matched_cui:
            perf_cui['tp'].add(pred_info)
        else:
            perf_cui['fp'].add(pred_info)

    # 2nd pass - register unmatched preds (FN)
    for gold_span in gold_spans:
        gold_start, gold_end = gold_span['start'], gold_span['end']
        gold_info = (doc_idx, sent_idx, gold_start, gold_end)

        if gold_info not in perf_ner['tp'].union(perf_ner['fp']):
            perf_ner['fn'].add(gold_info)

        if gold_info not in perf_st['tp'].union(perf_st['fp']):
            perf_st['fn'].add(gold_info)

        if gold_info not in perf_cui['tp'].union(perf_cui['fp']):
            perf_cui['fn'].add(gold_info)    



if __name__ == '__main__':

    use_gold_spans = False
    mm_ann = 'sty'
    # mm_ann = 'cui'
    # mm_ann = ''

    # st21pv
    cx_ner_path = 'models/ContextualNER/mm_st21pv_SCIBERT_uncased/'
    em_ner_path = 'models/ExactMatchNER/umls.2017AA.active.st21pv.nerfed_nlp_and_matcher.max3.p'
    ngram_db_path = 'models/SimString/umls.2017AA.active.st21pv.aliases.3gram.5toks.db'
    ngram_map_path = 'models/SimString/umls.2017AA.active.st21pv.aliases.5toks.map'
    st_vsm_path = 'models/VSMs/mm_st21pv.sts_anns.scibert_scivocab_uncased.vecs'
    cui_vsm_path = 'models/VSMs/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'
    # cui_idx_path = 'models/VSMs/umls.2017AA.active.st21pv.scibert_scivocab_uncased.cuis.index'
    # cui_lbs_path = 'models/VSMs/umls.2017AA.active.st21pv.scibert_scivocab_uncased.cuis.labels'
    # cui_val_path = 'models/Validators/mm_st21pv.lr_clf_cui.dev.joblib'
    # st_val_path = 'models/Validators/mm_st21pv.lr_clf_sty.dev.joblib'
    cui_clf_path = 'models/Classifiers/softmax.cui.h5'
    sty_clf_path = 'models/Classifiers/softmax.sty.h5'
    cui_val_path = 'models/Validators/mm_st21pv.lr_clf_cui.train2.joblib'
    sty_val_path = 'models/Validators/mm_st21pv.lr_clf_sty.train2.joblib'
 
    print('Loading MedNER ...')
    medner = MedNER(umls_kb)
    # medner.load_exactmatch_ner(em_ner_path)
    medner.load_contextual_ner(cx_ner_path, ws_tokenizer=True)

    print('Loading MedLinker ...')
    medlinker = MedLinker(medner, umls_kb)
    medlinker.load_string_matcher(ngram_db_path, ngram_map_path)
    # medlinker.exact_matcher = medner.exactmatch_ner

    predict_cui, require_cui = False, False
    predict_sty, require_sty = False, False
    if mm_ann == 'cui':
        # medlinker.load_cui_VSM(cui_vsm_path)
        # medlinker.load_cui_clf(cui_clf_path)
        # cui_val_path = 'models/Validators/mm_st21pv.lr_clf_cui.dev.joblib'
        # medlinker.load_cui_validator(cui_val_path, validator_thresh=0.70)

        predict_cui, require_cui = True, True

    elif mm_ann == 'sty':
        # medlinker.load_st_VSM(st_vsm_path)
        # medlinker.load_sty_clf(sty_clf_path)
        # sty_val_path = 'models/Validators/mm_st21pv.lr_clf_sty.dev.joblib'
        # medlinker.load_st_validator(sty_val_path, validator_thresh=0.45)

        predict_sty, require_sty = True, True

    perf_stats = {'n_gold_spans': 0, 'n_pred_spans': 0, 'n_sents': 0, 'n_docs': 0}
    perf_ner = {'tp': set(), 'fp': set(), 'fn': set()}
    perf_cui = {'tp': set(), 'fp': set(), 'fn': set()}
    perf_st  = {'tp': set(), 'fp': set(), 'fn': set()}

    logging.info('Loading MedMentions ...')
    mm_docs = read_mm_converted('data/MedMentions/st21pv/custom/mm_converted.test.json')

    logging.info('Processing Instances ...')
    for doc_idx, doc in enumerate(mm_docs):
        perf_stats['n_docs'] += 1

        logging.info('At doc #%d' % doc_idx)

        for sent_idx, gold_sent in enumerate(doc['sentences']):
            perf_stats['n_sents'] += 1

            if use_gold_spans:
                gold_spans = [(s['start'], s['end']) for s in gold_sent['spans']]
                gold_tokens = gold_sent['tokens']

                preds = medlinker.predict(sentence=' '.join(gold_sent['tokens']),
                                          gold_tokens=gold_tokens, gold_spans=gold_spans,
                                          predict_cui=predict_cui, predict_sty=predict_sty,
                                          require_cui=require_cui, require_sty=require_sty)
                # assert len(gold_sent['spans']) == len(preds['spans'])
            
            else:
                preds = medlinker.predict(sentence=' '.join(gold_sent['tokens']),  # expects ws separated text
                                          predict_cui=predict_cui, predict_sty=predict_sty,
                                          require_cui=require_cui, require_sty=require_sty)

            pred_spans = preds['spans']
            gold_spans = gold_sent['spans']
            # assert preds['tokens'] == gold_sent['tokens']  # hence, equal boundaries == equal text

            perf_stats['n_gold_spans'] += len(gold_spans)
            perf_stats['n_pred_spans'] += len(pred_spans)

            update_obs(doc_idx, sent_idx, gold_spans, pred_spans, perf_ner, perf_st, perf_cui)

        # in-progress performance metrics
        for pred_type, type_obs in [('NER', perf_ner), ('STY', perf_st), ('CUI', perf_cui)]:
            p, r, f1, acc = calc_metrics(type_obs)
            obs_str = stringify_obs(type_obs)
            print('[%s] P:%.2f R:%.2f F1:%.2f ACC:%.2f - %s' % (pred_type, p, r, f1, acc, obs_str))
        print(perf_stats)
        print()

