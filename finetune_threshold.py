import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


# from eval_mm_spans import medlinker
from umls import umls_kb_st21pv as umls_kb
from medner import MedNER
from medlinker import MedLinker

from eval_mm_spans import read_mm_converted
from eval_mm_spans import calc_metrics
from eval_mm_spans import update_obs


def write_perf(path, threshold, perf_ner, perf_st, perf_cui):
    # 
    with open(path, 'a') as f:
        for pred_type, type_obs in [('NER', perf_ner), ('STY', perf_st), ('CUI', perf_cui)]:
            p, r, f1, acc = calc_metrics(type_obs)

            elems = []
            elems.append(str(datetime.now()))
            elems.append(pred_type)
            elems.append('%.2f' % threshold)
            elems.append('%.4f' % p)
            elems.append('%.4f' % r)
            elems.append('%.4f' % f1)
            elems.append('%.4f' % acc)
            line = ','.join(elems)
            f.write(line+'\n')


if __name__ == '__main__':

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
    medlinker.load_cui_clf(cui_clf_path)
    medlinker.load_sty_clf(sty_clf_path)

    mm_ann = 'cui'
    # mm_ann = 'sty'
    mm_set, clf_set = 'dev', 'train2'
    # mm_set, clf_set = 'dev', 'dev'

    if mm_ann == 'cui':
        medlinker.load_cui_validator(cui_val_path, validator_thresh=0.0)

        predict_cui, require_cui = True, True
        predict_sty, require_sty = False, False

    elif mm_ann == 'sty':
        medlinker.load_st_validator(sty_val_path, validator_thresh=0.0)

        predict_cui, require_cui = False, False
        predict_sty, require_sty = True, True


    logging.info('Loading MedMentions ...')
    mm_docs = read_mm_converted('data/MedMentions/st21pv/custom/mm_converted.%s.json' % mm_set)

    candidate_thresholds = [t*0.01 for t in range(0, 101, 5)]  
    for threshold in candidate_thresholds:

        medlinker.cui_validator_thresh = threshold
        medlinker.st_validator_thresh = threshold

        perf_ner = {'tp': set(), 'fp': set(), 'fn': set()}
        perf_cui = {'tp': set(), 'fp': set(), 'fn': set()}
        perf_st  = {'tp': set(), 'fp': set(), 'fn': set()}

        logging.info('Processing Instances ...')
        for doc_idx, doc in enumerate(mm_docs):
            logging.info('At doc #%d' % doc_idx)

            for sent_idx, gold_sent in enumerate(doc['sentences']):

                preds = medlinker.predict(sentence=' '.join(gold_sent['tokens']),
                                          predict_cui=predict_cui, predict_sty=predict_sty,
                                          require_cui=require_cui, require_sty=require_sty)

                pred_spans = preds['spans']
                gold_spans = gold_sent['spans']

                update_obs(doc_idx, sent_idx, gold_spans, pred_spans, perf_ner, perf_st, perf_cui)
                
            # in-progress performance metrics
            for pred_type, type_obs in [('NER', perf_ner), ('STY', perf_st), ('CUI', perf_cui)]:
                p, r, f1, acc = calc_metrics(type_obs)
                print('[%s@%.1f] P:%.2f R:%.2f F1:%.2f ACC:%.2f' % (pred_type, threshold, p, r, f1, acc))
            print()

        logging.info('Writing Performance at Threshold ...')
        fn = '%s_finetune_results.%s.%s2.csv' % (mm_ann, mm_set, clf_set)
        write_perf(fn, threshold, perf_ner, perf_st, perf_cui)

