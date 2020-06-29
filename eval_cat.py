import json
import logging
from datetime import datetime

from matcher_exactmatch import WhitespaceTokenizer  # ???

from umls import umls_kb_st21pv as umls_kb
from medner import MedNER
from medlinker import MedLinker

from eval_mm_spans import read_mm_converted

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def calc_acc(target_sty_stats):
    try:
        acc = 100 * (target_sty_stats['correct']/target_sty_stats['n'])
    except ZeroDivisionError:
        acc = 0    
    return acc


if __name__ == '__main__':

    # st21pv
    cx_ner_path = 'models/ContextualNER/mm_st21pv_SCIBERT_uncased/'
    em_ner_path = 'models/ExactMatchNER/umls.2017AA.active.st21pv.nerfed_nlp_and_matcher.max3.p'
    ngram_db_path = 'models/SimString/umls.2017AA.active.st21pv.aliases.3gram.5toks.db'
    ngram_map_path = 'models/SimString/umls.2017AA.active.st21pv.aliases.5toks.map'
    sty_vsm_path = 'models/VSMs/mm_st21pv.sts_anns.scibert_scivocab_uncased.vecs'
    sty_val_path = 'models/Validators/mm_st21pv.lr_clf_sty.dev.joblib'
    cui_vsm_path = 'models/VSMs/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'
    # cui_val_path = 'models/Validators/mm_st21pv.lr_clf_cui.dev.joblib'
    cui_val_path = 'models/Validators/mm_st21pv.lr_clf_cui.train2.joblib'
    sty_val_path = 'models/Validators/mm_st21pv.lr_clf_sty.train2.joblib'
    cui_clf_path = 'models/Classifiers/softmax.cui.h5'
    sty_clf_path = 'models/Classifiers/softmax.sty.h5'

    print('Loading MedNER ...')
    medner = MedNER(umls_kb)
    medner.load_contextual_ner(cx_ner_path, ws_tokenizer=True)

    print('Loading MedLinker ...')
    medlinker = MedLinker(medner, umls_kb)
    medlinker.load_string_matcher(ngram_db_path, ngram_map_path)
    # medlinker.load_cui_VSM(cui_vsm_path)
    medlinker.load_cui_clf(cui_clf_path)
    medlinker.load_cui_validator(cui_val_path, validator_thresh=0.70)
    # medlinker.load_st_VSM(sty_vsm_path)
    medlinker.load_sty_clf(sty_clf_path)
    medlinker.load_st_validator(sty_val_path, validator_thresh=0.40)

    use_gold_spans = False
    mm_ann = 'cui'
    # mm_ann = 'sty'

    predict_sty, require_sty = False, False
    predict_cui, require_cui = False, False
    if mm_ann == 'cui':
        predict_cui, require_cui = True, True
    elif mm_ann == 'sty':
        predict_sty, require_sty = True, True

    logging.info('Loading MedMentions ...')
    mm_docs = read_mm_converted('data/MedMentions/st21pv/custom/mm_converted.test.json')

    for target_sty in umls_kb.get_all_stys():

        target_sty_stats = {'n': 0, 'correct': 0}

        logging.info('Processing Instances ...')
        for doc_idx, doc in enumerate(mm_docs):
            logging.info('[%s] At doc #%d - mentions #%d' % (target_sty, doc_idx, target_sty_stats['n']))

            for sent_idx, gold_sent in enumerate(doc['sentences']):

                # check if sent has spans of target_sty
                has_target_sty = False
                for gold_span in gold_sent['spans']:
                    if gold_span['st'] == target_sty:
                        has_target_sty = True
                        break
                if not has_target_sty:
                    continue

                if use_gold_spans:
                    gold_spans = [(s['start'], s['end']) for s in gold_sent['spans']]
                    gold_tokens = gold_sent['tokens']

                    preds = medlinker.predict(sentence=' '.join(gold_sent['tokens']),
                                            gold_tokens=gold_tokens, gold_spans=gold_spans,
                                            predict_cui=predict_cui, predict_sty=predict_sty,
                                            require_cui=require_cui, require_sty=require_sty)
                
                else:
                    preds = medlinker.predict(sentence=' '.join(gold_sent['tokens']),  # expects ws separated text
                                            predict_cui=predict_cui, predict_sty=predict_sty,
                                            require_cui=require_cui, require_sty=require_sty)


                for gold_span in gold_sent['spans']:
                    gold_start, gold_end = gold_span['start'], gold_span['end']
                    if gold_span['st'] != target_sty:
                        continue

                    target_sty_stats['n'] += 1
                    for pred_span in preds['spans']:
                        pred_start, pred_end = pred_span['start'], pred_span['end']

                        if (gold_start == pred_start) and (gold_end == pred_end):

                            if mm_ann == 'cui':
                                gold_span['cui'] = gold_span['cui'].lstrip('UMLS:')  # maybe fix in dataset...
                                if pred_span['cui'][0] == gold_span['cui']:
                                    target_sty_stats['correct'] += 1
                                    break

                            elif mm_ann == 'sty':
                                if pred_span['st'][0] == gold_span['st']:
                                    target_sty_stats['correct'] += 1
                                    break


            # in-progress performance metrics
            acc = calc_acc(target_sty_stats)
            print('N:%d C:%d Acc:%.4f' % (target_sty_stats['n'], target_sty_stats['correct'], acc))
            print()

        logging.info('Writing Performance for STY ...')
        if use_gold_spans:
            fn = '%s_cat_results.clf.goldspans.csv' % mm_ann
        else:
            fn = '%s_cat_results.clf.predspans.csv' % mm_ann

        acc = calc_acc(target_sty_stats)
        with open(fn, 'a') as results_f:
            line = '%s,%d,%d,%.4f' % (target_sty, target_sty_stats['n'], target_sty_stats['correct'], acc)
            results_f.write(line+'\n')
