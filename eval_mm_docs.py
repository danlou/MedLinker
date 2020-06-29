import json
import logging
import numpy as np
import itertools

from matcher_exactmatch import WhitespaceTokenizer  # ???

from medner import MedNER
from medlinker import MedLinker

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


logging.info('Loading MedLinker ...')

# st21pv
from umls import umls_kb_st21pv as umls_kb
cx_ner_path = 'models/ContextualNER/mm_st21pv_SCIBERT_uncased/'
em_ner_path = 'models/ExactMatchNER/umls.2017AA.active.st21pv.nerfed_nlp_and_matcher.max3.p'
ngram_db_path = 'models/SimString/umls.2017AA.active.st21pv.aliases.3gram.5toks.db'
ngram_map_path = 'models/SimString/umls.2017AA.active.st21pv.aliases.5toks.map'
st_vsm_path = 'models/VSMs/mm_st21pv.sts_anns.scibert_scivocab_uncased.vecs'
cui_vsm_path = 'models/VSMs/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'
cui_idx_path = 'models/VSMs/umls.2017AA.active.st21pv.scibert_scivocab_uncased.cuis.index'
cui_lbs_path = 'models/VSMs/umls.2017AA.active.st21pv.scibert_scivocab_uncased.cuis.labels'

# full
# from umls import umls_kb_full as umls_kb
# cx_ner_path = 'models/ContextualNER/mm_full_SCIBERT_uncased/'
# ngram_db_path = 'models/SimString/umls.2017AA.active.full.aliases.3gram.5toks.db'
# ngram_map_path = 'models/SimString/umls.2017AA.active.full.aliases.5toks.map'


print('Loading MedNER ...')
# medner = MedNER(cx_ner_path, em_ner_path)
medner = MedNER(contextual_ner_path=cx_ner_path)

print('Loading MedLinker ...')
medlinker = MedLinker(medner, umls_kb)
medlinker.load_st_VSM(st_vsm_path)
medlinker.load_string_matcher(ngram_db_path, ngram_map_path)
# medlinker.load_cui_FaissVSM(cui_idx_path, cui_lbs_path)
medlinker.load_cui_VSM(cui_vsm_path)

# input('...')

def read_mm_converted(mm_set_path):

    with open(mm_set_path, 'r') as json_f:
        mm_set = json.load(json_f)

    return list(mm_set['docs'])


def calc_p(metrics):
    try:
        return metrics['tp']/(metrics['tp'] + metrics['fp'])
    except ZeroDivisionError:
        return 0


def calc_r(metrics):
    try:
        return metrics['tp']/(metrics['tp'] + metrics['fn'])
    except ZeroDivisionError:
        return 0


def calc_f1(metrics):
    try:
        p = calc_p(metrics)
        r = calc_r(metrics)
        return 2*((p*r)/(p+r))
    except ZeroDivisionError:
        return 0


def calc_acc(metrics):
    try:
        return metrics['tp']/sum(metrics.values())
    except ZeroDivisionError:
        return 0


def calc_counts(metrics):
    metrics['n'] = sum(metrics.values())
    return metrics


def stringify_metrics(metrics):
    metrics_counts = calc_counts(metrics)
    return ' '.join(['%s:%d' % (l.upper(), c) for l, c in metrics_counts.items()])


if __name__ == '__main__':

    perf_stats = {'n_gold_spans': 0, 'n_pred_spans': 0, 'n_sents': 0, 'n_docs': 0}
    perf_cui = {'tp': 0, 'fp': 0, 'fn': 0}

    logging.info('Loading MedMentions ...')
    # mm_docs = read_mm_converted('data/MedMentions/full/custom/mm_converted.dev.json')
    # mm_docs = read_mm_converted('data/MedMentions/st21pv/custom/mm_converted.dev.json')
    mm_docs = read_mm_converted('data/MedMentions/st21pv/custom/mm_converted.test.json')

    logging.info('Processing Instances ...')
    for doc_idx, doc in enumerate(mm_docs):
        perf_stats['n_docs'] += 1

        # if doc_idx > 100:
        #     break

        logging.info('At doc #%d' % doc_idx)

        gold_ents = set()
        for gold_sent in doc['sentences']:
            for gold_span in gold_sent['spans']:
                gold_ents.add(gold_span['cui'].lstrip('UMLS:'))

        pred_ents = set()
        for gold_sent in doc['sentences']:
            sent_preds = medlinker.predict(' '.join(gold_sent['tokens']), use_em_ner=False)
            for pred_span in sent_preds['spans']:
                if pred_span['cui'] is not None:
                    pred_ents.add(pred_span['cui'][0])

        perf_cui['tp'] += len(gold_ents.intersection(pred_ents))
        perf_cui['fp'] += len([pred_ent for pred_ent in pred_ents if pred_ent not in gold_ents])
        perf_cui['fn'] += len([gold_ent for gold_ent in gold_ents if gold_ent not in pred_ents])

        # in-progress performance metrics
        p = calc_p(perf_cui) * 100
        r = calc_r(perf_cui) * 100
        f = calc_f1(perf_cui) * 100
        a = calc_acc(perf_cui) * 100

        counts = calc_counts(perf_cui)
        counts_str = '\t'.join(['%s:%d' % (l.upper(), c) for l, c in counts.items()])
        print('[CUI]\tP:%.2f\tR:%.2f\tF1:%.2f\tACC:%.2f - %s' % (p, r, f, a, counts_str))

        # print('doc_ents:', doc_ents)
        # print('pred_ents:', pred_ents)
        # input('...')    

        # for sent_idx, gold_sent in enumerate(doc['sentences']):
        #     perf_stats['n_sents'] += 1

        #     if use_gold_spans:
        #         gold_spans = [('Entity', (s['start'], s['end'] - 1)) for s in gold_sent['spans']]
        #         gold_tokens = gold_sent['tokens']

        #         preds = medlinker.predict(' '.join(gold_sent['tokens']), gold_tokens=gold_tokens, gold_spans=gold_spans, use_em_ner=False)
        #         # assert len(gold_sent['spans']) == len(preds['spans'])
            
        #     else:
        #         preds = medlinker.predict(' '.join(gold_sent['tokens']), use_em_ner=False)  # expects ws separated text

        #     assert preds['tokens'] == gold_sent['tokens']  # hence, equal boundaries == equal text

        #     perf_stats['n_gold_spans'] += len(gold_sent['spans'])
        #     perf_stats['n_pred_spans'] += len(preds['spans'])


        #     """
        #     print(gold_sent['text'])
        #     print('Gold (#%d):' % len(gold_sent['spans']))
        #     for gold_span in gold_sent['spans']:
        #         gold_span['cui'] = gold_span['cui'].lstrip('UMLS:')  # maybe fix in dataset...
        #         umls_kb.pprint(gold_span['cui'])
        #         print(gold_span)
        #     print('\nPred (#%d):' % len(preds['spans']))
        #     for pred_span in preds['spans']:
        #         if pred_span['cui'] is not None:
        #             umls_kb.pprint(pred_span['cui'][0])
        #         print(pred_span)
        #     input('\n...')
        #     """

        #     # 1st pass - register pred matched and unmatched (TP & FP)
        #     for pred_span in preds['spans']:
        #         pred_start, pred_end = pred_span['start'], pred_span['end']
        #         pred_info = (doc_idx, sent_idx, pred_start, pred_end)

        #         matched_ner, matched_st, matched_cui = False, False, False
        #         for gold_span in gold_sent['spans']:
        #             gold_start, gold_end = gold_span['start'], gold_span['end']
        #             gold_info = (doc_idx, sent_idx, gold_start, gold_end)

        #             if (pred_start == gold_start) and (pred_end == gold_end):
        #                 matched_ner = True

        #                 if pred_span['st'] is not None:
        #                     # print(pred_span['st'])
        #                     # print(gold_span['st'])
        #                     # print()
        #                     # if pred_span['st'][0][0] == gold_span['st']:
        #                     if pred_span['st'][0] == gold_span['st']:
        #                         matched_st = True  # matched st & NER

        #                 if pred_span['cui'] is not None:
        #                     gold_span['cui'] = gold_span['cui'].lstrip('UMLS:')  # maybe fix in dataset...
        #                     # print(pred_span['cui'])
        #                     # print(gold_span['cui'])
        #                     # if pred_span['cui'][0][0] == gold_span['cui']:
        #                     if pred_span['cui'][0] == gold_span['cui']:
        #                         matched_cui = True  # matched cui & NER
                
        #         if matched_ner:
        #             perf_ner['tp'].add(pred_info)
        #         else:
        #             perf_ner['fp'].add(pred_info)

        #         if matched_st:
        #             perf_st['tp'].add(pred_info)
        #         else:
        #             perf_st['fp'].add(pred_info)

        #         if matched_cui:
        #             perf_cui['tp'].add(pred_info)
        #         else:
        #             perf_cui['fp'].add(pred_info)

        #     # 2nd pass - register unmatched preds (FN)
        #     for gold_span in gold_sent['spans']:
        #         gold_start, gold_end = gold_span['start'], gold_span['end']
        #         gold_info = (doc_idx, sent_idx, gold_start, gold_end)

        #         # if gold_info not in perf_ner['tp'].union(perf_ner['fp']).union(perf_ner['fn']):
        #         if gold_info not in perf_ner['tp'].union(perf_ner['fp']):
        #             perf_ner['fn'].add(gold_info)

        #         # if pred_info not in perf_st['tp'].union(perf_st['fp']).union(perf_st['fn']):
        #         if gold_info not in perf_st['tp'].union(perf_st['fp']):
        #             perf_st['fn'].add(gold_info)

        #         # if pred_info not in perf_cui['tp'].union(perf_cui['fp']).union(perf_cui['fn']):
        #         if gold_info not in perf_cui['tp'].union(perf_cui['fp']):
        #             perf_cui['fn'].add(gold_info)
            

        # # in-progress performance metrics
        # for pred_type, type_metrics in [('NER', perf_ner), ('ST', perf_st), ('CUI', perf_cui)]:
        #     p = calc_p(type_metrics) * 100
        #     r = calc_r(type_metrics) * 100
        #     f = calc_f1(type_metrics) * 100
        #     a = calc_acc(type_metrics) * 100

        #     counts = calc_counts(type_metrics)
        #     counts_str = '\t'.join(['%s:%d' % (l.upper(), c) for l, c in counts.items()])

        #     print('[%s]\tP:%.2f\tR:%.2f\tF1:%.2f\tACC:%.2f - %s' % (pred_type, p, r, f, a, counts_str))
        # print(perf_stats)
        # print()

