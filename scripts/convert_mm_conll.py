import sys
import logging
import json

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def iterate_docs_converted(split_path):

    # load json dataset
    with open(split_path, 'r') as json_f:
        dataset = json.load(json_f)

    for doc in dataset['docs']:
        yield doc


if __name__ == '__main__':

    specify_st = False
    split_label = sys.argv[1]

    mm_path = 'data/MedMentions/st21pv/custom/mm_converted.%s.json' % split_label

    logging.info('Loading MedMentions - %s ...' % mm_path)
    mm_docs = list(iterate_docs_converted(mm_path))

    conll_lines = []

    logging.info('Processing Instances ...')
    for doc_idx, doc in enumerate(mm_docs):

        conll_lines.append('-DOCSTART- (%s)' % doc['pubmed_id'])
        conll_lines.append('')

        for sent in doc['sentences']:

            tokens = sent['tokens']
            tags = ['O' for t in tokens]

            for ent in sent['spans']:
                
                if specify_st:
                    tag = ent['st']
                else:
                    tag = 'Entity'

                if len(ent['tokens']) == 1:
                    marker = 'B'
                    tags[ent['start']] = '%s-%s' % (marker, tag)
                
                else:
                    B_added = False
                    for tag_idx in range(ent['start'], ent['end']):
                        if not B_added:
                            marker = 'B'
                            B_added = True
                        else:
                            marker = 'I'

                        tags[tag_idx] = '%s-%s' % (marker, tag)
            
            for token, tag in zip(tokens, tags):
                conll_lines.append('%s\tO\tO\t%s' % (token, tag))
            conll_lines.append('')


    if specify_st:
        filepath = 'data/MedMentions/st21pv/custom/mm_ner_sts.%s.conll' % split_label
    else:
        filepath = 'data/MedMentions/st21pv/custom/mm_ner_ent.%s.conll' % split_label

    logging.info('Writing CONLL - %s ...' % filepath)
    with open(filepath, 'w') as f:
        for line in conll_lines:
            f.write('%s\n' % line)
    
