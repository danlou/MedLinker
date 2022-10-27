"""
utils for reading MedMentions original format
adapted from scispacy: https://github.com/allenai/scispacy
"""

from typing import NamedTuple, List, Iterator, Dict, Tuple
import tarfile
import atexit
import os
import shutil
import tempfile

from scispacy.file_cache import cached_path

from scispacy.umls_semantic_type_tree import construct_umls_tree_from_tsv
umls_tree = construct_umls_tree_from_tsv("data/umls_semantic_type_tree.tsv")


class MedMentionEntity(NamedTuple):
    start: int
    end: int
    mention_text: str
    mention_type: str
    umls_id: str

class MedMentionExample(NamedTuple):
    title: str
    abstract: str
    text: str
    pubmed_id: str
    entities: List[MedMentionEntity]


def process_example(lines: List[str]) -> MedMentionExample:
    """
    Processes the text lines of a file corresponding to a single MedMention abstract,
    extracts the title, abstract, pubmed id and entities. The lines of the file should
    have the following format:
    PMID | t | Title text
    PMID | a | Abstract text
    PMID TAB StartIndex TAB EndIndex TAB MentionTextSegment TAB SemanticTypeID TAB EntityID
    ...
    """
    pubmed_id, _, title = [x.strip() for x in lines[0].split("|", maxsplit=2)]
    _, _, abstract = [x.strip() for x in lines[1].split("|", maxsplit=2)]

    entities = []
    for entity_line in lines[2:]:
        _, start, end, mention, mention_type, umls_id = entity_line.split("\t")
        # mention_type = mention_type.split(",")[0]
        mention_type = max(mention_type.split(","), key=lambda x: umls_tree.get_node_from_id(x).level)
        entities.append(MedMentionEntity(int(start), int(end),
                                         mention, mention_type, umls_id))

    # compose text from title and abstract
    text = title + ' ' + abstract

    return MedMentionExample(title, abstract, text, pubmed_id, entities)

def med_mentions_example_iterator(filename: str) -> Iterator[MedMentionExample]:
    """
    Iterates over a MedMentions file, yielding examples.
    """
    with open(filename, "r") as med_mentions_file:
        lines = []
        for line in med_mentions_file:
            line = line.strip()
            if line:
                lines.append(line)
            else:
                yield process_example(lines)
                lines = []
        # Pick up stragglers
        if lines:
            yield process_example(lines)

# def read_med_mentions(filename: str):
#     """
#     Reads in the MedMentions dataset into Spacy's
#     NER format.
#     """
#     examples = []
#     for example in med_mentions_example_iterator(filename):
#         # spacy_format_entities = [(x.start, x.end, x.mention_type) for x in example.entities]
#         spacy_format_entities = [(x.start, x.end, x.mention_text, x.mention_type, x.umls_id) for x in example.entities]
#         examples.append((example.text, {"entities": spacy_format_entities}))

#     return examples


def read_full_med_mentions(directory_path: str,
                           label_mapping: Dict[str, str] = None,
                           span_only: bool = False):

    def _cleanup_dir(dir_path: str):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    resolved_directory_path = cached_path(directory_path)
    if "tar.gz" in directory_path:
        # Extract dataset to temp dir
        tempdir = tempfile.mkdtemp()
        print(f"extracting dataset directory {resolved_directory_path} to temp dir {tempdir}")
        with tarfile.open(resolved_directory_path, 'r:gz') as archive:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(archive, tempdir)
        # Postpone cleanup until exit in case the unarchived
        # contents are needed outside this function.
        atexit.register(_cleanup_dir, tempdir)

        resolved_directory_path = tempdir

    expected_names = ["corpus_pubtator.txt",
                      "corpus_pubtator_pmids_all.txt",
                      "corpus_pubtator_pmids_dev.txt",
                      "corpus_pubtator_pmids_test.txt",
                      "corpus_pubtator_pmids_trng.txt"]

    corpus = os.path.join(resolved_directory_path, expected_names[0])
    examples = med_mentions_example_iterator(corpus)

    train_ids = {x.strip() for x in open(os.path.join(resolved_directory_path, expected_names[4]))}
    dev_ids = {x.strip() for x in open(os.path.join(resolved_directory_path, expected_names[2]))}
    test_ids = {x.strip() for x in open(os.path.join(resolved_directory_path, expected_names[3]))}

    train_examples = []
    dev_examples = []
    test_examples = []

    for example in examples:
        if example.pubmed_id in train_ids:
            train_examples.append(example)

        elif example.pubmed_id in dev_ids:
            dev_examples.append(example)

        elif example.pubmed_id in test_ids:
            test_examples.append(example)

    return train_examples, dev_examples, test_examples


############################################

import itertools
import json


class MedMentionSentenceEntity(NamedTuple):
    cui: str
    st: str
    tokens: List[str]
    start: int
    end: int


def iterate_annotations(sci_nlp, dataset_examples):

    for ex in dataset_examples:

        # get sentence positions to delimit annotations to sentences
        sent_span_idxs = []
        text = sci_nlp(ex.text)
        sents = list(text.sents)

        ch_idx = 0
        # first sent will include title (due to composition expected by start/end ent indices)
        # need to handle first sent differently
        sent = sents.pop(0)
        
        # start by adding title as first sentence
        sent_span_idxs.append((0, len(ex.title)))

        # add remaining as another sentence (if any left)
        if len(sent.text) > len(ex.title) + 1:
            sent_span_idxs.append((len(ex.title) + 1, len(sent.text)))

        ch_idx += len(sent.text) + 1

        for sent in sents:
            start_idx = ch_idx
            end_idx = ch_idx + len(sent.text)
            sent_span_idxs.append((start_idx, end_idx))

            if text[end_idx] != ' ':
                ch_idx = end_idx + 1  # ws separating sentences
        # ch_idx -= 1  # fix last added ws

        for ent in ex.entities:

            # sanity check 1 - mentions match in text
            text_mention_extraction = ex.text[ent.start:ent.end]
            assert ent.mention_text == text_mention_extraction

            for sent_start, sent_end in sent_span_idxs:
                if (ent.start >= sent_start) and (ent.end <= sent_end):
                    sent = ex.text[sent_start:sent_end]

                    # adjust start and end positions
                    ent = MedMentionEntity(ent.start - sent_start,
                                           ent.end - sent_start,
                                           ent.mention_text,
                                           ent.mention_type,
                                           ent.umls_id)

                    # sanity check 2 - mentions match in sentence
                    sent_mention_extraction = sent[ent.start:ent.end]
                    assert ent.mention_text == sent_mention_extraction

                    yield (ent, sent)


# def locate_tokens(all_tokens, subset_tokens):
#     """
#     Returns a list of indices (LoL) for all mention tokens within a list of tokens (i.e. sentence tokens).
#     """
#     # tests all combinations, very slow and fails for long spans
#     # gets the job done for now, to be improved later

#     def get_idxs(elems, e):  # assumes must occurr
#         return [i for i, e_ in enumerate(elems) if e == e_]

#     def is_linear(elems):
#         # return elems == [elems[0] + i for i in range(len(elems))]
#         return all(e1 == e2 - 1 for e1, e2 in zip(elems, elems[1:]))

#     # method isn't tractable for very long lists (also very rare)
#     if len(subset_tokens) > 10:
#         return [-1]

#     all_possible_idxs = []  # indices for overlaps between all_tokens and subset
#     for token in subset_tokens:
#         if token in all_tokens:
#             all_possible_idxs.append(get_idxs(all_tokens, token))
    
#     if len(all_possible_idxs) > 0:
#         for combination in itertools.product(*all_possible_idxs):
#             combination = list(combination)
#             if is_linear(combination):  # only want indices increasing by +1
#                 return combination
    
#     return [-1]


def locate_tokens(all_tokens, subset_tokens, reserved_spans=set()):

    def get_idxs(elems, e):
        return [i for i, e_ in enumerate(elems) if e == e_]

    for t0_idx in get_idxs(all_tokens, subset_tokens[0]):
        shift_idx = t0_idx + len(subset_tokens)
        if all_tokens[t0_idx:shift_idx] == subset_tokens:
            start = t0_idx
            end = shift_idx - 1

            if (start, end) not in reserved_spans:
                return [start, end]
    
    return [-1]


def get_sent_boundaries(sci_nlp, text, title):
    """
    Returns char indices for start and end of sentences from the full text.
    The title is concatenated with the text, needs to processed as first sentence.
    """

    # start with scispacy's sentence splitting
    sents = [sent.text for sent in sci_nlp(text).sents]

    sent_span_idxs = []

    ch_idx = 0
    # first sent will include title (due to composition expected by start/end ent indices)
    # need to handle first sent differently
    sent = sents.pop(0)
    
    # start by adding title as first sentence
    sent_span_idxs.append((0, len(title) - 1))

    # add remaining as another sentence (if any left)
    if len(sent) > len(title) + 1:
        sent_span_idxs.append((len(title) + 1, len(sent) - 1))

    ch_idx += (len(sent) - 1) + 2  # skip over ws to next char, len gives +1

    for sent in sents:
        start_idx = ch_idx
        end_idx = ch_idx + (len(sent) - 1)

        # move to next char, skips ws
        try:
            if text[end_idx + 1] == ' ':
                ch_idx = end_idx + 2
            else:  # happens when sentence splitting fails
                ch_idx = end_idx + 1
        except IndexError:  # end of text
            ch_idx = end_idx
        
        sent_span_idxs.append((start_idx, end_idx))

    return sent_span_idxs


def get_sent_ents(sci_nlp, sent_tokens, sent_start, sent_end, doc_entities):

    sent_ents = []
    reserved_spans = set()
    skipped_mentions = 0  # failed locating mention
    for ent in doc_entities:
        # only interested in entities located within sentence boundaries
        if (ent.start >= sent_start) and (ent.end <= sent_end):

            mention_tokens = [tok.text for tok in sci_nlp(ent.mention_text)]
            mention_tokens_idxs = locate_tokens(sent_tokens, mention_tokens, reserved_spans)

            if -1 in mention_tokens_idxs:
                skipped_mentions += 1  # something may have gone wrong with splitting
                continue
            
            mention_token_start = mention_tokens_idxs[0]
            mention_token_end = mention_tokens_idxs[-1] + 1  # +1 for easier slicing... not sure about this choice

            if (mention_token_start, mention_token_end) not in reserved_spans:  # no overlapping/duplicate spans

                sent_ent = MedMentionSentenceEntity(cui=ent.umls_id,
                                                    st=ent.mention_type,
                                                    tokens=mention_tokens,
                                                    start=mention_token_start,
                                                    end=mention_token_end)

                sent_ents.append(sent_ent)
                reserved_spans.add((mention_token_start, mention_token_end))

    return sent_ents, skipped_mentions


# def iterate_docs_converted(split_path):

#     # load json dataset
#     with open(split_path, 'r') as json_f:
#         dataset = json.load(json_f)

#     for doc in dataset['docs']:
#         yield doc

