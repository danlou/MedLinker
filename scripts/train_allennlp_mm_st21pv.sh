# Run allennlp training locally

# export BERT_VOCAB=data/bert/scibert_scivocab_uncased/vocab.txt
# export BERT_WEIGHTS=data/bert/scibert_scivocab_uncased/weights.tar.gz
# export IS_LOWERCASE=true

export BERT_VOCAB=data/bert/scibert_scivocab_cased/vocab.txt
export BERT_WEIGHTS=data/bert/scibert_scivocab_cased/weights.tar.gz
export IS_LOWERCASE=false

# export BERT_WEIGHTS=data/bert/NCBI_BERT_pubmed_uncased_large/pytorch_model.bin.tar.gz
# export BERT_VOCAB=data/bert/NCBI_BERT_pubmed_uncased_large/vocab.txt
# export IS_LOWERCASE=true

# export BERT_WEIGHTS=data/bert/biobert_v1.1_pubmed/pytorch_model.bin.tar.gz
# export BERT_VOCAB=data/bert/biobert_v1.1_pubmed/vocab.txt
# export IS_LOWERCASE=false


CONFIG_FILE=allennlp_config/mm_ner.json

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export TRAIN_PATH=data/MedMentions/st21pv/custom/conll2003/train.txt
export DEV_PATH=data/MedMentions/st21pv/custom/conll2003/dev.txt
export TEST_PATH=data/MedMentions/st21pv/custom/conll2003/test.txt

export CUDA_DEVICE=0

python -m allennlp.run train $CONFIG_FILE --include-package scibert -s "$@"
