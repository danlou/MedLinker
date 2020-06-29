# Run allennlp training locally

#
# edit these variables before running script
DATASET='mm_full_conll2003_fixed'
TASK='ner'
# with_finetuning='_finetune'  # or '' for not fine tuning
# with_finetuning=''  # or '' for not fine tuning
# dataset_size=999999

# export BERT_VOCAB=data/bert/scibert_scivocab_cased/vocab.txt
# export BERT_WEIGHTS=data/bert/scibert_scivocab_cased/weights.tar.gz

export BERT_VOCAB=data/bert/scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=data/bert/scibert_scivocab_uncased/weights.tar.gz

# export BERT_VOCAB=/mnt/E68C68028C67CC1D/projects/umls-contextual-linker/data/NCBI_BERT_pubmed_uncased_large/vocab.txt
# export BERT_WEIGHTS=/mnt/E68C68028C67CC1D/projects/umls-contextual-linker/data/NCBI_BERT_pubmed_uncased_large/pytorch_model.bin.tar.gz

# export BERT_VOCAB=data/bert/biobert_v1.1_pubmed/vocab.txt
# export BERT_WEIGHTS=data/bert/biobert_v1.1_pubmed/pytorch_model.bin.tar.gz

# export DATASET_SIZE=dataset_size

# CONFIG_FILE=allennlp_config/"$TASK""$with_finetuning".json
CONFIG_FILE=allennlp_config/mm_ner.json

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=true
export TRAIN_PATH=data/$TASK/$DATASET/train.txt
export DEV_PATH=data/$TASK/$DATASET/dev.txt
export TEST_PATH=data/$TASK/$DATASET/test.txt

export CUDA_DEVICE=0

python -m allennlp.run train $CONFIG_FILE --include-package scibert -s "$@"
