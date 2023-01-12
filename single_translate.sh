PROBLEM=iwslt14_de_en

MODE=align_train_multiple
#align_train_multiple
#align_train_multiple_c_both
#align_train_multiple_ll_both
#align_train_single
#align_train_single_c_both
#align_train_single_ll_both
CLASSIFIER=multiple
#single
#multiple

MODLE_PATH=checkpoints/$PROBLEM/$MODE
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined/
CKPT='checkpoint_best.pt'
LAYER=6

python -W ignore interactive.py \
    $DATA_PATH \
    --path $MODLE_PATH/$CKPT \
    --beam 5 --source-lang de --target-lang en \
    --classifier $CLASSIFIER  \
    --layer $LAYER \
    --remove-bpe \