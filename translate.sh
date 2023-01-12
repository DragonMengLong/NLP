gpu=0

model=transformer
PROBLEM=iwslt14_de_en
ARCH=depth_adaptive_iwslt_de_en
NUM=5
who=test

LAYER=0
MODE=align_train_multiple_ll_both
#align_train_multiple
#align_train_multiple_c_both
#align_train_multiple_ll_both
#align_train_single
#align_train_single_c_both
#align_train_single_ll_both

CLASSIFIER=multiple
#single
#multiple

DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined/
OUTPUT_PATH=checkpoints/$PROBLEM/$MODE 

BEAM_SIZE=5
LPEN=1.0
CKPT_ID=$(echo $CKPT | sed 's/checkpoint//g' | sed 's/\.pt//g' | sed 's/^_//g')

export CUDA_VISIBLE_DEVICES=$gpu
CKPT='checkpoint_last.pt'


echo $CKPT_ID
if [ -n "$cpu" ]; then
        use_cpu=--cpu
fi

python3 -W ignore generate.py \
    $DATA_PATH \
    --path $OUTPUT_PATH/$CKPT \
    --batch-size 128 \
    --beam $BEAM_SIZE \
    --lenpen $LPEN \
    --remove-bpe \
    --gen-subset $who \
    --log-format simple \
    --source-lang de \
    --target-lang en \
    --layer $LAYER \
    --classifier $CLASSIFIER  \
    --output $OUTPUT_PATH/${BEAM_SIZE}_${who}_hypo.txt \
    | tee $OUTPUT_PATH/translate-$who-$LAYER.log
echo -n $CKPT_ID ""
# $use_cpu \
# tail -n 1 $OUTPUT_PATH/res.txt
#     --quiet \
# --depth-select-target seq_LL \