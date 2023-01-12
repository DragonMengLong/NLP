device=0
model=transformer
PROBLEM=iwslt14_de_en
ARCH=depth_adaptive_iwslt_de_en
MODE=align_train_multiple_loss # align_train_multiple
#EPOCH=50
updates=50000
NUM=5
code_dir=code

DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined/
OUTPUT_PATH=checkpoints/$PROBLEM/$MODE

if [ ! -d $OUTPUT_PATH/$code_dir ]; then
        mkdir -p $OUTPUT_PATH/$code_dir
fi

cp ${BASH_SOURCE[0]} $OUTPUT_PATH/train.sh
cp fairseq/models/transformer.py $OUTPUT_PATH/$code_dir/
cp fairseq/criterions/label_smoothed_cross_entropy.py $OUTPUT_PATH/$code_dir/
cp fairseq/sequence_generator.py $OUTPUT_PATH/$code_dir/
cp fairseq/options.py $OUTPUT_PATH/$code_dir/
cp fairseq/tasks/fairseq_task.py $OUTPUT_PATH/$code_dir/
cp fairseq/modules/multihead_attention.py $OUTPUT_PATH/$code_dir/


cmd="python3 train.py $DATA_PATH \
  --arch $ARCH --lr 0.0007 \
  --clip-norm 0.1 --dropout 0.3 --max-tokens 8000 \
  --label-smoothing 0.1 --save-dir $OUTPUT_PATH \
  --seed 1 \
  --max-update $updates \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --attention-dropout 0.1 --relu-dropout 0.1 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
  --min-lr 1e-09 \
  --weight-decay 0.0001 \
  --update-freq 1 --no-progress-bar --log-interval 100 \
  --ddp-backend no_c10d \
  --save-interval-updates 10000 --keep-interval-updates 20 \
  --keep-last-epochs $NUM \
  --restore-file checkpoint_best.pt \
  --criterion label_smoothed_cross_entropy \
  --classifier multiple  \
  --output $OUTPUT_PATH"

# --max-epoch $EPOCH \
# --depth-select-method seq_depth --depth-select-target seq_LL 
# --lr-scheduler reduce_lr_on_plateau

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $OUTPUT_PATH/train.log 2>&1 &"
eval $cmd
tail -f $OUTPUT_PATH/train.log



