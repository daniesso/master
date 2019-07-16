mkdir encdec-large
mkdir rnnpb-large
mkdir encdec
mkdir rnnpb

wmt=../../data/wmt

train_wmt="--train_set1=$wmt/train.en --train_set2=$wmt/train.de --dev_set1=$wmt/dev_short.en --dev_set2=$wmt/dev_short.de"

test_wmt="--test_set=$wmt/test.en"

run='python3 -u ../../nmt.py'

training='--do_training --learning_rate=0.001 --batch_size=64 --max_trans_ratio=1.5 --gradient_clip=1.0 --beam_size=10 --early_stopping_steps=10 --num_dev_prints=0 --checkpoint_all --vocab1_max=30000 --vocab2_max=30000'

testing='--do_testing --device=cpu'

rnnpb_training='--pb_learning_rate=0.01 --binding_strength=1.0 --bind_hard --max_recog_epochs=100 --num_PB=1024 --p_reset=0.10 --dropout=0.40'
encdec_training='--reverse_source'

$run $training $train_wmt $encdec_training --model=encdec --working_dir=encdec-large --embedding_size=512 --units=1024 --num_layers=4 &> encdec-large/train_log.txt
$run $testing $test_wmt --working_dir=encdec-large &> encdec-large/test_log.txt &

$run $training $train_wmt $rnnpb_training --model=rnnpbnmt --working_dir=rnnpb-large --embedding_size=512 --units=1024 --num_layers=4 &> rnnpb-large/train_log.txt
$run $testing $test_wmt --working_dir=rnnpb-large &> rnnpb-large/test_log.txt &

$run $training $train_wmt $encdec_training --model=encdec --working_dir=encdec --embedding_size=128 --units=256 --num_layers=2 &> encdec/train_log.txt
$run $testing $test_wmt --working_dir=encdec &> encdec/test_log.txt &

$run $training $train_wmt $rnnpb_training --model=rnnpbnmt --working_dir=rnnpb --embedding_size=128 --units=256 --num_layers=2 &> rnnpb/train_log.txt
$run $testing $test_wmt --working_dir=rnnpb &> rnnpb/test_log.txt &
