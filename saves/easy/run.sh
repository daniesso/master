mkdir rnnpb20000
mkdir rnnpb50000
mkdir rnnpbfull
mkdir encdec20000
mkdir encdec50000
mkdir encdecfull

easy20000=../../data/easy/easy20000
easy50000=../../data/easy/easy50000
easyfull=../../data/easy/easy-full

train_easy20000="--train_set1=$easy20000/train.en --train_set2=$easy20000/train.de --dev_set1=$easy20000/dev.en --dev_set2=$easy20000/dev.de"
train_easy50000="--train_set1=$easy50000/train.en --train_set2=$easy50000/train.de --dev_set1=$easy50000/dev.en --dev_set2=$easy50000/dev.de"
train_easyfull="--train_set1=$easyfull/train.en --train_set2=$easyfull/train.de --dev_set1=$easyfull/dev.en --dev_set2=$easyfull/dev.de"

test_easy20000="--test_set=$easy20000/test.en"
test_easy50000="--test_set=$easy50000/test.en"
test_easyfull="--test_set=$easyfull/test.en"

run='python3 -u ../../nmt.py'

training='--do_training --embedding_size=128 --units=256 --num_layers=2 --learning_rate=0.001 --batch_size=64 --max_trans_ratio=1.5 --gradient_clip=1.0 --beam_size=10 --early_stopping_steps=20 --num_dev_prints=0 --checkpoint_all --vocab1_max=30000 --vocab2_max=30000'

testing='--do_testing --device=cpu'

rnnpb_training='--pb_learning_rate=0.01 --binding_strength=1.0 --bind_hard --max_recog_epochs=100 --num_PB=1024 --p_reset=0.10 --dropout=0.40'
encdec_training='--reverse_source'

# easy20000
$run $training $train_easy20000 $encdec_training --model=encdec --working_dir=encdec20000 &> encdec20000/train_log.txt
$run $testing $test_easy20000 --working_dir=encdec20000 &> encdec20000/test_log.txt &

$run $training $train_easy20000 $rnnpb_training --model=rnnpbnmt --working_dir=rnnpb20000 &> rnnpb20000/train_log.txt
$run $testing $test_easy20000 --working_dir=rnnpb20000 &> rnnpb20000/test_log.txt &


# easy50000
$run $training $train_easy50000 $encdec_training --model=encdec --working_dir=encdec50000 &> encdec50000/train_log.txt
$run $testing $test_easy50000 --working_dir=encdec50000 &> encdec50000/test_log.txt &

$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --working_dir=rnnpb50000 &> rnnpb50000/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000 &> rnnpb50000/test_log.txt &


# easyfull
$run $training $train_easyfull $encdec_training --model=encdec --working_dir=encdecfull &> encdecfull/train_log.txt
$run $testing $test_easyfull --working_dir=encdecfull &> encdecfull/test_log.txt &

$run $training $train_easyfull $rnnpb_training --model=rnnpbnmt --working_dir=rnnpbfull &> rnnpbfull/train_log.txt
$run $testing $test_easyfull --working_dir=rnnpbfull &> rnnpbfull/test_log.txt &
