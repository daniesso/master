mkdir rnnpb50000-0.10-0.10
mkdir rnnpb50000-0.10-0.20
mkdir rnnpb50000-0.10-0.30
mkdir rnnpb50000-0.10-0.40
mkdir rnnpb50000-0.10-0.50
mkdir rnnpb50000-0.20-0.10
mkdir rnnpb50000-0.20-0.20
mkdir rnnpb50000-0.20-0.30
mkdir rnnpb50000-0.20-0.40
mkdir rnnpb50000-0.20-0.50
mkdir rnnpb50000-0.30-0.10
mkdir rnnpb50000-0.30-0.20
mkdir rnnpb50000-0.30-0.30
mkdir rnnpb50000-0.30-0.40
mkdir rnnpb50000-0.30-0.50

easy50000=../../data/autoenc/easy50000

train_easy50000="--train_set1=$easy50000/train.en --train_set2=$easy50000/train.en --dev_set1=$easy50000/dev.en --dev_set2=$easy50000/dev.en"

test_easy50000="--test_set=$easy50000/test.en"

run='python3 -u ../../nmt.py'

training='--do_training --embedding_size=128 --units=256 --num_layers=2 --learning_rate=0.001 --batch_size=64 --max_trans_ratio=1.5 --gradient_clip=1.0 --beam_size=10 --early_stopping_steps=20 --num_dev_prints=0 --checkpoint_all'

testing='--do_testing --device=cpu'

rnnpb_training='--pb_learning_rate=0.01 --binding_strength=1.0 --bind_hard --max_recog_epochs=100 --num_PB=1024'

# easy50000
$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.10 --dropout=0.10 --working_dir=rnnpb50000-0.10-0.10 &> rnnpb50000-0.10-0.10/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.10-0.10   &> rnnpb50000-0.10-0.10/test_log.txt &

$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.10 --dropout=0.20 --working_dir=rnnpb50000-0.10-0.20 &> rnnpb50000-0.10-0.20/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.10-0.20   &> rnnpb50000-0.10-0.20/test_log.txt &

$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.10 --dropout=0.30 --working_dir=rnnpb50000-0.10-0.30 &> rnnpb50000-0.10-0.30/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.10-0.30   &> rnnpb50000-0.10-0.30/test_log.txt &

$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.10 --dropout=0.40 --working_dir=rnnpb50000-0.10-0.40 &> rnnpb50000-0.10-0.40/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.10-0.40   &> rnnpb50000-0.10-0.40/test_log.txt &

$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.10 --dropout=0.50 --working_dir=rnnpb50000-0.10-0.50 &> rnnpb50000-0.10-0.50/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.10-0.50   &> rnnpb50000-0.10-0.50/test_log.txt &



$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.20 --dropout=0.10 --working_dir=rnnpb50000-0.20-0.10 &> rnnpb50000-0.20-0.10/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.20-0.10   &> rnnpb50000-0.20-0.10/test_log.txt &

$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.20 --dropout=0.20 --working_dir=rnnpb50000-0.20-0.20 &> rnnpb50000-0.20-0.20/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.20-0.20   &> rnnpb50000-0.20-0.20/test_log.txt &

$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.20 --dropout=0.30 --working_dir=rnnpb50000-0.20-0.30 &> rnnpb50000-0.20-0.30/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.20-0.30   &> rnnpb50000-0.20-0.30/test_log.txt &

$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.20 --dropout=0.40 --working_dir=rnnpb50000-0.20-0.40 &> rnnpb50000-0.20-0.40/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.20-0.40   &> rnnpb50000-0.20-0.40/test_log.txt &

$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.20 --dropout=0.50 --working_dir=rnnpb50000-0.20-0.50 &> rnnpb50000-0.20-0.50/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.20-0.50   &> rnnpb50000-0.20-0.50/test_log.txt &



$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.30 --dropout=0.10 --working_dir=rnnpb50000-0.30-0.10 &> rnnpb50000-0.30-0.10/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.30-0.10   &> rnnpb50000-0.30-0.10/test_log.txt &

$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.30 --dropout=0.20 --working_dir=rnnpb50000-0.30-0.20 &> rnnpb50000-0.30-0.20/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.30-0.20   &> rnnpb50000-0.30-0.20/test_log.txt &

$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.30 --dropout=0.30 --working_dir=rnnpb50000-0.30-0.30 &> rnnpb50000-0.30-0.30/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.30-0.30   &> rnnpb50000-0.30-0.30/test_log.txt &

$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.30 --dropout=0.40 --working_dir=rnnpb50000-0.30-0.40 &> rnnpb50000-0.30-0.40/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.30-0.40   &> rnnpb50000-0.30-0.40/test_log.txt &

$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --p_reset=0.30 --dropout=0.50 --working_dir=rnnpb50000-0.30-0.50 &> rnnpb50000-0.30-0.50/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.30-0.50   &> rnnpb50000-0.30-0.50/test_log.txt &
