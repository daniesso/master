mkdir rnnpb50000-hard
mkdir rnnpb50000-100
mkdir rnnpb50000-10
mkdir rnnpb50000-1
mkdir rnnpb50000-0.1
mkdir rnnpb50000-0.01
mkdir rnnpb50000-0.001
mkdir rnnpb50000-0.0001

easy50000=../../data/autoenc/easy50000

train_easy50000="--train_set1=$easy50000/train.en --train_set2=$easy50000/train.en --dev_set1=$easy50000/dev.en --dev_set2=$easy50000/dev.en"

test_easy50000="--test_set=$easy50000/test.en"

run='python3 -u ../../nmt.py'

training='--do_training --embedding_size=128 --units=256 --num_layers=2 --learning_rate=0.001 --batch_size=64 --max_trans_ratio=1.5 --gradient_clip=1.0 --beam_size=10 --early_stopping_steps=20 --num_dev_prints=0 --checkpoint_all'

testing='--do_testing --device=cpu'

rnnpb_training='--num_PB=128 --pb_learning_rate=0.01 --max_recog_epochs=100'

# easy50000
$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --binding_strength=1.0    --working_dir=rnnpb50000-hard --bind_hard &>  rnnpb50000-hard/train_log.txt
$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --binding_strength=100    --working_dir=rnnpb50000-100    &> rnnpb50000-100/train_log.txt
$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --binding_strength=10     --working_dir=rnnpb50000-10     &> rnnpb50000-10/train_log.txt
$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --binding_strength=1      --working_dir=rnnpb50000-1      &> rnnpb50000-1/train_log.txt
$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --binding_strength=0.1    --working_dir=rnnpb50000-0.1    &> rnnpb50000-0.1/train_log.txt
$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --binding_strength=0.01   --working_dir=rnnpb50000-0.01   &> rnnpb50000-0.01/train_log.txt
$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --binding_strength=0.001  --working_dir=rnnpb50000-0.001  &> rnnpb50000-0.001/train_log.txt
$run $training $train_easy50000 $rnnpb_training --model=rnnpbnmt --binding_strength=0.0001 --working_dir=rnnpb50000-0.0001 &> rnnpb50000-0.0001/train_log.txt

$run $testing $test_easy50000 --working_dir=rnnpb50000-hard &> rnnpb50000-hard/test_log.txt &
$run $testing $test_easy50000 --working_dir=rnnpb50000-10   &> rnnpb50000-10/test_log.txt &
$run $testing $test_easy50000 --working_dir=rnnpb50000-1    &> rnnpb50000-1/test_log.txt &
$run $testing $test_easy50000 --working_dir=rnnpb50000-0.1  &> rnnpb50000-0.1/test_log.txt &
