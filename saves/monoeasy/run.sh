mkdir rnnpb50000

easy50000=../../data/easy/easy50000

train_easy50000="--train_set1=$easy50000/train.en --train_set2=$easy50000/train.de --dev_set1=$easy50000/dev.en --dev_set2=$easy50000/dev.de"
train_easymono="--mono_set1=$easy50000/mono.en --mono_set2=$easy50000/mono.de"

test_easy50000="--test_set=$easy50000/test.en"

run='python3 -u ../../nmt.py'

training='--do_training --embedding_size=128 --units=256 --num_layers=2 --learning_rate=0.001 --batch_size=64 --max_trans_ratio=1.5 --gradient_clip=1.0 --beam_size=10 --early_stopping_steps=20 --num_dev_prints=0 --vocab1_max=30000 --vocab2_max=30000'

testing='--do_testing --device=cpu'

rnnpb_training='--pb_learning_rate=0.01 --binding_strength=1.0 --bind_hard --max_recog_epochs=100 --num_PB=1024 --p_reset=0.10 --dropout=0.40 --p_mono=0.40'

# easy50000
$run $training $train_easy50000 $train_easymono $rnnpb_training --model=rnnpbnmt --working_dir=rnnpb50000 &> rnnpb50000/train_log.txt
$run $testing $test_easy50000 --working_dir=rnnpb50000 &> rnnpb50000/test_log.txt &
