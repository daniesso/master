import sys, getopt, os
import json
import re
import random

if "--device=cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from misc.dataset import Dataset
from models.encdec import EncDec
from models.rnnpbnmt import RNNPBNMT

models = {"encdec": EncDec, "rnnpbnmt" : RNNPBNMT}

def set_device(device):
    devices = tf.config.experimental.get_visible_devices()
    device = device.lower()

    if len(device) == 3 and device == 'cpu':
        tf.config.experimental.set_visible_devices(devices[0])
    elif len(device) == 4 and device[:3] == 'gpu':
        gpu = int(device[3])
        tf.config.experimental.set_visible_devices([devices[0], devices[1+gpu]])


def training(model_params, model_name, train_set1, train_set2, dev_set1, dev_set2, working_dir, warm_start = False, length_limit = 10**3, vocab1_max=10**6, vocab2_max=10**6, sample=None, reverse_source=False, max_epochs = 0, early_stopping_steps = 0, easy_subset=None, mono_set1 = None, mono_set2 = None):

    print("Starting training procedure.")

    model_name = model_name.lower()
    assert model_name in models
    
    batch_size = model_params["batch_size"]

    vocab_path = os.path.join(working_dir, "vocab")

    checkpoints_already_exists = os.path.isfile(os.path.join(working_dir, "checkpoints", "checkpoint"))
    vocab_already_exists = os.path.isfile(vocab_path)
    if warm_start and (not checkpoints_already_exists or not vocab_already_exists):
        print("Error: Warm start was specified, but working_dir lacks checkpoints or vocab")
        exit(1)
    elif not warm_start and (checkpoints_already_exists or vocab_already_exists):
        print("Error: Working directory contains checkpoints or vocab that would be overwritten.")
        exit(1)


    print("Loading training set...")
    vocab = vocab_path if warm_start else None
    train = Dataset(path1=train_set1, path2=train_set2, batch_size = batch_size, 
                    vocab1_max = vocab1_max, vocab2_max=vocab2_max, length_limit = length_limit, 
                    sample=sample, reverse_source=reverse_source, vocab = vocab, easy_subset=easy_subset)

    if not warm_start:
        print("Saving vocab defined by training set...")
        train.save_vocab(vocab_path)

    print("Loading development set...")
    dev = Dataset(path1=dev_set1, path2=dev_set2, batch_size = batch_size, 
                  length_limit = length_limit, vocab = (train.first, train.second), 
                  reverse_source=reverse_source)

    assert not ((not mono_set1) ^ (not mono_set2))
    if mono_set1:
        print("Loading mono set...")
        if model_name != "rnnpbnmt":
            print("Error: Only RNNPBNMT supports monolingual training")
            exit(1)

        offset = len(train.X)
        bind_hard = model_params["bind_hard"]

        mono = Dataset(path1=mono_set1, path2=mono_set2, batch_size=batch_size,
                       length_limit = length_limit, vocab = (train.first, train.second),
                       reverse_source=reverse_source, mono = (offset, bind_hard))

        model_params["num_training_sequences"] = len(train.X) + (len(mono.X) if not bind_hard else 2*len(mono.X))
    else:
        mono = None
        model_params["num_training_sequences"] = len(train.X)

    if not warm_start:
        print("Writing params file...")
        params_file = os.path.join(working_dir, "model.json")
        with open(params_file, "w") as f:
            model_json = {
                          "model_name": model_name,
                          "length_limit": length_limit,
                          "vocab1_max": vocab1_max,
                          "vocab2_max": vocab2_max,
                          "reverse_source": reverse_source,
                          "model_params": model_params
                          }

            f.write(json.dumps(model_json))

    print("Instantiating model...")
    model = models[model_name](vocab1_size = train.first.vocab_size, vocab2_size = train.second.vocab_size, working_dir=working_dir, **model_params)

    print("Ready for training.\n")
    print("Training summery:")
    print("\n=== Data ===")
    
    print("\nTraining set:")
    train.summarize()

    print("\n\nDevelopment set:")
    dev.summarize()

    print("\n\n=== Model ===")
    print("Name:", model_name)
    model.summarize()

    print("\n\n=== Training ===")
    print("Max epochs:", max_epochs)
    print("Early stopping steps:", early_stopping_steps)
    print("Warm start:", warm_start)
    print()

    maximized_dev = model.do_training(train, dev, mono, max_epochs = max_epochs, early_stopping_steps = early_stopping_steps, warm_start=warm_start)

    if maximized_dev:
        print("Finished training after development set stopped improving.")
    else:
        print("Finished training after exceeding max epochs.")


def testing(test_set, working_dir, outfile="test_trans.txt"):

    print("Starting testing.")

    output_file = os.path.join(working_dir, outfile)
    if os.path.isfile(output_file):
        print("Error: test_trans.txt already exists and would be overwritten.")

    print("Loading model.json...")
    with open(os.path.join(working_dir, "model.json"), "r") as f:
        model_json = json.loads(f.read())

    model_name = model_json["model_name"]
    reverse_source = model_json["reverse_source"]
    model_params = model_json["model_params"]

    test = Dataset(path1=test_set, path2=test_set, batch_size = 1, vocab=os.path.join(working_dir, "vocab"),
                   reverse_source=reverse_source, shuffle=False)

    model = models[model_name](vocab1_size = test.first.vocab_size, vocab2_size = test.second.vocab_size, 
                               working_dir=working_dir, **model_params)

    translations = model.do_testing(test)

    with open(output_file, "w") as f:
        f.write("\n".join(translations))

    print("Testing finished.")


def query(q, workdir):

    q = q.strip()

    replace = {"'" : " &apos;", '"' : "&quot;"}

    for k, v in replace.items():
        q = q.replace(k, v)

    # Insert space before punctuations
    q = re.sub(r"([?.!,])", r" \1 ", q)
    q = re.sub(r'[" "]+', " ", q)

    print("Query after tokenization:", q)

    folder = os.path.join(workdir, "query_tmp")
    query_file = os.path.join(folder, "query.txt")
    outfile = os.path.join("query_tmp", "result.txt")

    if not os.path.exists(folder):
        os.mkdir(folder)

    with open(query_file, "w") as f:
        f.write(q)

    testing(query_file, workdir, outfile = outfile)

    with open(os.path.join(workdir, outfile), "r") as f:
        print("Translation:", f.read().strip())


def load_and_consolidate_warm_start_params(optd):
    check_require(optd, ["working_dir"])

    working_dir = optd["working_dir"]

    model_json = os.path.join(working_dir, "model.json")
    if not os.path.isfile(model_json):
        print("Error: Warm start was specified, but working dir does not contain model.json")
        exit(1)

    with open(model_json, "r") as f:
        model_json = json.loads(f.read())
        model_params = model_json.pop("model_params")
        model_json.update(model_params)

    inconsistent = [k for k in optd.keys() if k in model_json and optd[k] != model_json[k]]
    if inconsistent:
        print("Error: Some arguments were inconsistent with regard to model.json:" + "\n- "+ "\n- ".join(inconsistent))
        exit(1)
    else:
        optd.update(model_json)

    return optd

def check_require(opts, requirements):

    missing = [x for x in requirements if x not in opts]

    if missing:
        print("Error: Missing keywords. " + " ".join("--"+x for x in missing))
        exit(1)


def normalize_arguments(opts):
    """
    Convert command line arguments to bool, int or float, or let them remain as strings.
    """

    optd = {k.strip("--"):v for k,v in opts}

    for k,v in optd.items():
        if v == "":
            optd[k] = True
            continue
        
        try:
            optd[k] = int(v)
            continue
        except: pass
        try:
            optd[k] = float(v)
            continue
        except: pass

    return optd

model_param_names = ["embedding_size", "units", "num_layers", "learning_rate", "batch_size", "max_trans_ratio", "num_dev_prints", "gradient_clip", "beam_size"]
model_flags = ["checkpoint_all"]
model_optional = ["dropout"]

rnnpb_param_names = ["binding_strength", "num_PB", "pb_learning_rate", "max_recog_epochs"]
rnnpb_flags = ["bind_hard", "autoencode"]
rnnpb_optional = ["sigma", "p_reset", "p_mono"]

def get_model_params(optd):

    if "num_dev_prints" not in optd:
        optd["num_dev_prints"] = 0

    check_require(optd, model_param_names)

    params = model_param_names + [x for x in model_flags+model_optional if x in optd]

    if optd["model_name"] == "rnnpbnmt":
        check_require(optd, rnnpb_param_names)
        params = params + rnnpb_param_names + [x for x in rnnpb_flags+rnnpb_optional if x in optd]

    return {x : optd[x] for x in params}


def main(args):

    try: 
        opts, args = getopt.getopt(args, "", ["do_training", "train_set1=", "train_set2=", "dev_set1=", "dev_set2=", "do_testing", "test_set=", "model_name=", "vocab1_max=", "vocab2_max=", "sample=", "max_epochs=", "early_stopping_steps=", "working_dir=", "warm_start", "reverse_source", "query=", "beam_size=", "easy_subset=", "device=", "mono_set1=", "mono_set2="] + [x+"=" for x in model_param_names+rnnpb_param_names+rnnpb_optional+model_optional]+model_flags+rnnpb_flags)

    except getopt.GetoptError as e:
        print("Error:", e)
        exit(1)

    optd = normalize_arguments(opts)

    if "device" in optd:
        set_device(optd["device"])

    do_training = ("do_training" in optd)
    do_testing = ("do_testing" in optd)

    # Check necessary arguments first
    if do_training:
        do_training = True

        if "warm_start" in optd: 
            optd = load_and_consolidate_warm_start_params(optd)

        train_require = ["model_name", "train_set1", "train_set2", "dev_set1", "dev_set2", "working_dir"]
        check_require(optd, train_require)

        train_optional = ["length_limit", "vocab1_max", "vocab2_max", "sample", "max_epochs", "early_stopping_steps", 
                          "warm_start", "reverse_source", "easy_subset", "mono_set1", "mono_set2"]

        model_params = get_model_params(optd)

    if do_testing:
        do_testing = True
        test_require = ["test_set", "working_dir"]
        check_require(optd, test_require)


    if do_training:
        training_args = {x : optd[x] for x in train_require}
        training_args.update({x : optd[x] for x in train_optional if x in optd})

        training(model_params, **training_args)

    if do_testing:
        testing(**{x : optd[x] for x in test_require})

    if "query" in optd:
        check_require(optd, ["working_dir"])

        query(optd["query"], optd["working_dir"])


if __name__ == '__main__':
    main(sys.argv[1:])
