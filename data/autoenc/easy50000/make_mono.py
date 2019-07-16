
easy50000 = open("train.en", "r").readlines()

mono = open("../easy-full/train.en", "r").readlines()

s = set(easy50000)
mono = [l for l in mono if l not in s]

with open("mono.en", "w") as f:
    f.writelines(mono)
