import sys
import os


def get_cmd(args):
    cmd = "python scripts/split_tfrecord.py {data_dir}/{lc} --output_dir {output_dir}/{lc} --chunk_size {chunk_size} --parallel_reader {reader} --shuffle_buffer_size {buffer_size}".format(**args)
    return cmd


args = {"data_dir": "/workspace/wiki/dataset",
        "output_dir":"/workspace/wiki/dataset-split",
        "lc":"bn",
        "chunk_size":10000,
        "reader": 90,
        "buffer_size": -1,
        }

#args["data_dir"] = "data/debug"
#args["output_dir"] = "data/dataset-split"
#cmd = get_cmd(args)
#print (cmd)
#os.system(cmd)
#exit()


lc_dict = {
        "ar": 1000,
        "bn": 100,
        "de": 2000,
        "nl": 1000,
        "el": 200,
        "en": 5000,
        "es": 1000,
        "fi": 400,
        "fr": 2000,
        "hi": 100,
        "id": 400,
        "it": 1000,
        "ja": 1000,
        "ko": 1000,
        "pl": 1000,
        "pt": 1000,
        "ru": 2000,
        "sv": 1000,
        "sw": 100,
        "te": 100,
        "th": 100,
        "tr": 400,
        "vi": 1000,
        "zh": 1000,
        }

for lc, chunk_size in lc_dict.items():
    args["lc"] = lc
    args["chunk_size"] = chunk_size
    cmd = get_cmd(args)

    print (cmd)
    os.system(cmd)
    exit()
