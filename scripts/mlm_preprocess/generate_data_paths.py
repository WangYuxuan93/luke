import json
import ujson
import argparse
import bz2
import os
from collections import OrderedDict

parser = argparse.ArgumentParser(description="Filter wikidata by entity vocab")
parser.add_argument("data", type=str, help="split file directory")
parser.add_argument("--output", type=str, help="output json file path")

args = parser.parse_args()

def print_json(a):
    output = json.dumps(a, indent=2)
    print (output)

lcs = ["ar","bn","de","el","en","es","fi","fr","hi","id","it","ja","ko","nl","pl","pt","ru","sv","sw","te","th","tr","vi","zh"]

meta_path = os.path.join(args.data, "wiki_raw_split")
data_path = os.path.join(args.data, "tokenized")

file_dict = OrderedDict()
for i in range(len(lcs)):
    lc = lcs[i]
    with open(os.path.join(meta_path, lc, "metadata."+lc+".json")) as fm:
        meta = json.load(fm)
        num_files = meta["num_files"]
        prefix = os.path.join(data_path, lc)
        assert os.path.exists(prefix + "/" + ".".join([lc,str(num_files-1),"txt"]))
        assert not os.path.exists(prefix + "/" + ".".join([lc,str(num_files),"txt"]))
    file_paths = []
    for j in range(num_files):
        file_paths.append(lc+"/"+".".join([lc,str(j),"txt"]))
    file_dict[lc] = file_paths
with open(args.output, "w") as fo:
    json.dump(file_dict, fo, indent=2)
