import json
import argparse
import os

parser = argparse.ArgumentParser(description="Generate output mention candidate json file path")
parser.add_argument("prefix", type=str, help="mention candidate file directory")
parser.add_argument("--output", type=str, help="output mention candidate json file path")

args = parser.parse_args()


def write_json(a, output):
    with open(output, "w") as f:
        json.dump(a, f, indent=2)
    #output = json.dumps(a, indent=2)
    #print (output)

lcs = ["ar","bn","de","el","en","es","fi","fr","hi","id","it","ja","ko","nl","pl","pt","ru","sv","sw","te","th","tr","vi","zh"]

prefix = args.prefix
meta = {}
for i in range(len(lcs)):
    lc = lcs[i]
    path = os.path.join(prefix, lc+"wiki-mention-candidates.json")
    meta[lc] = path
write_json(meta, args.output)
