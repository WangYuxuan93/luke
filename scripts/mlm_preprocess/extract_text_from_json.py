import json
import ujson
import argparse
import os

parser = argparse.ArgumentParser(description="Filter wikidata by entity vocab")
parser.add_argument("interval", type=str, help="language interval (e.g., 0:5)")
parser.add_argument("--output", type=str, help="entity vocab file path")

args = parser.parse_args()

lcs = ["ar","bn","de","nl","el","en","es","fi","fr","hi","id","it","ja","ko","pl","pt","ru","sv","sw","te","th","tr","vi","zh"]

prefix = "/work/wiki/wiki_json"
start, end = [int(x) for x in args.interval.strip().split("-")]

for lc_id in range(start, end):
    lc = lcs[lc_id] 
    lan_path = os.path.join(prefix, lc)
    output_path = os.path.join(args.output, lc+"wiki.txt")
    if os.path.exists(output_path):
        print ("Output file {} already exist!".format(output_path))
        exit()
    fo = open(output_path, "w")
    subs = os.listdir(lan_path)
    for sub_prefix in subs:
        subdir_path = os.path.join(lan_path, sub_prefix)
        if not os.path.exists(subdir_path):
            print ("Path {} does not exist!".format(subdir_path))
            exit() 
        files = os.listdir(subdir_path)
        for file_name in sorted(files):
            file = os.path.join(subdir_path, file_name)
            print ("Loading from {}".format(file))
            with open(file, "r") as f:
                data = f.read().strip().split("\n")
                for n, line in enumerate(data):
                    line = line.strip().encode("utf-8")
                    obj = ujson.loads(line)
                    if obj["text"] == "":
                        continue
                    fo.write(obj["text"])
