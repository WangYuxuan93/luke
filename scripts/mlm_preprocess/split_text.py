import json
import ujson
import argparse
import os
import sys

def read_by_line(input_file, output_dir, lc="en", chunk_size=10000):
    file_id = 0
    with open(input_file, "r") as fi:
        line = fi.readline()
        lines = []
        while line:
            lines.append(line.strip())
            if len(lines) == chunk_size:
                output_path = os.path.join(output_dir, "{}.{}.txt".format(lc, file_id))
                with open(output_path, "w") as fo:
                    fo.write("\n".join(lines))
                    file_id += 1
                    lines = []
            line = fi.readline()
    
    meta_file = os.path.join(output_dir, "metadata.{}.json".format(lc))
    with open(meta_file, "w") as fm:
        metadata = {"chunk_size": chunk_size, "num_files": file_id}
        json.dump(metadata, fm, indent=4)       

parser = argparse.ArgumentParser(description="Filter wikidata by entity vocab")
parser.add_argument("interval", type=str, help="input text data file")
parser.add_argument("--output", type=str, help="output directory")
#parser.add_argument("--interval", type=str, help="(e.g., 0-4)")

args = parser.parse_args()

lcs = ["ar","bn","de","nl","el","en","es","fi","fr","hi","id","it","ja","ko","pl","pt","ru","sv","sw","te","th","tr","vi","zh"]

prefix = "/work/wiki/wiki_text"
start, end = [int(x) for x in args.interval.strip().split("-")]
chunk_sizes = [10000, 5000, 2000, 1000, 500, 200, 100]
max_files = 6000
min_files = 1000

for lc_id in range(start, end):
    lc = lcs[lc_id]
    output_dir = os.path.join(args.output, lc)
    if os.path.exists(output_dir):
        print ("Output dir {} already exist!".format(output_dir))
        exit()
    os.makedirs(output_dir) 
    
    input_file = os.path.join(prefix, lc+"wiki.txt")
    if lc == "en":
        read_by_line(input_file, output_dir, lc="en", chunk_size=10000)
        continue
    with open(input_file, "r") as fi:
        data = fi.read().strip().split("\n")
    total_lines = len(data)
    for chunk_size in chunk_sizes:
        num_files = int(total_lines / chunk_size)
        if num_files > min_files and num_files < max_files:
            break
    print ("For language {}, total_lines = {}, chunk size = {}, num_files = {}".format(lc, total_lines, chunk_size, num_files))
    sys.stdout.flush()

    offset = 0
    file_id = 0
    while offset < len(data):
       output_path = os.path.join(output_dir, "{}.{}.txt".format(lc, file_id)) 
       with open(output_path, "w") as fo:
           fo.write("\n".join(data[offset:offset+chunk_size]))
           offset += chunk_size
           file_id += 1
    meta_file = os.path.join(output_dir, "metadata.{}.json".format(lc))
    with open(meta_file, "w") as fm:
        metadata = {"chunk_size": chunk_size, "num_files": file_id}
        json.dump(metadata, fm, indent=4)
