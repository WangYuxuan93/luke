import sys
import functools
import argparse
import json
import os
import random
import math
from tqdm import tqdm

def reader(file_path):
    with open(file_path, "r") as fi:
        data = fi.read().strip().split("\n")
    size = len(data)
    return size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--data_paths", type=str, required=True)
    parser.add_argument("--batch", type=int, default=256, required=True)
    parser.add_argument("--epoch", type=int, default=100, required=True)
    parser.add_argument("--output", type=str)
    #parser.add_argument("--smoothing_factor", type=float, default=0.7)
    #parser.add_argument("--seed", type=int, default=42)
    #parser.add_argument("--parallel_reader", type=int, default=32)
    #parser.add_argument("--shuffle_buffer_size", type=int, default=-1)

    args = parser.parse_args()
    with open(args.data_paths, "r") as fi:
        data_paths = json.load(fi)

    lan_sizes = {}
    for lc in data_paths:
        lan_sizes[lc] = 0
        with tqdm(total=len(data_paths[lc]), desc=lc) as pbar:
            for suffix in data_paths[lc]:
                file_path = os.path.join(args.data_dir, suffix)
                if os.path.isfile(file_path) and file_path.endswith(".txt"):
                    file_size = reader(file_path)
                    lan_sizes[lc] += file_size
                    pbar.update()
                else:
                    print ("Not exists", file_path)

    with open(args.output, "w") as fo:
        json.dump(lan_sizes, fo, indent=4)
    total_examples = sum(lan_sizes.values())
    print (lan_sizes)
    print ("Total examples = {}, batch size = {}, epochs = {}".format(total_examples, args.batch, args.epoch))
    total_steps = math.ceil(total_examples / args.batch * args.epoch)
    print ("Total steps = {}".format(total_steps))
    """
    smoothing_factor = args.smoothing_factor
    print ("Computing sampling rates with smoothing factor {}".format(smoothing_factor))
    print (lan_sizes)
    
    smoothed_sizes = {lc:size ** smoothing_factor for lc, size in lan_sizes.items()}
    size_sum = sum(smoothed_sizes.values())
    sample_rate = {lc:float(size) / size_sum for lc, size in smoothed_sizes.items()}
    print (sample_rate)
    #metadata["number_of_files"] = num_files
    #metadata["file_names:"] =  file_names
    with open(args.output, "w") as f:
        json.dump(sample_rate, f, indent=2)
    """

if __name__ == "__main__":
    main()
