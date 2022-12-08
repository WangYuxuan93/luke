#import tensorflow as tf
import sys
import functools
import argparse
import json
import os
import random
from tqdm import tqdm
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch", type=int, default=256, required=True)
    parser.add_argument("--epoch", type=int, default=10, required=True)
    parser.add_argument("--smoothing_factor", type=float, default=0.7)
    #parser.add_argument("--shuffle", type=int, default=1000)
    args = parser.parse_args()
    
    lan_sizes = {}
    with open(args.data_file, "r") as f:
        data = json.load(f)
        for lc in data:
            lan_sizes[lc] = data[lc]["total"]

    total_examples = sum(lan_sizes.values())
    print (lan_sizes)
    print ("Total examples = {}, batch size = {}, epochs = {}".format(total_examples, args.batch, args.epoch))
    total_steps = math.ceil(total_examples / args.batch * args.epoch)
    print ("Total steps = {}".format(total_steps))
    #print ("Total number of items={}\nsplitting into {} files each containing {} items".format(num_examples, num_splits, chunk_size))
    smoothing_factor = args.smoothing_factor
    print ("Computing sampling rates with smoothing factor {}".format(smoothing_factor))
    
    
    smoothed_sizes = {lc:size ** smoothing_factor for lc, size in lan_sizes.items()}
    size_sum = sum(smoothed_sizes.values())
    sample_rate = {lc:float(size) / size_sum for lc, size in smoothed_sizes.items()}
    print (sample_rate)
    #metadata["number_of_files"] = num_files
    #metadata["file_names:"] =  file_names
    sr_file = os.path.join(args.output_dir, "sampling_rate.json")
    with open(sr_file, "w") as f:
        json.dump(sample_rate, f, indent=2)

if __name__ == "__main__":
    main()
