import tensorflow as tf
import sys
import functools
import argparse
import json
import os
import random
from tqdm import tqdm

def reader(file_path, num_parallel_reader=32):
    #tf.compat.v1.disable_eager_execution()
    features = dict(word_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    entity_labels=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    mention_boundaries=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    page_id=tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),)

    dataset = tf.data.TFRecordDataset([file_path],
                                    compression_type="GZIP",
                                    num_parallel_reads=num_parallel_reader,)
    #size = len(list(dataset))
    size = sum(1 for _ in dataset) #len(list(dataset))
    return size
    """
    print ("buffer size={}, seed={}".format(shuffle_buffer_size, shuffle_seed))
    #dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    dataset = dataset.map(functools.partial(tf.io.parse_single_example, features=features))
    it = tf.compat.v1.data.make_one_shot_iterator(dataset)
    it = it.get_next()
    with tf.compat.v1.Session() as sess:
        try:
            while True:
                obj = sess.run(it)
                yield obj
        except tf.errors.OutOfRangeError:
            pass
    """

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--smoothing_factor", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel_reader", type=int, default=32)
    #parser.add_argument("--shuffle_buffer_size", type=int, default=-1)

    #parser.add_argument("--shuffle", type=int, default=1000)
    args = parser.parse_args()
    
    lan_dirs = os.listdir(args.data_dir)
    lc_dirs = []
    for lc in lan_dirs:
        path = os.path.join(args.data_dir, lc)
        if os.path.isdir(path):
            lc_dirs.append(path)
    lc_dirs = sorted(lc_dirs)
    print (lc_dirs)
    assert len(lc_dirs) == 24
 
    #print ("Total number of items={}\nsplitting into {} files each containing {} items".format(num_examples, num_splits, chunk_size))
    dataset_paths = {}
    lan_sizes = {}
    for lc_dir in lc_dirs:
        lc = lc_dir[-2:]
        print ("Loading {} datasets from {}".format(lc, lc_dir))
        dataset_paths[lc] = []
        lan_sizes[lc] = 0
        meta_path = os.path.join(lc_dir, "metadata.json")
        if os.path.exists(meta_path):
            metadata = json.load(open(meta_path,"r"))
            lan_sizes[lc] = metadata["number_of_items"]
            for name in os.listdir(lc_dir):
                file_path = os.path.join(lc_dir, name)
                if os.path.isfile(file_path) and file_path.endswith(".tf"):
                    dataset_paths[lc].append(lc+"/"+name)
            dataset_paths[lc] = sorted(dataset_paths[lc])
            assert len(dataset_paths[lc]) == metadata["number_of_files"]
        else:
            with tqdm(total=len(os.listdir(lc_dir))-1) as pbar:
                for name in os.listdir(lc_dir):
                    file_path = os.path.join(lc_dir, name)
                    if os.path.isfile(file_path) and file_path.endswith(".tf"):
                        #print ("Reading ", file_path)
                        file_size = reader(file_path, num_parallel_reader=args.parallel_reader)
                        #print (file_size)
                        lan_sizes[lc] += file_size
                        pbar.update()
                    else:
                        print ("Not exists", file_path)
                        #exit() 
        #print (lan_sizes)
        #print (dataset_paths)
        #exit()
    #print ("Shuffle buffer size: {}, random seed: {}".format(shuffle_buffer_size, args.seed))

    smoothing_factor = args.smoothing_factor
    print ("Computing sampling rates with smoothing factor {}".format(smoothing_factor))
    print (lan_sizes)
    
    smoothed_sizes = {lc:size ** smoothing_factor for lc, size in lan_sizes.items()}
    size_sum = sum(smoothed_sizes.values())
    sample_rate = {lc:float(size) / size_sum for lc, size in smoothed_sizes.items()}
    print (sample_rate)
    #metadata["number_of_files"] = num_files
    #metadata["file_names:"] =  file_names
    sr_file = os.path.join(args.output_dir, "sampling_rate.json")
    with open(sr_file, "w") as f:
        json.dump(sample_rate, f, indent=2)

    dataset_file = os.path.join(args.output_dir, "dataset_paths.json")
    with open(dataset_file, "w") as f:
        json.dump(dataset_paths, f, indent=2)

if __name__ == "__main__":
    main()
