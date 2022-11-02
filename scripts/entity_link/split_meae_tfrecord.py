import tensorflow as tf
import sys
import functools
import argparse
import json
import os
import random
from tqdm import tqdm

def reader(file_path, shuffle_buffer_size=100000, num_parallel_reader=64, shuffle_seed=42):
    tf.compat.v1.disable_eager_execution()
    features = dict(word_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    entity_labels=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    mention_boundaries=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    page_id=tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),)

    dataset = tf.data.TFRecordDataset([file_path],
                                    compression_type="GZIP",
                                    num_parallel_reads=num_parallel_reader,)
    print ("buffer size={}, seed={}".format(shuffle_buffer_size, shuffle_seed))
    dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
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

def write(itr, output_dir, num_each_file, left_item_start_idx, num_exp):
    file_names = []
    options = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.GZIP)
    file_id = 0
    tf_file = os.path.join(output_dir, "dataset-"+str(file_id)+".tf")
    file_names.append(tf_file)
    writer = tf.io.TFRecordWriter(tf_file, options=options)
    num_total = 0
    
    with tqdm(total=num_exp) as pbar:
        for data in itr:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature=dict(
                        page_id=tf.train.Feature(int64_list=tf.train.Int64List(value=[data["page_id"][0]])),
                    word_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=data["word_ids"])),
                    entity_labels=tf.train.Feature(int64_list=tf.train.Int64List(value=data["entity_labels"])),
                    mention_boundaries=tf.train.Feature(int64_list=tf.train.Int64List(value=data["mention_boundaries"])),
                )
            )
        )

            writer.write(example.SerializeToString())
            num_total += 1
            # save the few left items in the last file
            if num_total % num_each_file == 0 and num_total < left_item_start_idx - 1:
                writer.close()
                file_id += 1
                tf_file = os.path.join(output_dir, "dataset-"+str(file_id)+".tf")
                file_names.append(tf_file)
                writer = tf.io.TFRecordWriter(tf_file, options=options)
            pbar.update()
        writer.close()
    print ("Total examples: {}, files: {}".format(num_total, file_id+1))
    return file_id+1, file_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel_reader", type=int, default=64)
    parser.add_argument("--shuffle_buffer_size", type=int, default=-1)

    #parser.add_argument("--shuffle", type=int, default=1000)
    args = parser.parse_args()

    shuffle = True

    file_path = os.path.join(args.data_path, "dataset.tf")
    meta_path = os.path.join(args.data_path, "metadata.json")

    metadata = json.load(open(meta_path,"r"))
    
    
    num_examples = metadata["number_of_items"]

    chunk_sizes = [10000, 5000, 2000, 1000, 500, 200, 100]
    max_files = 6000
    min_files = 1000
    chunk_size = args.chunk_size
    if chunk_size is not None:
        num_splits = int(num_examples / chunk_size)
        print ("Total number of items={}\nsplitting into {} files each containing (given) {} items".format(num_examples, num_splits, chunk_size))
    else:
        for chunk_size in chunk_sizes:
            num_splits = int(num_examples / chunk_size)
            if num_splits > min_files and num_splits < max_files:
                break
        print ("Total number of items={}\nsplitting into {} files each containing (calc) {} items".format(num_examples, num_splits, chunk_size))

    shuffle_buffer_size = args.shuffle_buffer_size if args.shuffle_buffer_size > 0 else num_examples
    
    print ("Loading from {} (with {} workers)".format(file_path, args.parallel_reader))
    print ("Shuffle buffer size: {}, random seed: {}".format(shuffle_buffer_size, args.seed))
    itr = reader(file_path, shuffle_buffer_size=shuffle_buffer_size, num_parallel_reader=args.parallel_reader, shuffle_seed=args.seed)

    #for i in range(5):
    #    item = itr.__next__()
        #print (item)
    

    #dataset = [data for data in itr]
    #
    num_left = num_examples - chunk_size * num_splits
    left_item_start_idx = num_examples - num_left
    #if shuffle:
    #    random.seed(args.seed)
    #    random.shuffle(dataset)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    num_files, file_names = write(itr, args.output_dir, chunk_size, left_item_start_idx, num_examples)

    metadata["number_of_files"] = num_files
    metadata["chunk_size"] = chunk_size
    #metadata["file_names:"] =  file_names
    out_meta_file = os.path.join(args.output_dir, "metadata.json")
    with open(out_meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
