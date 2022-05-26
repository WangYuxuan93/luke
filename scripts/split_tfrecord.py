import tensorflow as tf
import sys
import functools
import argparse
import json
import os

def reader(file_path):
    tf.compat.v1.disable_eager_execution()
    features = dict(word_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    entity_labels=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    mention_boundaries=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    page_id=tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),)

    dataset = tf.data.TFRecordDataset([file_path],
                                    compression_type="GZIP",
                                    num_parallel_reads=1,)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--num_splits", type=int, default=1000)
    args = parser.parse_args()

    file_path = os.path.join(args.data_path, "dataset.tf")
    meta_path = os.path.join(args.data_path, "metadata.json")
        
    itr = reader(file_path)
    for i in range(5):
        item = itr.__next__()
        print (item)

if __name__ == "__main__":
    main()
