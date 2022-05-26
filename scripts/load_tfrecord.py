import tensorflow as tf
import sys
import functools

file_path = sys.argv[1]
tf.compat.v1.disable_eager_execution()
features = dict(word_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        entity_labels=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        mention_boundaries=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        page_id=tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),)

dataset = tf.data.TFRecordDataset([file_path],
        compression_type="GZIP",
        num_parallel_reads=1,)

#dataset = dataset.shard(1, 0)
dataset = dataset.map(functools.partial(tf.io.parse_single_example, features=features))
it = tf.compat.v1.data.make_one_shot_iterator(dataset)
it = it.get_next()

with tf.compat.v1.Session() as sess:
    for _ in range(2):
        obj = sess.run(it)
        print (obj)
