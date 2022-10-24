import functools
import itertools
import json
import logging
import multiprocessing
import os
import random
from contextlib import closing
from multiprocessing.pool import Pool
from typing import Optional

import click
import numpy as np
import tensorflow as tf
from sentencepiece import SentencePieceProcessor
from luke.utils.dictionary import Dictionary
import transformers
from tensorflow.io import TFRecordWriter
from tensorflow.train import Int64List
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from wikipedia2vec.dump_db import DumpDB

#from luke.pretraining.tokenization import tokenize, tokenize_segments
from luke.utils.entity_vocab import UNK_TOKEN, EntityVocab
from luke.utils.model_utils import (
    ENTITY_VOCAB_FILE,
    METADATA_FILE,
    get_entity_vocab_file_path,
)
from luke.utils.sentence_splitter import SentenceSplitter
from luke.utils.wiki_entity_linker import JsonWikiEntityLinker, WikiEntityLinker

import re
from typing import List

#from transformers import PreTrainedTokenizer, RobertaTokenizer, XLMRobertaTokenizer


logger = logging.getLogger(__name__)

DATASET_FILE = "dataset.tf"

# global variables used in pool workers
_dump_db = _tokenizer = _sentence_splitter = _entity_vocab = _max_num_tokens = _max_entity_length = None
_max_mention_length = _min_sentence_length = _include_sentences_without_entities = _include_unk_entities = None
_abstract_only = _language = None


@click.command()
@click.argument("dump_db_file", type=click.Path(exists=True))
@click.argument("model_path")
@click.argument("entity_vocab_file", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--examples-per-file", default=1000)
@click.option("--sentence-splitter", default="en")
@click.option("--max-seq-length", default=512)
@click.option("--max-entity-length", default=128)
@click.option("--max-mention-length", default=30)
@click.option("--min-sentence-length", default=5)
@click.option("--abstract-only", is_flag=True)
@click.option("--include-sentences-without-entities", is_flag=True)
@click.option("--include-unk-entities/--skip-unk-entities", default=False)
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", default=100)
@click.option("--max-num-documents", default=None, type=int)
@click.option("--predefined-entities-only", is_flag=True)
@click.option("--mention_candidate")
@click.option("--use-entity-linker", is_flag=True)
def build_wikipedia_pretraining_dataset_for_meae(
        dump_db_file: str, model_path: str, entity_vocab_file: str, output_dir: str, sentence_splitter: str, 
        examples_per_file: int, mention_candidate: str, **kwargs
):
    dump_db = DumpDB(dump_db_file)
    tokenizer_path = os.path.join(model_path, "sentencepiece.bpe.model")
    dict_path = os.path.join(model_path, "dict.txt")
    #tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
    dictionary = Dictionary.load(dict_path)
    sentence_splitter = SentenceSplitter.from_name(sentence_splitter)

    entity_vocab = EntityVocab(entity_vocab_file)

    mention_candidate_json_file_paths = json.load(open(mention_candidate, "r"))
    entity_linker = JsonWikiEntityLinker(
        tokenizer, 
        mention_candidate_json_file_paths,
        entity_vocab,
        max_mention_length=10,
        language=dump_db.language,
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    WikipediaPretrainingDataset.build(
        dump_db, 
        tokenizer, 
        dictionary, 
        sentence_splitter, 
        entity_vocab, 
        output_dir, 
        examples_per_file=examples_per_file, 
        entity_linker=entity_linker, 
        **kwargs
    )


class WikipediaPretrainingDataset:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        with open(os.path.join(dataset_dir, METADATA_FILE)) as metadata_file:
            self.metadata = json.load(metadata_file)

    def __len__(self):
        return self.metadata["number_of_items"]

    @property
    def max_seq_length(self):
        return self.metadata["max_seq_length"]

    @property
    def max_entity_length(self):
        return self.metadata["max_entity_length"]

    @property
    def max_mention_length(self):
        return self.metadata["max_mention_length"]

    @property
    def language(self):
        return self.metadata.get("language", None)

    @property
    def tokenizer(self):
        tokenizer_class_name = self.metadata.get("tokenizer_class", "")
        tokenizer_class = getattr(transformers, tokenizer_class_name)
        return tokenizer_class.from_pretrained(self.dataset_dir)

    @property
    def entity_vocab(self):
        vocab_file_path = get_entity_vocab_file_path(self.dataset_dir)
        return EntityVocab(vocab_file_path)

    def create_iterator(
        self,
        skip: int = 0,
        num_workers: int = 1,
        worker_index: int = 0,
        shuffle_buffer_size: int = 1000,
        shuffle_seed: int = 0,
        num_parallel_reads: int = 10,
        repeat: bool = True,
        shuffle: bool = True,
    ):

        # The TensorFlow 2.0 has enabled eager execution by default.
        # At the starting of algorithm, we need to use this to disable eager execution.
        tf.compat.v1.disable_eager_execution()

        features = dict(
            word_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_position_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            page_id=tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),
        )
        dataset = tf.data.TFRecordDataset(
            [os.path.join(self.dataset_dir, DATASET_FILE)],
            compression_type="GZIP",
            num_parallel_reads=num_parallel_reads,
        )
        if repeat:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
        dataset = dataset.skip(skip)
        dataset = dataset.shard(num_workers, worker_index)
        dataset = dataset.map(functools.partial(tf.io.parse_single_example, features=features))
        it = tf.compat.v1.data.make_one_shot_iterator(dataset)
        it = it.get_next()

        with tf.compat.v1.Session() as sess:
            try:
                while True:
                    obj = sess.run(it)
                    yield dict(
                        page_id=obj["page_id"][0],
                        word_ids=obj["word_ids"],
                        entity_ids=obj["entity_ids"],
                        entity_position_ids=obj["entity_position_ids"].reshape(-1, self.metadata["max_mention_length"]),
                    )
            except tf.errors.OutOfRangeError:
                pass

    @classmethod
    def build(
        cls,
        dump_db: DumpDB,
        tokenizer: SentencePieceProcessor,
        dictionary: Dictionary,
        sentence_splitter: SentenceSplitter,
        entity_vocab: EntityVocab,
        output_dir: str,
        examples_per_file: int,
        entity_linker: WikiEntityLinker,
        max_seq_length: int,
        max_entity_length: int,
        max_mention_length: int,
        min_sentence_length: int,
        abstract_only: bool,
        include_sentences_without_entities: bool,
        include_unk_entities: bool,
        pool_size: int,
        chunk_size: int,
        max_num_documents: Optional[int],
        predefined_entities_only: bool,
        use_entity_linker: bool,
    ):

        target_titles = [
            title
            for title in dump_db.titles()
            if not (":" in title and title.lower().split(":")[0] in ("image", "file", "category"))
        ]

        if predefined_entities_only:
            lang = dump_db.language  # None <- entity_vocab の parse に合わせる
            target_titles = [title for title in target_titles if entity_vocab.contains(title, lang)]

        random.shuffle(target_titles)

        if max_num_documents is not None:
            target_titles = target_titles[:max_num_documents]

        max_num_tokens = max_seq_length - 2  # 2 for [CLS] and [SEP]

        #tokenizer.save_pretrained(output_dir)

        entity_vocab.save(os.path.join(output_dir, ENTITY_VOCAB_FILE))
        number_of_items = 0
        num_total_entity = 0
        num_ignored = 0
        len_dist = {32:0, 64:0, 128:0, 256:0, 512:0}

        options = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.GZIP)
        #file_id = 0
        #tf_file = os.path.join(output_dir, "dataset-"+str(file_id)+".tf")
        tf_file = os.path.join(output_dir, DATASET_FILE)
        #writer = TFRecordWriter(tf_file, options=options)
        
        with TFRecordWriter(tf_file, options=options) as writer:
            with tqdm(total=len(target_titles)) as pbar:
                initargs = (
                    dump_db,
                    tokenizer,
                    dictionary,
                    sentence_splitter,
                    entity_vocab,
                    max_num_tokens,
                    max_entity_length,
                    max_mention_length,
                    min_sentence_length,
                    abstract_only,
                    include_sentences_without_entities,
                    include_unk_entities,
                    entity_linker,
                    use_entity_linker,
                )
                with closing(
                    Pool(pool_size, initializer=WikipediaPretrainingDataset._initialize_worker, initargs=initargs)
                ) as pool:
                    for item in pool.imap_unordered(
                        WikipediaPretrainingDataset._process_page, target_titles, chunksize=chunk_size
                    ):
                        ret, n_total_entity, n_ignored, seq_len_dist = item
                        for data in ret:
                            #data, n_collected_entity, n_ignored, seq_len = item
                            writer.write(data)
                            number_of_items += 1
                        
                        num_total_entity += n_total_entity
                        num_ignored += n_ignored
                        for max_len in [32, 64, 128, 256, 512]:
                            len_dist[max_len] += seq_len_dist[max_len]
                        pbar.update()
                
                len_dist_str = ", ".join([str(max_len)+":"+str(len_dist[max_len]) for max_len in len_dist])
                logger.info("Total/Ignored entities = {}/{}".format(num_total_entity, num_ignored))
                logger.info("Example length distribution: {}".format(len_dist_str))



        with open(os.path.join(output_dir, METADATA_FILE), "w") as metadata_file:
            json.dump(
                dict(
                    number_of_items=number_of_items,
                    max_seq_length=max_seq_length,
                    max_entity_length=max_entity_length,
                    max_mention_length=max_mention_length,
                    min_sentence_length=min_sentence_length,
                    tokenizer_class=tokenizer.__class__.__name__,
                    language=dump_db.language,
                ),
                metadata_file,
                indent=2,
            )

    @staticmethod
    def _initialize_worker(
        dump_db: DumpDB,
        tokenizer: SentencePieceProcessor,
        dictionary: Dictionary,
        sentence_splitter: SentenceSplitter,
        entity_vocab: EntityVocab,
        max_num_tokens: int,
        max_entity_length: int,
        max_mention_length: int,
        min_sentence_length: int,
        abstract_only: bool,
        include_sentences_without_entities: bool,
        include_unk_entities: bool,
        entity_linker: WikiEntityLinker,
        use_entity_linker: bool,
    ):
        global _dump_db, _tokenizer, _dictionary, _sentence_splitter, _entity_vocab, _max_num_tokens, _max_entity_length
        global _max_mention_length, _min_sentence_length, _include_sentences_without_entities, _include_unk_entities
        global _abstract_only
        global _language
        global _entity_linker, _use_entity_linker

        _dump_db = dump_db
        _tokenizer = tokenizer
        _dictionary = dictionary
        _sentence_splitter = sentence_splitter
        _entity_vocab = entity_vocab
        _max_num_tokens = max_num_tokens
        _max_entity_length = max_entity_length
        _max_mention_length = max_mention_length
        _min_sentence_length = min_sentence_length
        _include_sentences_without_entities = include_sentences_without_entities
        _include_unk_entities = include_unk_entities
        _abstract_only = abstract_only
        _language = dump_db.language
        _entity_linker = entity_linker
        _use_entity_linker = use_entity_linker

    @staticmethod
    def _process_page(page_title: str):
        if _entity_vocab.contains(page_title, _language):
            page_id = _entity_vocab.get_id(page_title, _language)
        else:
            page_id = -1

        sentences = []

        for paragraph in _dump_db.get_paragraphs(page_title):

            if _abstract_only and not paragraph.abstract:
                continue

            paragraph_text = paragraph.text
            paragraph_text = paragraph_text.encode("utf-8", "ignore").decode("utf-8")

            # First, get paragraph links.
            # Parapraph links are represented its form (link_title) and the start/end positions of strings
            # (link_start, link_end).
            paragraph_links = []
            for link in paragraph.wiki_links:
                try:
                    link_title = _dump_db.resolve_redirect(link.title)
                except:
                    logger.info("Failed to resolve title: {}, {}".format(link.title, repr(link.title)))
                # remove category links
                if link_title.startswith("Category:") and link.text.lower().startswith("category:"):
                    paragraph_text = (
                        paragraph_text[: link.start] + " " * (link.end - link.start) + paragraph_text[link.end :]
                    )
                else:
                    if _entity_vocab.contains(link_title, _language):
                        paragraph_links.append((link_title, link.start, link.end))
                    elif _include_unk_entities:
                        paragraph_links.append((UNK_TOKEN, link.start, link.end))
            
            if _use_entity_linker:
                candidate_links = _entity_linker.link_entities_in_text(paragraph_text, _language, page_title, _language)
                paragraph_links = merge_links(paragraph_links, candidate_links, paragraph_text)
            
            sent_spans = _sentence_splitter.get_sentence_spans(paragraph_text.rstrip())
            for sent_start, sent_end in sent_spans:
                cur = sent_start
                sent_words = []
                sent_links = []
                # Look for links that are within the tokenized sentence.
                # If a link is found, we separate the sentences across the link and tokenize them.
                for link_title, link_start, link_end in paragraph_links:
                    if not (sent_start <= link_start < sent_end and link_end <= sent_end):
                        continue
                    entity_id = _entity_vocab.get_id(link_title, _language)

                    sent_tokenized, link_words = tokenize_segments(
                        [paragraph_text[cur:link_start], paragraph_text[link_start:link_end]],
                        tokenizer=_tokenizer,
                        add_prefix_space=cur == 0 or paragraph_text[cur - 1] == " ",
                    )

                    sent_words += sent_tokenized

                    sent_links.append((entity_id, len(sent_words), len(sent_words) + len(link_words)))
                    sent_words += link_words
                    cur = link_end

                sent_words += tokenize(
                    text=paragraph_text[cur:sent_end],
                    tokenizer=_tokenizer,
                    add_prefix_space=cur == 0 or paragraph_text[cur - 1] == " ",
                )

                if len(sent_words) < _min_sentence_length or len(sent_words) > _max_num_tokens:
                    continue
                sentences.append((sent_words, sent_links))

        ret = []
        words = []
        links = []
        n_total_entity = 0
        n_ignored = 0
        seq_lens = []
        len_dist = {32:0, 64:0, 128:0, 256:0, 512:0}
        for i, (sent_words, sent_links) in enumerate(sentences):
            links += [(id_, start + len(words), end + len(words)) for id_, start, end in sent_links]
            words += sent_words
            if i == len(sentences) - 1 or len(words) + len(sentences[i + 1][0]) > _max_num_tokens:
                if links or _include_sentences_without_entities:
                    links = links[:_max_entity_length]
                    #word_ids = _tokenizer.convert_tokens_to_ids(words)
                    word_ids = word_to_idx(_dictionary, words)
                    assert _min_sentence_length <= len(word_ids) <= _max_num_tokens
                    entity_ids = [id_ for id_, _, _, in links]
                    assert len(entity_ids) <= _max_entity_length
                    entity_position_ids = itertools.chain(
                        *[
                            (list(range(start, end)) + [-1] * (_max_mention_length - end + start))[:_max_mention_length]
                            for _, start, end in links
                        ]
                    )
                    entity_labels = np.zeros_like(word_ids)
                    mention_boundaries = np.zeros_like(word_ids)
                    for id_, start, end in links:
                        n_total_entity += 1
                        if start == end:
                            n_ignored += 1
                            continue
                        entity_labels[start] = id_
                        mention_boundaries[start] = 1
                        for tok_id in range(start+1, end):
                            mention_boundaries[tok_id] = 2
                    
                    if False:
                        print ("words:\n", words)
                        print ("word_ids:\n", word_ids)
                        print ("entity_labels:\n", entity_labels)
                        print ("mention_boundaries:\n", mention_boundaries)
                        print ("entity_ids:\n", entity_ids)
                        print ("links:\n", links)
                        print ("entity_position_ids:\n", list(entity_position_ids))
                        for id in entity_ids:
                            print ("entity_id={}, title={}".format(id, _entity_vocab.get_title_by_id(id, _language))) 
                        exit()

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature=dict(
                                page_id=tf.train.Feature(int64_list=tf.train.Int64List(value=[page_id])),
                                word_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=word_ids)),
                                entity_labels=tf.train.Feature(int64_list=tf.train.Int64List(value=entity_labels)),
                                mention_boundaries=tf.train.Feature(int64_list=tf.train.Int64List(value=mention_boundaries)),
                                #entity_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=entity_ids)),
                                #entity_position_ids=tf.train.Feature(int64_list=Int64List(value=entity_position_ids)),
                            )
                        )
                    )
                    ret.append((example.SerializeToString()))
                    seq_len = len(word_ids)
                    for max_len in [32,64,128,256,512]:
                        if seq_len < max_len:
                            len_dist[max_len] += 1
                            break
                    #seq_lens.append(len(word_ids))

                words = []
                links = []
        return (ret, n_total_entity, n_ignored, len_dist)


def merge_links(paragraph_links, candidate_links, paragraph_text):
    link_starts = {}
    links = []
    print ("Text:\n{}\nLinks:".format(paragraph_text))
    for entity, start, end in paragraph_links:
        link_starts[start] = (entity, start, end)
        links.append((entity, start, end))
        print ("{}:{}-{}({})".format(entity, start, end, paragraph_text[start:end]))
    print ("Cand Links:")
    for entity, start, end in candidate_links:
        if start not in link_starts:
            links.append((entity, start, end))
            print ("{}:{}-{}({})".format(entity, start, end, paragraph_text[start:end]))
        else:
            print ("{}:{}-{}({}) ignored".format(entity, start, end, paragraph_text[start:end]))
    return links

# this is a chinese character that will produce ['_', '龘', ...] after tokenized
# when added before the original text
# mainly works for languages like Chinese to remove the '_' for in-text words
# since roberta treats words at beginning and in-text words differently
# the first word is automatically treated as beginning word
# but here we have to tokenize in-text mention word for entity separately from 
# normal text. And this helps to change them back to in-text words
XLM_ROBERTA_UNK_CHAR = "龘"


def tokenize(text: str, tokenizer: SentencePieceProcessor, add_prefix_space: bool):
    text = re.sub(r"\s+", " ", text).rstrip()
    add_prefix_space = text.startswith(" ") or add_prefix_space
    if not text:
        return []
    try:
        if add_prefix_space:
            return tokenizer.encode(text, out_type=str)
        else:
            return tokenizer.encode(XLM_ROBERTA_UNK_CHAR + text, out_type=str)[2:]
    except TypeError:
        print ("text:\n", repr(text))#text.encode("utf-8").decode("utf-8"))
        logger.info("Error occured during tokenization. Skip.")
        return []


def tokenize_segments(
    segments: List[str], tokenizer: SentencePieceProcessor, add_prefix_space: bool = True
) -> List[List[str]]:
    tokenized_segments = []
    for i, text in enumerate(segments):
        if i == 0:
            tokenized_segments.append(tokenize(text, tokenizer, add_prefix_space=add_prefix_space))
        else:
            prev_text = segments[i - 1]
            tokenized_segments.append(
                tokenize(text, tokenizer, add_prefix_space=prev_text.endswith(" ") or len(prev_text) == 0)
            )

    return tokenized_segments

def word_to_idx(dictionary, words):
    ids = []
    for i, word in enumerate(words):
        idx = dictionary.index(word)
        ids.append(idx)
    return ids
