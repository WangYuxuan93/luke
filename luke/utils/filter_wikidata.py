import datetime
import glob
import json
import logging
import math
import os
import time

import bz2
import ujson

from argparse import Namespace
from typing import List, Tuple

import click
import numpy as np

from luke.utils.entity_vocab import EntityVocab
from luke.utils.aligned_entity_vocab import AlignedEntityVocab

logger = logging.getLogger(__name__)

@click.command()
@click.option("--wiki-data-file", type=click.Path(), required=True)
@click.option("--entity-vocab-path", type=click.Path(), required=True)
@click.option("--out-file", type=click.Path(), required=True)
@click.option("--filter-by", default="sitelinks")
def filter_wikidata_with_entity_vocab(wiki_data_file, entity_vocab_path, out_file, filter_by):
    lcs = ["en", "ar","bn","de","nl","el","es","fi","fr","hi","id","it","ja","ko","pl","pt","ru","sv","sw","te","th","tr","vi","zh"]
    logger.info("Loading entity vocab from {}".format(entity_vocab_path))
    entity_vocab = EntityVocab(entity_vocab_path)
    assert filter_by in ("labels", "sitelinks")
    logger.info("Filtering by {}".format(filter_by))
    #out = entity_vocab.contains("United States", "en")
    #print (out)
    num_remained = 0
    with bz2.BZ2File(wiki_data_file) as f, open(out_file, "w") as fo:
        for (n, line) in enumerate(f):
            if n % 1000 == 0 and n != 0:
                logger.info("Processed %d lines / %d collected", n, num_remained)
            
            line = line.rstrip().decode("utf-8")
            if line in ("[", "]"):
                continue
            
            if line[-1] == ",":
                line = line[:-1]
            obj = ujson.loads(line)
            if obj["type"] != "item":
                continue

            if filter_by == "labels":
                labels = obj["labels"]
                for lc in lcs:
                    if lc in labels:
                        if entity_vocab.contains(labels[lc]["value"], lc):
                            json.dump(obj, fo)
                            fo.write("\n")
                            num_remained += 1
                            break
            elif filter_by == "sitelinks":
                sitelinks = obj["sitelinks"]
                for lc in lcs:
                    site = lc+"wiki"
                    if site in sitelinks:
                        if entity_vocab.contains(sitelinks[site]["title"], lc):
                            json.dump(obj, fo)
                            fo.write("\n")
                            num_remained += 1
                            break
            #if n >= 10000:
            #    break

            #print (obj["labels"])
            #exit()
    print ("Remained lines = {}".format(str(num_remained)))


@click.command()
@click.option("--wiki-data-file", type=click.Path(), required=True)
@click.option("--entity-vocab-path", type=click.Path(), required=True)
@click.option("--out-file", type=click.Path(), required=True)
def align_entity_vocab_with_wikidata_id(wiki_data_file, entity_vocab_path, out_file):
    lcs = ["en", "ar","bn","de","nl","el","es","fi","fr","hi","id","it","ja","ko","pl","pt","ru","sv","sw","te","th","tr","vi","zh"]
    logger.info("Loading entity vocab from {}".format(entity_vocab_path))
    entity_vocab = AlignedEntityVocab(entity_vocab_path, init_from_entity_vocab=True)
    entity_vocab._align(wiki_data_file)
    entity_vocab.save(out_file)    
