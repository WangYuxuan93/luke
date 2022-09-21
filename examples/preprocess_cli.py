import sys
sys.path.append('.')

import logging
import multiprocessing
import os
import random

import click
import numpy as np
import torch
from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader


import examples.utils.wiki_entity_linker.mention_db
import examples.utils.wiki_entity_linker.wiki_link_db


@click.group()
@click.option("--verbose", is_flag=True)
@click.option("--seed", type=int, default=None)
def cli(verbose: bool, seed: int):
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
        logging.getLogger("transformers").setLevel(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)
        logging.getLogger("transformers").setLevel(level=logging.WARNING)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


cli.add_command(examples.utils.wiki_entity_linker.mention_db.build_from_wikipedia)
cli.add_command(examples.utils.wiki_entity_linker.wiki_link_db.build_wiki_link_db)

if __name__ == "__main__":
    cli()
