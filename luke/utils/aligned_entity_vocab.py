import bz2
import ujson
from functools import partial
import json
import logging
import math
import multiprocessing
from collections import Counter, OrderedDict, defaultdict, namedtuple
from contextlib import closing
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Dict, List, TextIO

import click
from tqdm import tqdm
from wikipedia2vec.dump_db import DumpDB

from luke.utils.entity_vocab import EntityVocab

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"

SPECIAL_TOKENS = {PAD_TOKEN, UNK_TOKEN, MASK_TOKEN}

Entity = namedtuple("Entity", ["title", "language"])

_dump_db = None  # global variable used in multiprocessing workers

logger = logging.getLogger(__name__)

class AlignedEntityVocab(EntityVocab):
    def __init__(self, vocab_file: str, init_from_entity_vocab=True):
        self.vocab: Dict[Entity, int] = {}
        self.counter: Dict[Entity, int] = {}
        self.inv_vocab: Dict[int, List[Entity]] = defaultdict(list)
                
        self.id2qid: Dict[int, str] = {}
        self.qid2id: Dict[str, int] = {}

        if init_from_entity_vocab:
            super().__init__(vocab_file)
        else:
            self._parse_aligned_jsonl_vocab_file(vocab_file)
        
            self.special_token_ids = {}
            for special_token in SPECIAL_TOKENS:
                special_token_entity = self.search_across_languages(special_token)[0]
                self.special_token_ids[special_token] = self.get_id(*special_token_entity)
            for id_ in self.special_token_ids.values():
                self.id2qid[id_] = None
        
        
    def _align(self, wiki_data_file, lcs = ["en", "ar","bn","de","nl","el","es","fi","fr","hi","id","it","ja","ko","pl","pt","ru","sv","sw","te","th","tr","vi","zh"]):

        num_entity = self.__len__()
        num_covered = 0
        if wiki_data_file.endswith("bz2"):
            loader = bz2.BZ2File
        elif wiki_data_file.endswith("jsonl"):
            loader = partial(open, mode="r")
            #f = bz2.BZ2File(wiki_data_file)

        #with bz2.BZ2File(wiki_data_file) as f:
        with loader(wiki_data_file) as f:
            for (n, line) in enumerate(f):
                if n % 1000 == 0 and n != 0:
                    logger.info("Processed %d lines, Total/Covered entities: %d/%d", n, num_entity, num_covered)
                if wiki_data_file.endswith("bz2"):
                    line = line.rstrip().decode("utf-8")
                elif wiki_data_file.endswith("jsonl"):
                    line = line.rstrip()
                if line in ("[", "]"):
                    continue

                if line[-1] == ",":
                    line = line[:-1]
                obj = ujson.loads(line)
                if obj["type"] != "item":
                    continue

                sitelinks = obj["sitelinks"]
                qid = obj["id"]
                for lc in lcs:
                    site = lc+"wiki"
                    if site in sitelinks:
                        if self.contains(sitelinks[site]["title"], lc):
                            id_ = self.get_id(sitelinks[site]["title"], lc)
                            if self._check(id_, sitelinks):
                                self.id2qid[id_] = qid
                                self.qid2id[qid] = id_
                                num_covered += 1
                                break
                #if n >= 10000:
                #    break
        #print (self.id2qid)
        #print (len(self.id2qid))
        logger.info("Total/Covered entities: %d/%d", num_entity, num_covered)
                            
    # check if all the title in entity vocab matches title in sitelinks
    def _check(self, id_, sitelinks):
        entities = self.get_entities_by_id(id_)
        all_match = True
        for entity in entities:
            site = entity.language+"wiki"
            if site not in sitelinks:
                all_match = False
                logger.info("Not Found! Title {} Site {} not in sitelinks".format(entity.title, site))
                break
            if not sitelinks[site]["title"] == entity.title:
                all_match = False
                logger.info("Mismatch, Entity Vocab: {}, Sitelink: {}".format(entity.title, sitelinks[site]["title"]))
                break 
        
        return all_match

    def _from_pretrained_mluke(self, transformer_model_name: str):
        from transformers.models.mluke.tokenization_mluke import MLukeTokenizer

        mluke_tokenizer = MLukeTokenizer.from_pretrained(transformer_model_name)
        title_to_idx = mluke_tokenizer.entity_vocab
        mluke_special_tokens = SPECIAL_TOKENS | {"[MASK2]"}
        for title, idx in title_to_idx.items():
            if title in mluke_special_tokens:
                entity = Entity(title, None)
            else:
                language, title = title.split(":", maxsplit=1)
                entity = Entity(title, language)
            self.vocab[entity] = idx
            self.counter[entity] = None
            self.inv_vocab[idx].append(entity)

    def _from_pretrained_luke(self, transformer_model_name: str):
        from transformers.models.luke.tokenization_luke import LukeTokenizer

        luke_tokenizer = LukeTokenizer.from_pretrained(transformer_model_name)
        title_to_idx = luke_tokenizer.entity_vocab
        for title, idx in title_to_idx.items():
            entity = Entity(title, None)
            self.vocab[entity] = idx
            self.counter[entity] = None
            self.inv_vocab[idx].append(entity)

    def _parse_tsv_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            for (index, line) in enumerate(f):
                title, count = line.rstrip().split("\t")
                entity = Entity(title, None)
                self.vocab[entity] = index
                self.counter[entity] = int(count)
                self.inv_vocab[index] = [entity]

    def _parse_aligned_jsonl_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            entities_json = [json.loads(line) for line in f]

        for item in entities_json:
            for title, language in item["entities"]:
                entity = Entity(title, language)
                self.vocab[entity] = item["id"]
                self.counter[entity] = item["count"]
                self.inv_vocab[item["id"]].append(entity)
                self.id2qid[item["id"]] = item["qid"]
                self.qid2id[item["qid"]] = item["id"]

    @property
    def size(self) -> int:
        return len(self)

    def __reduce__(self):
        return (self.__class__, (self._vocab_file,))

    def __len__(self):
        return len(self.inv_vocab)

    def __contains__(self, item: str):
        return self.contains(item, language=None)

    def __getitem__(self, key: str):
        return self.get_id(key, language=None)

    def __iter__(self):
        return iter(self.vocab)

    def contains(self, title: str, language: str = None):
        return Entity(title, language) in self.vocab

    def get_id(self, title: str, language: str = None, default: int = None) -> int:
        try:
            return self.vocab[Entity(title, language)]
        except KeyError:
            return default

    def get_title_by_id(self, id_: int, language: str = None) -> str:
        for entity in self.inv_vocab[id_]:
            if entity.language == language:
                return entity.title

    def get_count_by_title(self, title: str, language: str = None) -> int:
        entity = Entity(title, language)
        return self.counter.get(entity, 0)

    def get_entities_by_id(self, id_: int) -> List[Entity]:
        return self.inv_vocab[id_]

    def get_qid_by_id(self, id_: int) -> int:
        return self.id2qid[id_] if id_ in self.id2qid else None

    def search_across_languages(self, title: str) -> List[Entity]:
        results = []
        for entity in self.vocab.keys():
            if entity.title == title:
                results.append(entity)
        return results

    def save(self, out_file: str):

        if Path(out_file).suffix != ".jsonl":
            raise ValueError(
                "The saved file has to explicitly have the jsonl extension so that it will be loaded properly,\n"
                f"but the name provided is {out_file}."
            )

        with open(out_file, "w") as f:
            for ent_id, entities in self.inv_vocab.items():
                count = self.counter[entities[0]]
                item = {"id": ent_id, "entities": [(e.title, e.language) for e in entities], "count": count, "qid": self.get_qid_by_id(ent_id)}
                json.dump(item, f)
                f.write("\n")

    @staticmethod
    def build(
        dump_db: DumpDB,
        out_file: str,
        vocab_size: int,
        white_list: List[str],
        white_list_only: bool,
        pool_size: int,
        chunk_size: int,
        language: str,
    ):
        counter = Counter()
        with tqdm(total=dump_db.page_size(), mininterval=0.5) as pbar:
            with closing(Pool(pool_size, initializer=EntityVocab._initialize_worker, initargs=(dump_db,))) as pool:
                for ret in pool.imap_unordered(EntityVocab._count_entities, dump_db.titles(), chunksize=chunk_size):
                    counter.update(ret)
                    pbar.update()

        title_dict = OrderedDict()
        title_dict[PAD_TOKEN] = 0
        title_dict[UNK_TOKEN] = 0
        title_dict[MASK_TOKEN] = 0

        for title in white_list:
            if counter[title] != 0:
                title_dict[title] = counter[title]

        if not white_list_only:
            valid_titles = frozenset(dump_db.titles())
            for title, count in counter.most_common():
                if title in valid_titles and not title.startswith("Category:"):
                    title_dict[title] = count
                    if len(title_dict) == vocab_size:
                        break

        with open(out_file, "w") as f:
            for ent_id, (title, count) in enumerate(title_dict.items()):
                json.dump({"id": ent_id, "entities": [[title, language]], "count": count}, f, ensure_ascii=False)
                f.write("\n")

    @staticmethod
    def _initialize_worker(dump_db: DumpDB):
        global _dump_db
        _dump_db = dump_db

    @staticmethod
    def _count_entities(title: str) -> Dict[str, int]:
        counter = Counter()
        for paragraph in _dump_db.get_paragraphs(title):
            for wiki_link in paragraph.wiki_links:
                try:
                    title = _dump_db.resolve_redirect(wiki_link.title)
                    counter[title] += 1
                except:
                    logger.info(f"Failed in resolve redirect {wiki_link.title}")
        return counter



