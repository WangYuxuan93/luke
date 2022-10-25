from abc import abstractmethod, ABCMeta
from typing import Dict, List, Tuple, NamedTuple
import json
from tqdm import tqdm
import re
import logging

from sentencepiece import SentencePieceProcessor
from allennlp.common import Registrable
from allennlp.data import Token
#from transformers.tokenization_utils import PreTrainedTokenizer

from luke.utils.entity_vocab import PAD_TOKEN, Entity, EntityVocab
#from luke.pretraining.tokenization import tokenize
from transformers.models.xlm_roberta.tokenization_xlm_roberta import SPIECE_UNDERLINE

import difflib

# from transformers/tokenization_xlm_roberta.py
def convert_tokens_to_string(tokens):
    """Converts a sequence of tokens (strings for sub-words) in a single string."""
    out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ")#.strip()
    return out_string

def normalize_mention(text: str) -> str:
    return " ".join(text.split(" ")).strip()

def count_prefix_space(text):
        for i, j in enumerate(text):
            if j not in [' ','\t','\u200b','\xa0']:
                return i

class Mention(NamedTuple):
    entity: Entity
    start: int
    end: int

class WikiEntityLinker(Registrable, metaclass=ABCMeta):
    def __init__(
        self,
        tokenizer: SentencePieceProcessor,
        entity_vocab: EntityVocab,
        max_mention_length: int = 10,
    ):
        self.entity_vocab = entity_vocab
        self.max_mention_length = max_mention_length
        self.tokenizer = tokenizer

    def link_entities_in_tokens(self, tokens: List[Token], token_language: str, title: str, title_language: str) -> List[Mention]:
        mention_candidates = self.get_mention_candidates(title, title_language, token_language)
        return self._link_entities([t.text for t in tokens], mention_candidates, language=token_language)
    
    def link_entities_in_text(self, text: str, token_language: str, title: str, title_language: str, debug = False) -> List[Mention]:
        mention_candidates = self.get_mention_candidates(title, title_language, token_language)
        tokens = tokenize(text, self.tokenizer, add_prefix_space=False)
        mentions = self._link_entities(tokens, mention_candidates, language=token_language)
        links = []
        if debug:
            print ("@@@@@@@@@@@\nText:\n[{}]\nTokens:\n{}\nLinks:\n".format(text, tokens))
        if convert_tokens_to_string(tokens) == text:
            #print ("###Regular")
            for entity, start, end in mentions:
                text_start = len(convert_tokens_to_string(tokens[:start]))
                if tokens[start].startswith(SPIECE_UNDERLINE):
                    text_start += 1
                text_end = len(convert_tokens_to_string(tokens[:end]))
                links.append((entity.title, text_start, text_end))
                if False:
                    print ("\ntoken links| {}:{}-{} ({})".format(entity, start, end, tokens[start:end]))
                    print ("prefix: {}\nsuffix: {}".format(convert_tokens_to_string(tokens[:start]), convert_tokens_to_string(tokens[:end])))
                    print ("text links | {}:{}-{} ({})".format(entity, text_start, text_end, text[text_start:text_end]))
        else:
            #print ("$$$Align")
            token_id_to_text_id = self.get_token_id_to_text_id(tokens, text)
            if token_id_to_text_id is None:
                return []
            for entity, start, end in mentions:
                text_start = token_id_to_text_id[start]
                if tokens[start].startswith(SPIECE_UNDERLINE):
                    text_start += 1
                text_end = token_id_to_text_id[end-1] + len(tokens[end-1].replace(SPIECE_UNDERLINE, " "))
                links.append((entity.title, text_start, text_end))
                if debug:
                    print ("\ntoken links| {}:{}-{} ({})".format(entity, start, end, tokens[start:end]))
                    print ("prefix: {}\nsuffix: {}".format(convert_tokens_to_string(tokens[:start]), convert_tokens_to_string(tokens[:end])))
                    print ("text links | {}:{}-{} ({})".format(entity, text_start, text_end, text[text_start:text_end]))
        
        return links
    
    def get_token_id_to_text_id(self, tokens, text, debug=False):
        str = ""
        token_id_to_text_id = {}
        for token_id, token in enumerate(tokens):
            if len(str) >= len(text):
                return None
            num_space_prefix = count_prefix_space(text[len(str):])
            try:
                if num_space_prefix >= 2:
                    #print ("$$$$$$$$$$\nnum_space_prefix >= 2:\nlen={}\nstr :[{}]".format(len(str), str))
                    token_id_to_text_id[token_id] = len(str) + (num_space_prefix - 1)
                    str += ' ' * (num_space_prefix-1) + token.replace(SPIECE_UNDERLINE, " ")
                else:
                    token_id_to_text_id[token_id] = len(str)
                    str += token.replace(SPIECE_UNDERLINE, " ")
            except:
                if debug:
                    print ("!!!!!!!\nFailed \ntext:[{}]\nstr :[{}]".format(text, str))
                return None
        try:
            assert str == text.rstrip() or len(str) == len(text.rstrip())
        except:
            if debug:
                print ("*********\nMismatch:\nText({}):[{}]\nCcat({}):[{}]\nToks:{}".format(len(text), text, len(str), str, tokens))
                print ("Last token: Text:'{}', Ccat:'{}'".format(text[-1], str[-1]))
                d = difflib.Differ()
                diff = d.compare([text], [str])
                print ("\n".join(list(diff)))
            return None
        return token_id_to_text_id

    @abstractmethod
    def get_mention_candidates(self, title: str, title_language: str, token_language: str) -> Dict[str, str]:
        raise NotImplementedError()

    def _link_entities(self, tokens: List[str], mention_candidates: Dict[str, str], language: str) -> List[Mention]:
        mentions = []
        cur = 0
        for start, token in enumerate(tokens):
            if start < cur:
                continue

            for end in range(min(start + self.max_mention_length, len(tokens)), start, -1):

                mention_text = convert_tokens_to_string(tokens[start:end])
                mention_text = normalize_mention(mention_text)

                title = mention_candidates.get(mention_text, None)
                if title is None:
                    continue

                cur = end
                if self.entity_vocab.contains(title, language):
                    mention = Mention(Entity(title, language), start, end)
                    mentions.append(mention)
                break

        return mentions

    def mentions_to_entity_features(self, tokens: List[Token], mentions: List[Mention]) -> Dict:

        if len(mentions) == 0:
            entity_ids = [self.entity_vocab.special_token_ids[PAD_TOKEN]]
            entity_segment_ids = [0]
            entity_attention_mask = [0]
            entity_position_ids = [[-1 for _ in range(self.max_mention_length)]]
        else:
            entity_ids = [0] * len(mentions)
            entity_segment_ids = [0] * len(mentions)
            entity_attention_mask = [1] * len(mentions)
            entity_position_ids = [[-1 for _ in range(self.max_mention_length)] for x in range(len(mentions))]

            for i, (entity, start, end) in enumerate(mentions):
                entity_ids[i] = self.entity_vocab.get_id(entity.title, entity.language)
                entity_position_ids[i][: end - start] = range(start, end)

                if tokens[start].type_id is not None:
                    entity_segment_ids[i] = tokens[start].type_id

        return {
            "entity_ids": entity_ids,
            "entity_attention_mask": entity_attention_mask,
            "entity_position_ids": entity_position_ids,
            "entity_segment_ids": entity_segment_ids,
        }


@WikiEntityLinker.register("json")
class JsonWikiEntityLinker(WikiEntityLinker):
    def __init__(
        self,
        tokenizer: SentencePieceProcessor,
        mention_candidate_json_file_paths: Dict[Tuple[str, str], str],
        entity_vocab: EntityVocab,
        max_mention_length: int = 10,
        language: str = None,
    ):
        super().__init__(tokenizer=tokenizer, entity_vocab=entity_vocab, max_mention_length=max_mention_length)
        self.mention_candidates = {}
        if language is not None:
            load_paths = {language: mention_candidate_json_file_paths[language]}
        else:
            load_paths = mention_candidate_json_file_paths
        with tqdm(total=len(load_paths.items())) as pbar:
            for (title_token_language), path in load_paths.items():
                self.mention_candidates[title_token_language] = json.load(open(path))
                pbar.update()
        #self.mention_candidates = {
        #    title_token_language: json.load(open(path))
        #    for (title_token_language), path in mention_candidate_json_file_paths.items()
        #}

    def get_mention_candidates(self, title: str, title_language: str, token_language: str) -> Dict[str, str]:
        #mention_candidates = self.mention_candidates[f"{title_language}-{token_language}"][title]
        mention_candidates = self.mention_candidates[f"{title_language}"][title]
        return mention_candidates


XLM_ROBERTA_UNK_CHAR = "龘"

logger = logging.getLogger(__name__)

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
