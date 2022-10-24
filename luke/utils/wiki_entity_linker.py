from abc import abstractmethod, ABCMeta
from typing import Dict, List, Tuple, NamedTuple
import json
from tqdm import tqdm

from allennlp.common import Registrable
from allennlp.data import Token
from transformers.tokenization_utils import PreTrainedTokenizer

from luke.utils.entity_vocab import PAD_TOKEN, Entity, EntityVocab
#from luke.pretraining.meae_dataset import tokenize
from luke.pretraining.tokenization import tokenize

def normalize_mention(text: str) -> str:
    return " ".join(text.split(" ")).strip()

class Mention(NamedTuple):
    entity: Entity
    start: int
    end: int

class WikiEntityLinker(Registrable, metaclass=ABCMeta):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        entity_vocab: EntityVocab,
        max_mention_length: int = 10,
    ):
        self.entity_vocab = entity_vocab
        self.max_mention_length = max_mention_length
        self.tokenizer = tokenizer

    def link_entities_in_tokens(self, tokens: List[Token], token_language: str, title: str, title_language: str) -> List[Mention]:
        mention_candidates = self.get_mention_candidates(title, title_language, token_language)
        return self._link_entities([t.text for t in tokens], mention_candidates, language=token_language)
    
    def link_entities_in_text(self, text: str, token_language: str, title: str, title_language: str) -> List[Mention]:
        mention_candidates = self.get_mention_candidates(title, title_language, token_language)
        tokens = tokenize(text, self.tokenizer, add_prefix_space=False)
        mentions = self._link_entities(tokens, mention_candidates, language=token_language)
        links = []
        for entity, start, end in mentions:
            text_start = len(self.tokenizer.convert_tokens_to_string(tokens[:start]))
            text_end = len(self.tokenizer.convert_tokens_to_string(tokens[:end]))
            links.append((entity.title, text_start, text_end))
        return links

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

                mention_text = self.tokenizer.convert_tokens_to_string(tokens[start:end])
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

    #def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
    #    self.tokenizer = tokenizer

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
        tokenizer: PreTrainedTokenizer,
        mention_candidate_json_file_paths: Dict[Tuple[str, str], str],
        entity_vocab: EntityVocab,
        max_mention_length: int = 10,
    ):
        super().__init__(tokenizer=tokenizer, entity_vocab=entity_vocab, max_mention_length=max_mention_length)
        self.mention_candidates = {}
        with tqdm(total=len(mention_candidate_json_file_paths.items())) as pbar:
            for (title_token_language), path in mention_candidate_json_file_paths.items():
                self.mention_candidates[title_token_language] = json.load(open(path))
                pbar.update()
        #self.mention_candidates = {
        #    title_token_language: json.load(open(path))
        #    for (title_token_language), path in mention_candidate_json_file_paths.items()
        #}

    def get_mention_candidates(self, title: str, title_language: str, token_language: str) -> Dict[str, str]:
        mention_candidates = self.mention_candidates[f"{title_language}-{token_language}"][title]
        return mention_candidates

