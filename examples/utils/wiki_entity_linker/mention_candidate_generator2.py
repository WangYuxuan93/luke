from typing import Dict

import joblib
from allennlp.common import FromParams

from wikipedia2vec.dump_db import DumpDB
from .wiki_link_db import WikiLinkDB


def normalize_mention(text: str) -> str:
    return " ".join(text.split(" ")).strip()


class MentionCandidatesGenerator(FromParams):
    """
    Detect entity mentions in text from Wikipedia articles.
    """

    def __init__(
        self,
        wiki_link_db_path: str,
        #link_redirect_mappings_path: str,
        #model_redirect_mappings_path: str,
        dump_db: DumpDB,
        min_mention_link_prob: float = 0.01,
        max_mention_length: int = 10,
    ):
        self.wiki_link_db = WikiLinkDB(wiki_link_db_path)
        self.dump_db = dump_db
        #self.link_redirect_mappings: Dict[str, str] = joblib.load(link_redirect_mappings_path)
        #self.model_redirect_mappings: Dict[str, str] = joblib.load(model_redirect_mappings_path)

        self.min_mention_link_prob = min_mention_link_prob

        self.max_mention_length = max_mention_length

    def get_mention_candidates(self, title: str) -> Dict[str, str]:
        """
        Returns a dict of [mention, entity (title)]
        """

        if "_" in title:
            import warnings

            warnings.warn(f"The title ``{title}`` contains under-bars. However, WikiLinkDB expects white-spaced title.")

        # the input title is directly from dump_db.titles(),
        # which is exactly the keys of wiki_link_db

        #title = self.link_redirect_mappings.get(title, title)
        #try:
        #    title = self.dump_db.resolve_redirect(title)
        #except:
            #print ("mention_candidate_generator2.py: Failed in resolve redirect.")

        # mention_to_entity
        mention_candidates = dict()
        ambiguous_mentions = set()
        for link in self.wiki_link_db.get(title):
            if link.link_prob < self.min_mention_link_prob:
                continue
            link_text = normalize_mention(link.text)
            #link_title = self.model_redirect_mappings.get(link.title, link.title)
            #try:
            #    link_title = self.dump_db.resolve_redirect(link.title)
            #except:
            
            # here the link.title is from wiki_link_db which has been resolved
            # therefore it's the same as titles in entity_vocab

            link_title = link.title
            if link_text in mention_candidates:
                ambiguous_mentions.add(link_text)
                del mention_candidates[link_text]

            if not link_text in ambiguous_mentions:
                mention_candidates[link_text] = link_title
        return mention_candidates
