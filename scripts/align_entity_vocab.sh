dir=/workspace/wiki
#wikidata=$dir/wikidata/wikidata-20220117-all.json.bz2
wikidata=$dir/wikidata/filtered_wikidata.jsonl
entity=$dir/entity_vocab/mluke_entity_vocab.jsonl
out=$dir/entity_vocab/mluke_entity_vocab_aligned.jsonl

python luke/cli.py align-entity-vocab-with-wikidata-id --wiki-data-file=$wikidata --entity-vocab-path=$entity --out-file=$out
