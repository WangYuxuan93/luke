dir=/workspace/wiki
wikidata=$dir/wikidata/wikidata-20220117-all.json.bz2
entity=$dir/entity_vocab/mluke_entity_vocab.jsonl
out=multi_vocab_aligned.jsonl
out2=multi2.jsonl

python luke/cli.py align-entity-vocab-with-wikidata-id --wiki-data-file=$wikidata --entity-vocab-path=$out --out-file=$out2
