dir=/workspace/wiki
wikidata=$dir/wikidata/wikidata-20220117-all.json.bz2
entity=$dir/entity_vocab/mluke_entity_vocab.jsonl
out=$dir/wikidata/filtered_wikidata.jsonl

python luke/cli.py filter-wikidata-with-entity-vocab --wiki-data-file=$wikidata --entity-vocab-path=$entity --out-file=$out --filter-by sitelinks
