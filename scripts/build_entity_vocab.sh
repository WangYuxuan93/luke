#l=sw
lcs=(ar bn de nl el en es fi fr hi id it ja ko pl pt ru sv sw te th tr vi zh)
for ((i=$1;i<=$2;i++))
do
  l=${lcs[$i]}
  dir=/workspace/wiki
  db="$dir/wikidb/${l}wiki.db"
  vocab="$dir/entity_vocab/mluke_entity_vocab_${l}.jsonl"
  echo python luke/cli.py build-entity-vocab $db $vocab
  python luke/cli.py build-entity-vocab $db $vocab
done
