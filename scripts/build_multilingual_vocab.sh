dir=/workspace/wiki
interdb=$dir/wikidb/interwiki.db
out=$dir/entity_vocab/mluke_entity_vocab.jsonl
COMMAND="python luke/cli.py build-multilingual-entity-vocab -i $interdb -o $out --vocab-size 1200000 --min-num-languages 3 "
# add options by for loop because there are so many..
for l in ar bn de nl el en es fi fr hi id it ja ko pl pt ru sv sw te th tr vi zh
do
COMMAND=$COMMAND" -v $dir/entity_vocab/mluke_entity_vocab_${l}.jsonl"
done
#COMMAND=${COMMAND%?}
echo ${COMMAND}
eval $COMMAND
