dir=/workspace/wiki
entityvocab=$dir/entity_vocab/mluke_entity_vocab.jsonl
modelname=xlm-roberta-base

lcs=(ar bn de nl el en es fi fr hi id it ja ko pl pt ru sv sw te th tr vi zh)
l=en
command="python luke/cli.py build-wikipedia-pretraining-dataset $dir/wikidb/${l}wiki.db ${modelname} ${entityvocab} $dir/dataset/${l} --sentence-splitter=${l}"
echo $command
$command
