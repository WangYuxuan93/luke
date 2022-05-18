dir=/workspace/wiki
entityvocab=$dir/entity_vocab/mluke_entity_vocab.jsonl
#entityvocab=$dir/entity_vocab/small_entity_vocab.jsonl
#modelname=xlm-roberta-base
modelname=/workspace/ptm/xlm-roberta-base

lcs=(ar bn de nl el en es fi fr hi id it ja ko pl pt ru sv sw te th tr vi zh)
for ((i=$1;i<=$2;i++))
do
  l=${lcs[$i]}
  command="python luke/cli.py build-wikipedia-pretraining-dataset-for-meae $dir/wikidb/${l}wiki.db ${modelname} ${entityvocab} $dir/dataset/${l} --sentence-splitter=${l}" #--pool-size 1"
  echo $command
  $command
done

