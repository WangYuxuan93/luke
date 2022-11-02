dir=/work/wiki
entityvocab=$dir/entity_vocab/mluke_entity_vocab.jsonl
#entityvocab=$dir/entity_vocab/small_entity_vocab.jsonl
#modelname=xlm-roberta-base
modelname=/work/ptm/fairseq/xlm-roberta-base
#tokenizer=/workspace/ptm/fairseq/xlm-roberta-base/sentencepiece.bpe.model
mention=/work/wiki/mlm/configs/mention_candidate_paths.json

lcs=(ar bn de nl el en es fi fr hi id it ja ko pl pt ru sv sw te th tr vi zh)
for ((i=$1;i<=$2;i++))
do
  l=${lcs[$i]}
  command="poetry run python luke/cli.py build-wikipedia-pretraining-dataset-for-meae $dir/wikidb/${l}wiki.db ${modelname} ${entityvocab} $dir/augmented_dataset/dataset/${l} --mention_candidate ${mention} --use-entity-linker --sentence-splitter=${l} --include-sentences-without-entities --pool-size 8 --chunk-size 1024"
  echo $command
  $command
done

