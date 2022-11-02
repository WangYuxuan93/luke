
lcs=(ar bn de nl el en es fi fr hi id it ja ko pl pt ru sv sw te th tr vi zh)
for ((i=$1;i<=$2;i++))
do
  lc=${lcs[$i]}
  cmd="python scripts/data_shard_wiki_tokenize.py --data wiki_raw_split@@@${lc}@@@0 --output_folder wiki_tokenize_split/ --spm /work/ptm/transformers/xlm-roberta-base/sentencepiece.bpe.model"
  echo $cmd
  $cmd
done
