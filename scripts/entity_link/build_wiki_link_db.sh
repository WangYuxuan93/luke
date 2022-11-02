#l=sw
lcs=(ar bn de nl el en es fi fr hi id it ja ko pl pt ru sv sw te th tr vi zh)
#lcs=(zh)
for ((i=$1;i<=$2;i++))
do
  l=${lcs[$i]}
  dir=/work/wiki
  db="$dir/wikidb/${l}wiki.db"
  mentiondb="/work/wiki/wiki_mention_db/${l}wiki-mention.db"
  output="/work/wiki/wiki_link_db/${l}wiki-link.db"
  cmd="poetry run python examples/preprocess_cli.py build-wiki-link-db $db $mentiondb $output"
  echo $cmd
  $cmd
done
