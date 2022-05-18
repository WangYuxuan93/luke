#l=sw
lcs=(ar bn de nl el en es fi fr hi id it ja ko pl pt ru sv sw te th tr vi zh)
for ((i=$1;i<=$2;i++))
do
  l=${lcs[$i]}
  dir=/workspace/wiki/wikipedia
  src="$dir/${l}wiki-20220420-pages-articles.xml.bz2"
  output="/workspace/wiki/wikidb/${l}wiki.db"
  echo python luke/cli.py build-dump-db $src $output
  python luke/cli.py build-dump-db $src $output
done
