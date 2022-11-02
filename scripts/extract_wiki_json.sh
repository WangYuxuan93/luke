lcs=(ar bn de nl el en es fi fr hi id it ja ko pl pt ru sv sw te th tr vi zh)
#lcs=(zh)
for ((i=$1;i<=$2;i++))
do
  l=${lcs[$i]}
  dump=/work/wiki/wikipedia/${l}wiki-20220420-pages-articles.xml.bz2
  output=/work/wiki/wiki_json/${l}

  cmd="python -m wikiextractor.WikiExtractor $dump -o $output -b 100M --json"
  echo $cmd
  $cmd

done

