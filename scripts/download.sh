lcs=(de nl el es fi fr hi id it ja ko pl pt ru sv sw te th tr vi)
for ((i=$1;i<=$2;i++))
do
  lc=${lcs[$i]}
  url=https://wikimedia.bringyour.com/${lc}wiki/20220420/${lc}wiki-20220420-pages-articles.xml.bz2
  echo "#####"Downloading $url
  wget -c $url
done
