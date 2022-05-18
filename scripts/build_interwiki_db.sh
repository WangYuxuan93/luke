l=sw
wikidata=/workspace/wiki/wikidata/wikidata-20220117-all.json.bz2
db="data/wikidb/interwiki.db"

python luke/cli.py build-interwiki-db $wikidata $db
