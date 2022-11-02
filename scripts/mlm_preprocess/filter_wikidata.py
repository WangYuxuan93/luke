import json
import ujson
import argparse
import bz2

parser = argparse.ArgumentParser(description="Filter wikidata by entity vocab")
parser.add_argument("wikidata", type=str, help="Wikidata file path")
parser.add_argument("--vocab", type=str, help="entity vocab file path")

args = parser.parse_args()

print (args.wikidata, args.vocab)

def print_json(a):
    output = json.dumps(a, indent=2)
    print (output)

lcs = ["en", "ar","bn","de","nl","el","es","fi","fr","hi","id","it","ja","ko","pl","pt","ru","sv","sw","te","th","tr","vi","zh"]

num_mismatch = 0
num_not_found = 0

with bz2.BZ2File(args.wikidata) as f:
    for (n, line) in enumerate(f):
        if n & 1000 == 0 and n != 0:
            print ("Processed {} lines, Mismatch/Notfound: {}/{}".format(n,num_mismatch, num_not_found))

        line = line.rstrip().decode("utf-8")
        if line in ("[","]"):
            continue

        if line[-1] == ",":
            line = line[:-1]
        obj = ujson.loads(line)
        if obj["type"] != "item":
            continue

        labels = obj["labels"]
        sitelinks = obj["sitelinks"]
        output = json.dumps(obj, indent=2)
        for lc in labels:
            if lc not in lcs: continue
            lcwiki = lc+"wiki"
            if lcwiki not in sitelinks:
                #print ("NOT FOUND {}'s sitelink".format(lc))
                #print_json(labels[lc])
                #print_json(obj["sitelinks"])
                num_not_found += 1
            else:
                if not sitelinks[lcwiki]["title"] == labels[lc]["value"]:
                    #print ("TITLE AND VALUE mismatch for {}".format(lc))
                    #print_json(labels[lc])
                    #print_json(sitelinks[lcwiki])
                    num_mismatch += 1
