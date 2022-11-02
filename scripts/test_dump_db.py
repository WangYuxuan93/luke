from wikipedia2vec.dump_db import DumpDB
import sys

dump_db_file = sys.argv[1]
dump_db = DumpDB(dump_db_file)


title = "Royal College of Obstetricians and Gynaecologists"
title = "Category:People from Luanda"
t = dump_db.resolve_redirect(title)

print (t)
