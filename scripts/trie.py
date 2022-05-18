from marisa_trie import Trie
import sys

a = ["abc:en", "ab:en", "dc:zh", "edn:zh", "bc:en"]
a_trie = Trie(a)

print (a_trie)

print (a)
out = a_trie[a[int(sys.argv[1])]]
print ("id:{},a:{},a_trie:{}".format(sys.argv[1], a[int(sys.argv[1])], out))

print (a_trie.restore_key(out))
