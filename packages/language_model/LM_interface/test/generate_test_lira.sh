ngram-count -order 3 -lm lm3gram.arpa -text my_text.txt -write-vocab lm3gram.voc -unk 2> /dev/null
april-ann.debug /home/baha/april-ann/tools/LM/arpa2lira.lua lm3gram.voc lm3gram.arpa lm3gram.lira
