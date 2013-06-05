
return {
  recog="htr",
  m="BESTMODELS/hmms.lua",
  n="BESTMODELS/mlp.net",
  --hmmdefs="...", -- HTK HMMs and GMMs
  context=4,
  t="tiedlist.Graves.id",
  d="/home/experimentos/PAMI_MODELOS/Graves/lexicon_20000.intersected.SILOPT",
  lm="/home/experimentos/PAMI_MODELOS/Graves/latticeArpa_f.intersected.lira.gz",
  gsf=9,
  wip=-15,
  optsil="yes",
  feats_format="mat",
  feats_norm="pats/means_and_devs.PRESERVANDOX.lua",
  ngram_beam=2000,
  ngram_size=400,
  ngram_nstates=10000,
  -- one_step="no",
  -- nested="no",
  -- dog="no",
  -- filter="...",
  -- cache_size=18,
  -- trie_size=25,
  -- ose="htr"
  -- use_word_probs="yes",
  -- unk=nil,
  -- only valid for two steps algorithm
  wgen_beam=1000,
  wgen_size=200,
  wgen_nstates=4000,
}
