#!/bin/bash

prefix=$1
feats=$2
nbest=$3
trie_size=$4
nnlm=$5
vocab=$6
use_ccs=$7

if [ -z $use_ccs ]; then
  use_ccs="yes"
fi

if [ -e $prefix ]; then
    echo "ERROR, exists $prefix, sure this is PREFIX???"
    exit 12;
fi

april=~experimentos/HERRAMIENTAS/bin/april
tools=~experimentos/HERRAMIENTAS
april_tools="$tools/april_tools"
generate_conf=$april_tools/HMM_ANN/generate_conf.lua

echo "# computing NNLM feature"
stdout=$prefix.NNLM_feature
if [ ! -e $stdout.gz ]; then
    $april ~experimentos/HERRAMIENTAS/april_tools/NBEST/compute_nnlm_score.lua \
	$nbest $nnlm $vocab $trie_size $use_ccs $use_ccs 2> $prefix.time.txt | gzip -c > $stdout.gz
fi

nnlmfeats=$prefix.`basename $feats .gz`.nnlm.gz
if [ ! -e $nnlmfeats ]; then
    echo "# substituting LM features with " $nnlmfeats
    $april ~experimentos/HERRAMIENTAS/april_tools/NBEST/substitute_feature.lua 2 $feats $prefix.NNLM_feature.gz |
    gzip -c > $nnlmfeats
fi

if [ ! -e $prefix.recog.out ]; then
    echo "# rescoring"
    $april ~experimentos/HERRAMIENTAS/april_tools/NBEST/rerank.lua \
	weights.txt $nbest $nnlmfeats $prefix.reranked-nbest.out.gz > $prefix.recog.out
fi
