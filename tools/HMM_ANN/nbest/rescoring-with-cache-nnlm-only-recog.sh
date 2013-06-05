#!/bin/bash

prefix=$1
feats=$2
nbest=$3
trie_size=$4
nnlm=$5
vocab=$6
extra_cache_set=$7
use_ccs=$8

scale=15
epsilon=0.0001

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

feats0=$prefix.reranked-feats.run0.out.gz
nbest0=$prefix.reranked-nbest.run0.out.gz
if [ ! -e $nbest0 ]; then
    ln -s $nbest $nbest0;
    ln -s $feats $feats0;
fi
nbest=$nbest0
feats=$feats0

prevrun=0
run=1
while true; do
    echo "# computing NNLM feature at run $run with $nbest"
    stdout=$prefix.NNLM_feature.run$run
    if [ ! -e $stdout.gz ]; then
	$april ~experimentos/HERRAMIENTAS/april_tools/NBEST/compute_nnlm_score.lua \
	    $nbest $nnlm $vocab $trie_size $use_ccs $use_ccs $extra_cache_set 2> $prefix.time.run$run.txt | gzip -c > $stdout.gz
    fi
    nnlmfeats=`basename $feats .gz`.nnlm.gz
    if [ ! -e $nnlmfeats ]; then
	echo "# substituting LM features with " $nnlmfeats
	$april ~experimentos/HERRAMIENTAS/april_tools/NBEST/substitute_feature.lua 2 $feats $stdout.gz |
	gzip -c > $nnlmfeats
    fi
    
    if [ ! -e $prefix.recog.run$run.out ]; then
	echo "# rescoring at run $run"
	$april ~experimentos/HERRAMIENTAS/april_tools/NBEST/rerank.lua \
	    weights.run`cat finished.txt`.txt $nbest $nnlmfeats \
	    $prefix.reranked-nbest.run$run.out.gz \
	    $prefix.reranked-feats.run$run.out.gz > $prefix.recog.run$run.out
    fi

    stop=0
    if [ -e $prefix.recog.run$prevrun.out ]; then
	diff=`diff $prefix.recog.run$prevrun.out $prefix.recog.run$run.out | wc -l`
	if (( "$diff" == 0 )); then
	    stop=1
	fi
    fi
    if (( "$stop" == 1 )); then
	echo $run > $prefix.finished.txt
	break;
    fi
    nbest=$prefix.reranked-nbest.run$run.out.gz
    feats=$prefix.reranked-feats.run$run.out.gz
    prevrun=$run
    run=`expr $run + 1`
done
