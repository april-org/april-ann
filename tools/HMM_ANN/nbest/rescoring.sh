#!/bin/bash

deven=$1
nbest=$2
trie_size=$3
nnlm=$4
vocab=$5
use_ccs=$6

if [ -z $use_ccs ]; then
  use_ccs="yes"
fi

scale=15

echo $@ > rescoring.conf

if [ ! -e init.opt ]; then
    echo "Needs a init.opt file!!!"
    exit 1;
fi

april=~experimentos/HERRAMIENTAS/bin/april
tools=~experimentos/HERRAMIENTAS
april_tools="$tools/april_tools"
scorer="$april $april_tools/HMM_ANN/extract_WER_weights.lua"
generate_conf=$april_tools/HMM_ANN/generate_conf.lua
mert="$april $april_tools/optimizer_weights.lua -s WER -t simplex -a 2  "

echo "# computing NNLM feature"
stdout=NNLM_feature
if [ ! -e $stdout.gz ]; then
    $april ~experimentos/HERRAMIENTAS/april_tools/NBEST/compute_nnlm_score.lua \
	$nbest $nnlm $vocab $trie_size $use_ccs $use_ccs 2> time.txt | gzip -c > $stdout.gz
fi

if [ ! -e scores.dat.gz ]; then
    $scorer -r $deven \
	-S scores.dat \
	-F feats.dat \
	-n $nbest

    gzip scores.dat
    gzip feats.dat
fi

feats=feats.dat.gz
scores=scores.dat.gz

nnlmfeats=`basename $feats .gz`.nnlm.gz
if [ ! -e $nnlmfeats ]; then
    echo "# substituting LM features with " $nnlmfeats
    $april ~experimentos/HERRAMIENTAS/april_tools/NBEST/substitute_feature.lua 2 $feats NNLM_feature.gz |
    gzip -c > $nnlmfeats
fi

if [ ! -e mert.log ]; then
    echo "# MERT"
    weights=( $(tail -n 1 "init.opt") )
    sizew=${#weights[@]}
    seed=$RANDOM
    if [ -e seed.txt ]; then
	seed=`cat seed.txt`;
    fi
    echo $seed > seed.txt
    $mert -n 20 -d 3 -S scores.dat.gz \
	-r $seed \
	-F $nnlmfeats -i init.opt >& mert.log
    grep -i "best point: " mert.log |
    cut -d':' -f 2 | cut -d'=' -f 1 > weights.txt
    weights=( $(tail -n 1 "weights.txt") )
    new_gsf=`echo "scale=$scale; ${weights[1]}/${weights[0]}" | bc`
    new_wip=`echo "scale=$scale; ${weights[2]}/${weights[0]}" | bc`
    echo "1 $new_gsf $new_wip" > weights.txt
fi

if [ ! -e dev.recog.out ]; then
    echo "# rescoring"
    $april ~experimentos/HERRAMIENTAS/april_tools/NBEST/rerank.lua \
	weights.txt $nbest $nnlmfeats dev.reranked-nbest.out.gz > dev.recog.out
fi

echo "# Evaluando el resultado"

$april $april_tools/HMM_ANN/compute_WER.lua $deven dev.recog.out > results.log
