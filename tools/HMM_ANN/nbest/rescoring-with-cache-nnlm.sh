#!/bin/bash

deven=$1
nbest=$2
trie_size=$3
nnlm=$4
vocab=$5
extra_cache_set=$6
use_ccs=$7

if [ -z $use_ccs ]; then
  use_ccs="yes"
fi

scale=15
epsilon=0.0001

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

nbest0=dev.run0.reranked-nbest.out.gz
if [ ! -e $nbest0 ]; then
    ln -s $nbest $nbest0;
fi
nbest=$nbest0
prevrun=0
run=1
while true; do
    echo "# computing NNLM feature for run $run using $nbest"
    stdout=NNLM_feature.run$run
    if [ ! -e $stdout.gz ]; then
	$april ~experimentos/HERRAMIENTAS/april_tools/NBEST/compute_nnlm_score.lua \
	    $nbest $nnlm $vocab $trie_size $use_ccs $use_ccs $extra_cache_set 2> time.run$run.txt | gzip -c > $stdout.gz
    fi

    if [ ! -e scores.run$run.dat.gz ]; then
	$scorer -r $deven \
	    -S scores.run$run.dat \
	    -F feats.run$run.dat \
	    -n $nbest
	
	gzip scores.run$run.dat
	gzip feats.run$run.dat
    fi

    feats=feats.run$run.dat.gz
    scores=scores.run$run.dat.gz
    
    nnlmfeats=`basename $feats .gz`.nnlm.gz
    if [ ! -e $nnlmfeats ]; then
	echo "# substituting LM features with " $nnlmfeats
	$april ~experimentos/HERRAMIENTAS/april_tools/NBEST/substitute_feature.lua 2 $feats $stdout.gz |
	gzip -c > $nnlmfeats
    fi
    
    if [ ! -e mert.run$run.log ]; then
	echo "# MERT"
	weights=( $(tail -n 1 "init.opt") )
	sizew=${#weights[@]}
	seed=$RANDOM
	if [ -e seed.run$run.txt ]; then
	    seed=`cat seed.run$run.txt`;
	fi
	echo $seed > seed.run$run.txt
	$mert -n 20 -d 3 -S scores.run$run.dat.gz \
	    -r $seed \
	    -F $nnlmfeats -i init.opt >& mert.run$run.log
	grep -i "best point: " mert.run$run.log |
	cut -d':' -f 2 | cut -d'=' -f 1 > weights.run$run.txt
	weights=( $(tail -n 1 "weights.run$run.txt") )
	new_gsf=`echo "scale=$scale; ${weights[1]}/${weights[0]}" | bc`
	new_wip=`echo "scale=$scale; ${weights[2]}/${weights[0]}" | bc`
	echo "1 $new_gsf $new_wip" > weights.run$run.txt
    fi
    if [ ! -e dev.run$run.recog.out ]; then
	echo "# rescoring at run $run"
	$april ~experimentos/HERRAMIENTAS/april_tools/NBEST/rerank.lua \
	    weights.run$run.txt $nbest $nnlmfeats dev.run$run.reranked-nbest.out.gz > dev.run$run.recog.out
    fi
    
    echo "# Evaluando el resultado at run $run"
    
    $april $april_tools/HMM_ANN/compute_WER.lua $deven dev.run$run.recog.out > results.run$run.log
    
    stop=0
    if [ -e results.run$prevrun.log ]; then
	wer=`grep WER results.run$run.log | awk '{ print $2 }'`
	prevwer=`grep WER results.run$prevrun.log | awk '{print $2}'`
	diff=`echo "scale=$scale; $prevwer - $wer" | bc`
	res=`echo "scale=$scale; $diff*$diff > $epsilon*$epsilon" | bc`
	# echo $diff ">" $epsilon "=" $res
	if (( "$res" == 0 )); then
	    stop=1
	fi
    fi
    if (( "$stop" == 1 )); then
	echo $run > finished.txt
	break;
    fi
    nbest=dev.run$run.reranked-nbest.out.gz
    prevrun=$run
    run=`expr $run + 1`
done
