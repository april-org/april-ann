#!/bin/bash

refs=$1
nbest=$2
score=$3
numreps=$4
baseline=$5
maxnbest=$6
confidence=$7 # by default 0.95

tools=/home/experimentos/HERRAMIENTAS
april_tools=/home/experimentos/HERRAMIENTAS/april_tools
april=$tools/bin/april

refs2=`mktemp /tmp/april_XXXXXXX`
refs3=`mktemp /tmp/april_XXXXXXX`
cp   $refs    $refs2
zcat $refs2 > $refs3 ||
cat  $refs2 > $refs3
refs=$refs3

nbest2=`mktemp /tmp/april_XXXXXXX`
zcat $nbest > $nbest2 ||
cat  $nbest > $nbest2
nbest=$nbest2

if [ ! -z $baseline ]; then
    baseline2=`mktemp /tmp/april_XXXXXXX`
    baseline3=`mktemp /tmp/april_XXXXXXX`
    cp   $baseline    $baseline2
    zcat $baseline2 > $baseline3 ||
    cat  $baseline2 > $baseline3
    baseline=$baseline3
fi

scores=`mktemp /tmp/april_XXXXXXX`
feats=`mktemp /tmp/april_XXXXXXX`

scores2=""
feats2=""

scorer=""

if [ $score = "WER" ]; then
    scorer="$april $april_tools/HMM_ANN/extract_WER_weights.lua"  
    $scorer -r $refs \
	-S $scores \
	-F $feats \
	-n $nbest || exit
    if [ ! -z $baseline ]; then
	scores2=`mktemp /tmp/april_XXXXXXX`
	feats2=`mktemp /tmp/april_XXXXXXX`
	$scorer -r $refs \
	    -S $scores2 \
	    -F $feats2 \
	    -n $baseline  || exit
    fi
elif [ $score = "CER" ]; then
    scorer="$april $april_tools/HMM_ANN/extract_WER_weights.lua -t CER"  
    $scorer -r $refs \
	-S $scores \
	-F $feats \
	-n $nbest || exit
    if [ ! -z $baseline ]; then
	scores2=`mktemp /tmp/april_XXXXXXX`
	feats2=`mktemp /tmp/april_XXXXXXX`
	$scorer -r $refs \
	    -S $scores2 \
	    -F $feats2 \
	    -n $baseline  || exit
    fi
    score="WER"
elif [ $score = "SER" ]; then
    scorer="$april $april_tools/HMM_ANN/extract_WER_weights.lua -t SER"
    $scorer -r $refs \
	-S $scores \
	-F $feats \
	-n $nbest || exit
    if [ ! -z $baseline ]; then
	scores2=`mktemp /tmp/april_XXXXXXX`
	feats2=`mktemp /tmp/april_XXXXXXX`
	$scorer -r $refs \
	    -S $scores2 \
	    -F $feats2 \
	    -n $baseline  || exit
    fi
    score="WER"
elif [ $score = "TER" ]; then
    scorer="$april_tools/MT/compute_TER.sh"
    $scorer $refs $nbest $scores  || exit
    $april $april_tools/MT/add_head_to_scorer_from_nbest.lua $scores $nbest > $scores.2
    mv -f $scores.2 $scores
    if [ ! -z $baseline ]; then
	scores2=`mktemp /tmp/april_XXXXXXX`
	feats2=`mktemp /tmp/april_XXXXXXX`
	$scorer $refs $baseline $scores2  || exit
	$april $april_tools/MT/add_head_to_scorer_from_nbest.lua $scores2 $baseline > $scores.2
	mv -f $scores.2 $scores2
    fi
elif [ $score = "BLEU" ]; then
    scorer="$tools/bin/extractor"
    $scorer -r $refs \
	--scconfig "reflen:closest" \
	-S $scores \
	-F $feats \
	-n $nbest  || exit
    if [ ! -z $baseline ]; then
	scores2=`mktemp /tmp/april_XXXXXXX`
	feats2=`mktemp /tmp/april_XXXXXXX`
	$scorer -r $refs \
	    --scconfig "reflen:closest" \
	    -S $scores2 \
	    -F $feats2 \
	    -n $baseline  || exit
    fi
else
    echo "Invalid score type: $score"
    exit
fi

echo "# $@"
($april $april_tools/NBEST/compute_errors.lua $scores $score $numreps $scores2 $maxnbest $confidence &&
rm -f $scores $feats $scores2 $refs2 $nbest2 $baseline2 $baseline3 $refs3) || echo $scores $scores2 > /dev/stderr
