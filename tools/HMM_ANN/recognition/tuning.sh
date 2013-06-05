#!/bin/bash

initrun=$1
lastrun=$2
defopt=$3
lm=$4
devtarget=$5
tuningdir=${6}/
numnbest=${7}    # por defecto 500
maxhout=${8}     # por defecto 100
use_prev_nbest=${9} # por defecto yes
initweights=${10}

if [ -z $APRILANN_ROOT ]; then
    echo "APRILANN_ROOT environment variable is needed"
    exit 1
fi

gsf=10
wip=0

scale=15
epsilon=0.00001

if [ -z $use_prev_nbest ]; then
    use_prev_nbest="yes"
fi

if [ ! -z $initweights ]; then
    waux=( $(tail -n 1 "$initweights") )
    gsf=`echo "scale=$scale; ${waux[1]}/${waux[0]}" | bc`
    wip=`echo "scale=$scale; ${waux[2]}/${waux[0]}" | bc`
fi

if [ -z $maxhout ]; then maxhout=100; fi
if [ -z $numnbest ]; then numnbest=500; fi
optimizer=simplex;
objfunc="WER";

aprilann_tools=$APRILANN_ROOT/tools
aprilann=$APRILANN_ROOT/bin/april-ann

recog=$aprilann_tools/HMM_ANN/recog.lua
scorer="$aprilann $aprilann_tools/HMM_ANN/extract_WER_weights.lua"
generate_conf=$aprilann_tools/HMM_ANN/generate_conf.lua
mert="$aprilann $aprilann_tools/optimizer_weights.lua -s WER -t simplex -a 2  "

if [ -z $lm ]; then tm="nil"; fi
if [ $lm = "nil" ]; then
    lm="";
else
    lm="--lm=$lm"
fi

if [ -z $tuningdir ]; then tuningdir=tuning; fi
mkdir -p $tuningdir

if [ $initrun -gt 1 ]; then
    prevrun=`expr $initrun - 1`
    if [ ! -e $tuningdir/run$prevrun/weights.txt ]; then
	echo "ERROR!!! $tuningdir/run$prevrun/weights.txt not found"
	exit 128
    fi
    weights=( $(tail -n 1 "$tuningdir/run$prevrun/weights.txt") )
    gsf=`echo "scale=$scale; ${weights[1]}/${weights[0]}" | bc`
    wip=`echo "scale=$scale; ${weights[2]}/${weights[0]}" | bc`
else
    cp $defopt $tuningdir/tuning-conf.lua
fi

if [ ! -e $tuningdir/tuning-conf.lua ]; then
    cp $defopt $tuningdir/tuning-conf.lua
fi

endrun=$initrun

echo $0 $@ > $tuningdir/last_tuning_exec.sh
chmod +x $tuningdir/last_tuning_exec.sh

rm -f $tuningdir/finished.txt

for run in `seq $initrun $lastrun`; do
    mkdir -p $tuningdir/run$run
    nbestlog=$tuningdir/run$run/run$run.best$numnbest.out
    tuplenbestlog=$tuningdir/run$run/run$run.tuplebest$numnbest.out
    featlog=$tuningdir/run$run/run$run.feats.opt
    stdout=$tuningdir/run$run/run$run.out
    stderr=$tuningdir/run$run/run$run.err
    
    # cargamos los pesos y generamos gsf y wip
    echo "FECHA: "`date`
    echo "Reconociendo... etapa $run (esto puede tarder MUCHO tiempo)"
    
    $aprilann $generate_conf $tuningdir/tuning-conf.lua $lm \
	   --uniq-nbest=yes --gsf=$gsf --wip=$wip > $tuningdir/run$run/conf.lua
    
    $aprilann $recog --nbest=$numnbest --save-nbest=$nbestlog --features=$featlog \
	  --max-h-out=$maxhout \
	  -f $tuningdir/run$run/conf.lua > $stdout 2> $stderr
    if [ $? -ne 0 ]; then
	echo "ERROR!!!"
	exit 128;
    fi
    gzip -f $nbestlog
    gzip -f $featlog
    
    # scorer
    refs=$devtarget
    
    prevsc=""
    prevfe=""
    if [ $run -gt 1 ]; then
	if [ $use_prev_nbest = "yes" ]; then
	    prevsc="-R $tuningdir/prev-scores.dat.gz"
	    prevfe="-E $tuningdir/prev-feats.dat.gz"
	fi
    fi
    $scorer -r $refs \
	    -S $tuningdir/scores.dat \
	    -F $tuningdir/feats.dat \
	    -n $nbestlog.gz $prevsc $prevfe
    if [ $? -ne 0 ]; then
	echo "ERROR!!!"
	exit 128;
    fi

    echo "MERT"
    echo "1 $gsf $wip" > $tuningdir/init.opt
    oldpwd=`pwd`
    seed=$RANDOM
    if [ -e $tuningdir/run$run/run$run.seed.txt ]; then
	seed=`cat $tuningdir/run$run/run$run.seed.txt`;
    fi
    echo $seed > $tuningdir/run$run/run$run.seed.txt
    cd $tuningdir/
    $mert -n 20 -d 3 -S scores.dat \
	-r $seed \
	-F feats.dat -i init.opt >& run$run/run$run.mert.log
    if [ $? -ne 0 ]; then
	echo "ERROR!!!"
	exit 128;
    fi
    cd $oldpwd
    
    gzip -f $tuningdir/scores.dat
    gzip -f $tuningdir/feats.dat
    
    mv -f $tuningdir/scores.dat.gz $tuningdir/prev-scores.dat.gz
    mv -f $tuningdir/feats.dat.gz $tuningdir/prev-feats.dat.gz
    
    lastwfile=$tuningdir/run$run/weights.txt
    grep -i "best point: " $tuningdir/run$run/run$run.mert.log |
    cut -d':' -f 2 | cut -d'=' -f 1 > $lastwfile
    weights=( $(tail -n 1 "$tuningdir/run$run/weights.txt") )
    new_gsf=`echo "scale=$scale; ${weights[1]}/${weights[0]}" | bc`
    new_wip=`echo "scale=$scale; ${weights[2]}/${weights[0]}" | bc`
    stop=1
    diffs=""
    echo GSF= $new_gsf   WIP= $new_wip

    aux=`echo "${gsf[$i]} - ${new_gsf[$i]}" | bc`
    diff=`echo "scale=$scale; sqrt($aux * $aux)" | bc` # hack FEO para valor absoluto
    diffs="$diffs $diff"
    res=`echo "$diff > $epsilon" | bc`
    if (( "$res" == 1 )); then
	stop=0
    fi


    aux=`echo "${wip[$i]} - ${new_wip[$i]}" | bc`
    diff=`echo "scale=$scale; sqrt($aux * $aux)" | bc` # hack FEO para valor absoluto
    diffs="$diffs $diff"
    res=`echo "$diff > $epsilon" | bc`
    if (( "$res" == 1 )); then
	stop=0
    fi
    
    echo "sizew= 3    stop= $stop    dir= $tuningdir"
    endrun=$run
    if (( "$stop" == 1 )); then
	echo $endrun > $tuningdir/finished.txt
	break;
    fi
    
    gsf=$new_gsf
    wip=$new_wip
done

cp -f $tuningdir/run$endrun/conf.lua $tuningdir/conf.lua

oldpwd=`pwd`
cd $tuningdir
rm -f last
ln -s run$endrun last
cd $oldpwd

echo $endrun > $tuningdir/finished.txt
echo "Stop en la etapa " $endrun
echo "OK"
