#!/bin/bash
conf_file=$1
outdir=$2
ini=$3
fin=$4

april=/home/experimentos/HERRAMIENTAS/bin/april
recog=/home/experimentos/HERRAMIENTAS/april_tools/HMM_ANN/recog.lua
mkdir -p $outdir
for i in `seq $ini $fin`; do
     $april $recog -f $conf_file -m models/hmms/ci?_cd?_em$i.lua \
            -n models/redes/ci?_cd?_em$i.net 2> $outdir/resultados.em`printf "%02d" $i`.log > $outdir/resultados.em`printf "%02d" $i`.out;
done
