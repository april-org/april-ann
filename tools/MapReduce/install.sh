#!/bin/sh
dest=/etc/APRIL-ANN-MAPREDUCE/
mkdir -p $dest
cp -i etc/* $dest
echo "Please, edit scripts at $dest"
