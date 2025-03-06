#!/usr/bin/env bash
# Copy over tarballs for fild8, unpack and create parquet files

eval "$(/data/users/cap/miniconda3/bin/conda shell.bash hook)"
conda activate py310

DPATH=/net/isilon/ifs/arch/home/glatopr/GLATMODEL #202202/fild8/fild8/
YYYY=2022
for MM in $(seq -w 2 12); do
TBALL=$DPATH/${YYYY}${MM}/fild8/fild8/fild8_${YYYY}${MM}.tar
if [ -f $TBALL ]; then
  cp $TBALL .
  tar xvf $TBALL
  cd fild8
  for F in *gz; do
    gunzip $F
    FILE=$(basename $F .gz)
    ../process_profiles_fild8.py $FILE
    rm $FILE
  done
  cd ..
  rmdir fild8  
else 
  echo "$TBALL not available"
fi
done
