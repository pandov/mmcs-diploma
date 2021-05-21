#!/bin/bash

LOGDIR=$1
TAG=$2

git pull
mkdir $LOGDIR/code
cp -r src $LOGDIR/code/src
cp $TAG.ipynb $LOGDIR/code/$TAG.ipynb
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
rm -r $LOGDIR/checkpoints/*_full.pth
rm -r $LOGDIR/checkpoints/train.*.pth
zip $LOGDIR.zip -r $LOGDIR
cp $LOGDIR.zip ../drive/MyDrive/$TAG.zip
