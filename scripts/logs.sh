#!/bin/bash

LOGDIR=$1
TAG=$2

git pull
mkdir $LOGDIR/code
cp -r src $LOGDIR/code/src
cp $TAG.ipynb $LOGDIR/code/$TAG.ipynb
zip $LOGDIR.zip -r $LOGDIR
cp $LOGDIR.zip ../drive/MyDrive/$TAG.zip
