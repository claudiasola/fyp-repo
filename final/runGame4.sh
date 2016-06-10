#!/bin/bash

agent_num=$1
seed=1000
lower=3
upper=9
mean=5
var=1.25
COMPETITION_NUM=10000
#START=2

python setupGame.pyc $agent_num $seed $lower $upper $mean $var $COMPETITION_NUM

#for COMPETITION_NUMBER in 10 
#for COMPETITION_NUMBER in 10 100 200 300 500 600 800 950 
for COMPETITION_NUMBER in {7501..10000}  
#for (( c=$START; c<=$COMPETITION_NUM; c=$(($c+50))))
do
#       python game.py $agent_num $seed $lower $upper $c
        python game.pyc $agent_num $seed $lower $upper $COMPETITION_NUMBER

done







