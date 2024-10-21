#!/bin/bash
for i in {1..10}
do
    python finetune_esol.py 
done

for i in {1..10}
do
    python finetune_freesolv.py 
done

for i in {1..10}
do
    python finetune_lipo.py 
done
