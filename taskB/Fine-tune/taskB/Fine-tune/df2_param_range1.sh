#! /bin/bash
for triplet_margin in 0.1
do
    for batch_size in 16
    do 
       for epoches in 1 2 3 4
	     do
            python ./combined_with_losses1.py --batch_size $batch_size --triplet_margin $triplet_margin --epoches $epoches
	     done
    done
done
	