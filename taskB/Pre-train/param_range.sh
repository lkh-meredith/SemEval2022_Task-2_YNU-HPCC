#! /bin/bash
for batch_size in 16
do
   for epoches in 1 2 3 4 5 6 7 8 9 10
   do
    python ./pretrainSentTransModel_a.py --batch_size $batch_size --epoches $epoches
   done
done


	