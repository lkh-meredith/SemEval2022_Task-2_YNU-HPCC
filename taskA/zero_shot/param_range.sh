#! /bin/bash
for batch_size in 32
do 
	for epoches in 10
	do
		python ./train_bert_12cls.py --batch_size $batch_size --epoches $epoches
	done
done


	