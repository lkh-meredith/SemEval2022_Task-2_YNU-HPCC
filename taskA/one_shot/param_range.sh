#! /bin/bash
for batch_size in 4 8  
do 
	for epoches in 10
	do
		python ./prompt.py --batch_size $batch_size --epoches $epoches 
	done
done


	