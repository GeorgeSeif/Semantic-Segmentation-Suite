# perform grid search for hyperparameters

#nohup python3 train.py --model DeepLabV3_plus --dataset CCP --batch_size 8

learning_rate=(0.00001 0.0001 0.001 0.01 0.1 1.0)
regularization=(0.001 0.01 0.1 1.0 10.0 100.0)
batch_size=4
nb_epoch=15
dataset="CCP"
model="DeepLabV3_plus"

echo "starting grid search"
echo "learning rate => " ${learning_rate}
echo "regularization => " ${regularization}

mkdir result/grid_search 2> /dev/null
mkdir result/grid_search/log 2> /dev/null

for LEARNING_RATE in "${learning_rate[@]}"; do
	for REGULARIZATION in "${regularization[@]}"; do
	
		echo "training with learning rate " ${LEARNING_RATE} " and regularization " ${REGULARIZATION}
		python3 train.py --num_epochs ${nb_epoch} --model ${model} --dataset ${dataset} --batch_size ${batch_size} --learning_rate $LEARNING_RATE --regularization $REGULARIZATION |& tee result/grid_search/log/deeplabv3plus_CCP_LR${LEARNING_RATE}_R${REGULARIZATION}.log
		#mkdir result/grid_search/${LEARNING_RATE}_${REGULARIZATION}_deeplabv3plus_CCP 2> /dev/null
		#mv
	done
done
