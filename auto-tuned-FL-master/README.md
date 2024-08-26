
## Requirments
Required packages can be found in `requirments.txt`.

##验证试验:
python src/run_main.py --model=resnet18 --dataset=cifar --alg=fedavg \
--diric=1 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64 --arg=1

python src/run_main.py --model=resnet18 --dataset=cifar --alg=fedavg \
--diric=1 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64 --arg=3

python src/run_main.py --model=resnet18 --dataset=cifar --alg=fedavg \
--diric=1 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64 --arg=5

python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=fedavg \
--diric=1 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

python src/run_main.py --model=resnet18 --dataset=cifar100 --alg=fedavg \
--diric=1 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

##CIFAR10对比实验:
python src/run_main.py --model=resnet18 --dataset=cifar --alg=fedavg \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

python src/run_main.py --model=resnet18 --dataset=cifar --alg=fedprox \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

python src/run_main.py --model=resnet18 --dataset=cifar --alg=fedadam \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

python src/run_main.py --model=resnet18 --dataset=cifar --alg=moon \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

python src/run_main.py --model=resnet18 --dataset=cifar --alg=feddc \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

##MNIST对比实验:
python src/run_main.py --model=cnn --dataset=mnist --alg=fedavg \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

python src/run_main.py --model=cnn --dataset=mnist --alg=fedprox \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

python src/run_main.py --model=cnn --dataset=mnist --alg=fedadam \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

python src/run_main.py --model=cnn --dataset=mnist --alg=moon \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

python src/run_main.py --model=cnn --dataset=mnist --alg=feddc \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

##CIFAR100对比实验:
python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=fedavg \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=fedprox \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=fedadam \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=moon \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=feddc \
--diric=0.01 --epochs=2000 --optimizer=sgd --exp_name=exp1 --local_bs=64 --local_ep=1 --item_per_user=64

## Data
* Datasets will be automatically downloaded from torchvision datasets.
* Experiments are run on MNIST, Fashion MNIST, CIFAR10, CIFAR100 datasets for image classification tasks, and on Agnews and Dbpedia datasets for text classification tasks.

## Usage
* To run an image classification experiment on CIFAR100 dataset using ResNet-18:
```
python src/run_main.py --model=resnet18 --dataset=cifar100 --alg=fedavg \
--diric=1 --epochs=2000 --optimizer=adsgd --exp_name=exp1 --local_bs=64 --local_ep=1 
```
* To run a text classification experiment on Agnews dataset using DistilBERT:
```
python src/federated_main_cuda0.py --is_text=true --model=bert --dataset=agnews --alg=fedavg \
--diric=1 --epochs=2000 --optimizer=adsgd --exp_name=exp1 --local_bs=64 --local_ep=1 
```
There are other parameters---please refer to the options section below.
For more examples, please refer to `run.sh`.

## Options
Some details of the options are provided below. Please also refer to ```options.py```.

* ```--alg:```  Default: 'fedavg'. Options: 'fedavg', 'moon', 'fedprox', 'fedadam'.
* ```--optimizer:```  Default: 'adsgd'. Options: 'adsgd', 'sgd', 'adam', 'adagrad', 'sps'.
* ```--dataset:```  Default: 'cifar'. Options: 'mnist', 'fmnist', 'cifar', 'cifar100', 'agnews', 'dbpedia_14'.
* ```--model:```    Default: 'cnn'. Options: 'cnn', 'resnet18', 'resnet50', 'bert'.
* ```--epochs:```   Number of rounds of training.
* ```--verbose:```  Detailed log outputs. Set to 0 to deactivate.
* ```--diric:```    Dirichlet concentration parameter for generating non-iid federated data. The default is 0.1.
* ```--num_users:```Number of users. The default is 100.
* ```--frac:```     Fraction of participating users. The default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. The default is 1.
* ```--local_bs:``` Batch size of local updates in each user. The default is 64.
