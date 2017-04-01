# About

SqueezeNet-SimpleByPass by chainer

# Paper

[160224 SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

# Model

squeezeNet with simple-bypass

# How to run
git clone git@github.com:amazarashi/squeeze-chainer.git

cd ./squeeze-simpleByPass-chainer

python main.py -g 1

# Inspection

##### dataset
 Cifar10 [link](https://www.cs.toronto.edu/~kriz/cifar.html)

##### Result

- Optimizer momentumSGD (0.04[as mentioned in paper] ~ scheduling lineary )

![accuracy_0.04](https://github.com/amazarashi/squeeze-simpleByPass-chainer/blob/develop/result/0.04/accuracy.png "")

![loss_0.04](https://github.com/amazarashi/squeeze-simpleByPass-chainer/blob/develop/result/0.04/loss.png "")
