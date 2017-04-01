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

- Optimizer momentumSGD (0.01[custom value for batch normarlization] ~ scheduling lineary )

accuracy
![accuracy_0.01](https://github.com/amazarashi/squeeze-simpleByPass-chainer/blob/develop/result/0.01/accuracy.png "")

loss
![loss_0.01](https://github.com/amazarashi/squeeze-simpleByPass-chainer/blob/develop/result/0.01/loss.png "")

- Optimizer momentumSGD (0.04[as mentioned in paper] ~ scheduling lineary )

accuracy
![accuracy_0.02](https://github.com/amazarashi/squeeze-simpleByPass-chainer/blob/develop/result/0.02/accuracy.png "")

loss
![loss_0.02](https://github.com/amazarashi/squeeze-simpleByPass-chainer/blob/develop/result/0.02/loss.png "")
