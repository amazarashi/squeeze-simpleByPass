import chainer
import chainer.functions as F
import chainer.links as L
import skimage.io as io
import numpy as np
from chainer import utils

class FireModule(chainer.Chain):

    def __init__(self,input_size,s1,e1,e3):
        initializer = chainer.initializers.HeNormal()
        super(FireModule,self).__init__(
            conv1 = L.Convolution2D(input_size,s1,1,initialW=initializer),
            conv2 = L.Convolution2D(s1,e1,1,initialW=initializer),
            conv3 = L.Convolution2D(s1,e3,3,pad=(1,1),initialW=initializer),
            bn1 = L.BatchNormalization(s1),
            bn2 = L.BatchNormalization(e1+e3)
        )

    def __call__(self,x,train=False):
        h = F.relu(self.conv1(x))
        bn = self.bn1(h, test=not train)
        h1 = self.conv2(bn)
        h2 = self.conv3(h)
        h_expand = F.relu(F.concat([h1,h2],axis=1))
        bn = self.bn2(h_expand, test=not train)
        return bn

class SqueezeSimpleByPass(chainer.Chain):

    def __init__(self,category_num=10):
        initializer = chainer.initializers.HeNormal()
        super(SqueezeSimpleByPass,self).__init__(
            conv1 = L.Convolution2D(3,96,7,stride=2,initialW=initializer),
            fire2 = FireModule(96,16,64,64),
            fire3 = FireModule(128,16,64,64),
            fire4 = FireModule(128,32,128,128),
            fire5 = FireModule(256,32,128,128),
            fire6 = FireModule(256,48,192,192),
            fire7 = FireModule(384,48,192,192),
            fire8 = FireModule(384,64,256,256),
            fire9 = FireModule(512,64,256,256),
            conv10 = L.Convolution2D(512,category_num,1,stride=1,initialW=initializer),
        )

    def __call__(self,x,train=True):
        #x = chainer.Variable(x)
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h,3,stride=2,pad=1)

        h = self.fire2(h,train=train)
        h = self.fire3(h,train=train) + h
        h = self.fire4(h,train=train)
        h = F.max_pooling_2d(h,3,stride=2,pad=1)

        h = self.fire5(h,train=train) + h
        h = self.fire6(h,train=train)
        h = self.fire7(h,train=train) + h
        h = F.max_pooling_2d(h,3,stride=2,pad=1)

        h = self.fire8(h,train=train)
        h = F.max_pooling_2d(h,3,stride=2,pad=1)

        h = self.fire9(h,train=train) + h
        h = F.dropout(h,ratio=0.5,train=train)

        h = self.conv10(h)
        num, categories, y, x = h.data.shape
        h = F.reshape(F.average_pooling_2d(h,(y, x)), (num, categories))

        return h

    def calc_loss(self,y,t):
        loss = F.softmax_cross_entropy(y,t)
        return loss

    def accuracy_of_each_category(self,y,t):
        y.to_cpu()
        t.to_cpu()
        categories = set(t.data)
        accuracy = {}
        for category in categories:
            supervise_indices = np.where(t.data==category)[0]
            predict_result_of_category = np.argmax(y.data[supervise_indices],axis=1)
            countup = len(np.where(predict_result_of_category==category)[0])
            accuracy[category] = countup
        return accuracy

if __name__ == "__main__":
    imgpath = "/Users/suguru/Desktop/test.jpg"
    img = io.imread(imgpath)
    img = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
    img = img[np.newaxis]
    ex = model(img)
    print(ex)
