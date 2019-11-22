###############################################################################


# BP神经网络 by WRC


###############################################################################


# 数据集调用阶段
#from sklearn.datasets import load_iris  # 从库中直接load数据集
#data = load_iris() # 调用方法
#print(dir(data))  # 查看data所具有的属性或方法
#print(data.DESCR)  # 查看数据集的简介

# 数据集处理阶段
#import pandas as pd # 调用pandas处理数据
#a=pd.DataFrame(data=data.data, columns=data.feature_names)  # 直接读到pandas的数据框中
#print(a.head()) # 展示前五行数据

# 换一种方式调用数据集，下载好数据集的csv文件
import math  # 调用math库
import random  # 调用随机数库
import pandas as pd  # 调用数组处理库
import numpy as np  # 调用数学数组处理库

# 打开csv文件，发现有三类花，下面建立label字典
# 花品种科普，setoma山鸢尾，versicolor变色鸢尾，virginica维吉尼亚鸢尾
flowerLabels = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}


def rand(a, b):
    #random.seed(0)  # 是否让随机数固定
    #return (b - a)*random.random()+a   # 第一种生成随机数的写法
    return random.uniform(a, b)  # 第二种生成随机数的写法

# 0矩阵的生成方式1


def initMatrix(i, j, fill=0.0):
    m = []
    for a in range(i):
        m.append([fill]*j)
    return m


x = initMatrix(3, 4)

# 0矩阵的生成方式2


def initMatrixNP(i, j, fill=0.0):
    return np.ones((3, 4))*fill

# 详细参考sigmoid公式 1/（1+e^(-x)）


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(sig):
    return sig*(1-sig)


class BPNN:  # Back Propagation Neural Network 即反向传播神经网络
    #
    def __init__(self, input, hidden, output):
        # 输入层、隐层、输出层初始化
        self.input = input + 1  # 在输入最后加一个偏置bias
        self.hidden = hidden + 1  # 在隐层最后加一个偏置bias
        self.output = output

        # 激活层设置
        self.activation_input = [1.0] * self.input
        self.activation_hidden = [1.0] * self.hidden
        self.activation_output = [1.0] * self.output

        # 建立权重
        self.weight_input = initMatrix(self.input, self.hidden)
        self.weight_output = initMatrix(self.hidden, self.output)

        # 随机权重初始化
        for i in range(self.input):
            for j in range(self.hidden):
                self.weight_input[i][j] = rand(-0.2, 0.2)
        for j in range(self.hidden):
            for k in range(self.output):
                self.weight_output[j][k] = rand(-2, 2)

    def update(self, new_input):
        if len(new_input) != self.input - 1:    # 记得比较之前把bias排除掉
            raise ValueError('input error: not matching!')

        # 激活输入层赋值，即激活输入层
        for i in range(self.input - 1):  # 记得减去偏置
            self.activation_input[i] = new_input[i]

        # 激活隐藏层赋值，即激活隐藏层
        for j in range(self.hidden - 1):    # 记得减去偏置
            sum = 0.0
            for i in range(self.input):
                sum = sum + self.activation_input[i] * self.weight_input[i][j]
            self.activation_hidden[j] = sigmoid(sum)

        # 激活输出层赋值，即输出层
        for k in range(self.output):
            sum1 = 0.0   # 虽然是局部变量，但是还是换个名字容易看一眼,当然不换名字可以节省变量空间
            for j in range(self.hidden):
                sum1 = sum1 + \
                    self.activation_hidden[j] * self.weight_output[j][k]
            self.activation_output[k] = sigmoid(sum1)

        return self.activation_output[:]

    def backPropagate(self, targets, lr):   # 反向传播
        # 反向传播主要用来计算误差，由于是反向传播，因此反向计算误差
        # 由于输入层不需要更新误差，所以只有两个误差计算部分

        # 计算输出层误差
        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = targets[k] - self.activation_output[k]  # 计算GT和目前激活值的误差
            output_deltas[k] = sigmoid_derivative(
                self.activation_output[k]) * error

        # 计算隐藏层的误差
        hidden_deltas = [0.0] * self.hidden
        for j in range(self.hidden):
            error1 = 0.0
            for k in range(self.output):
                error1 = error1 + output_deltas[k] * self.weight_output[j][k]
            hidden_deltas[j] = sigmoid_derivative(
                self.activation_hidden[j]) * error1

        # 更新输出层权重
        for j in range(self.hidden):
            for k in range(self.output):
                new_w = output_deltas[k] * self.activation_hidden[j]
                self.weight_output[j][k] = self.weight_output[j][k] + lr * new_w

        # 更新输入层权重
        for i in range(self.input):
            for j in range(self.hidden):
                new_w1 = hidden_deltas[j] * self.activation_input[i]
                self.weight_input[i][j] = self.weight_input[i][j] + lr * new_w1

        # 计算误差方便输出

        error2 = 0.0
        # GT和输出的差距
        error2 = error2 + 0.5 * \
            (targets[k]-self.activation_output[k]) ** 2  # 计算平方误差
        return error2

    def test(self, Labels):
        count = 0
        for p in Labels:
            GT = flowerLabels[(p[1].index(1))]
            result = self.update(p[0])

            # 最大的结果即预测结果
            index = result.index(max(result))
            print(p[0], ":", GT, "->", flowerLabels[index])

            # 预测类别和GT相同时，正确计数器增加
            count = count + (GT == flowerLabels[index])
        # 计算测试准确率
        accuracy = float(count/len(Labels))
        print("accuracy: %-.9f" % accuracy)

    def weights(self):
        print("input weights:")
        for i in range(self.input):
            print(self.weight_input[i])
        print()
        print("output weights:")
        for j in range(self.hidden):    # 注意权重建立是按hidden建立的行
            print(self.weight_output[j])

    def train(self, Labels, iterations=1000, lr=0.1):
        for i in range(iterations):
            error = 0.0
            for p in Labels:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, lr)
            if i % 100 == 0:
                print("loss: %-.9f" % error)


def iris():
    data = []
    raw = pd.read_csv("D:\\Pythonplace\\bp-iris\\iris.csv")
    raw_data = raw.values
    raw_feature = raw_data[0:, 0:4]

    # 最后一列进行one-hot编码
    for i in range(len(raw_feature)):
        ele = []
        ele.append(list(raw_feature[i]))

        if raw_data[i][4] == 'setosa':
            print("000")
            ele.append([1, 0, 0])
        elif raw_data[i][4] == 'versicolor':
            print("111")
            ele.append([0, 1, 0])
        else:
            print("222")
            ele.append([0, 0, 1])
        data.append(ele)
    # 乱序
    random.shuffle(data)
    # 前100做训练
    trainset = data[0:100]
    # 后面做测试
    testset = data[101:]
    # 定义神经网络节点数，隐层数和类别数
    nn = BPNN(4, 7, 3)
    # 开始训练
    nn.train(trainset, iterations=10000)
    # 测试
    nn.test(testset)


iris()
