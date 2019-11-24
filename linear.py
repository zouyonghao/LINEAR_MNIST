import numpy as np
import random
import math
import matplotlib.pyplot as plt


class LinearClassification:
    def __init__(self):
        # 学习率的初始值
        self.learn_speed_start = 0.1
        # 学习率
        self.learn_speed = 0.0
        # 偏置
        self.b = 1
        # 最小误差精度
        self.min_error_signal = 0.05
        # 衰减因子
        self.r = 5.0
        # 训练次数
        self.train_count = 100

        self.train_data = []
        self.data_classes = []
        self.weight = [self.b]

    def min_error_signal_init(self, min_error_signal):
        self.min_error_signal = min_error_signal

    def add_train_data(self, data, data_class):
        for item in data:
            # 训练数据
            self.train_data.append([1] + item)
            # 训练数据的分类
            self.data_classes.append(data_class)

    def learn_speed_start_init(self, init_learn_speed_start):
        self.learn_speed_start = init_learn_speed_start

    def r_init(self, init_r):
        self.r = init_r

    def train_count_init(self, init_train_count):
        self.train_count = init_train_count

    def sgn(self, v):
        if v > 0:
            return 1
        else:
            return -1

    def get_sgn(self, current_weight, current_train_data):
        # print (current_train_data.shape)
        # print (current_weight.shape)
        return self.sgn(np.dot(current_weight.T, current_train_data))

    def get_error_signal(self, current_weight, current_train_data, current_class):
        return current_class - self.get_sgn(current_weight, current_train_data)

    def update_weight(self, old_weight, current_train_data, current_class, current_learn_speed, current_train_count):
        # error
        current_error_signal = self.get_error_signal(
            old_weight, current_train_data, current_class)
        # update learn speed
        self.learn_speed = self.learn_speed_start / \
            (1 + (current_train_count / float(self.r)))
        # self.learn_speed = 0.1
        new_weight = old_weight + \
            (current_learn_speed * current_error_signal * current_train_data)
        return new_weight

    def train(self):

        self.train_data = np.array(self.train_data)
        self.data_classes = np.array(self.data_classes)
        # 初始化权值
        for i in range(len(self.train_data[0]) - 1):
            self.weight.append(0.0)
        self.weight = np.array(self.weight)
        current_count = 0
        while True:
            error_signal = 0
            i = 0
            # for j in self.data_classes:
            #     # print(j)
            #     current_error_signal = self.get_error_signal(
            #         self.weight, self.train_data[i], j)
            #     self.weight = self.update_weight(
            #         self.weight, self.train_data[i], j, self.learn_speed, current_count)
            #     i += 1
            #     error_signal += math.pow(current_error_signal, 2)

            # batch
            for j in range(0, 3000):
                # print(j)
                i = random.randint(0, len(self.train_data) - 1)
                current_error_signal = self.get_error_signal(
                    self.weight, self.train_data[i], self.data_classes[i])
                self.weight = self.update_weight(
                    self.weight, self.train_data[i], self.data_classes[i], self.learn_speed, current_count)

            # i = 0
            for j in range(0, 5000):
            # for j in range(0, len(self.train_data) - 1):
                # print(j)
                current_error_signal = self.get_error_signal(
                    self.weight, self.train_data[j], self.data_classes[j])
                # i += 1
                error_signal += math.pow(current_error_signal, 2)
            # error_signal = error_signal / len(self.train_data)
            error_signal = error_signal / 5000
            current_count += 1

            # print(" count = ", current_count,
            #       " weight = ", self.weight,
            #       " error = ", error_signal)
            print(" count = ", current_count,
                  " error = ", error_signal)

            if abs(error_signal) < self.min_error_signal:
                break
            if current_count > self.train_count:
                break

    def classify(self, test_data):
        if self.get_sgn(self.weight, np.array([1] + test_data)) > 0:
            return 1
        else:
            return -1


if __name__ == "__main__":

    train_data1 = [[1, 2, 1], [5, 8, 2], [4, 7, 3], [2, 9, 4], [6, 3, 5]]
    train_data2 = [[11, 34, 8], [12, 67, 9], [
        13, 44, 8], [77, 45, 7], [15, 54, 7]]
    linear = LinearClassification()
    # 样本初始化
    linear.add_train_data(train_data1, 1)
    linear.add_train_data(train_data2, -1)

    linear.train()

    # 验证
    simulate_result = linear.classify([23, 67, 0])
    if simulate_result == 1:
        print("1")
    else:
        print("-1")
