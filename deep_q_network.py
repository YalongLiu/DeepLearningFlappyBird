# coding:utf-8
# !/usr/bin/env python
from __future__ import print_function

import pygame
from pygame.locals import *

import tensorflow as tf
import cv2
import sys

sys.path.append("game/")
import wrapped_flappy_bird as game  # 这里wrapped_flappy_bird报错不用管
import random
import numpy as np
from collections import deque

GAME = 'bird'  # 游戏名字-用于记录日志和保存网络
ACTIONS = 2  # 动作种数
GAMMA = 0.99  # 经验依赖系数
OBSERVE = 3000.  # 训练网络前的观察步数
EXPLORE = 2000000.  # epsilon（贪心度）退火系数
FINAL_EPSILON = 0.0001  # epsilon最终值
INITIAL_EPSILON = 0.0001  # epsilon初始值
REPLAY_MEMORY = 50000  # 记忆容量
BATCH = 32  # minibatch的大小
FRAME_PER_ACTION = 1  # 几帧输出一个动作


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)  # 标准差为0.01的截断正态分布
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)  # 常数
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
                        padding="SAME")  # [batch, height, width, channels]，SAME：长度除以步长向上取整


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def createNetwork():
    # 网络权值
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # 输入层
    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])  # [batch, height, width, channels]

    # 隐层
    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])  # -1-->none  网络扁平化

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)  # 全连接层

    # 输出层
    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1  # 输入图像，网络输出，全连接层输出


def trainNetwork(s, readout, h_fc1, sess):
    # 定义代价函数
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])  # 初始化动作占位符
    y = tf.placeholder("float", [None])  # 初始化网络直接输出占位符
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)  # 网络输出值
    cost = tf.reduce_mean(tf.square(y - readout_action))  # 均方差
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)  # 选择优化器

    # 打开游戏交互界面
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # 存储观察数据
    # store the previous observations in replay memory
    D = deque()

    # 输出信息到文件（可以注释）
    # printing
    # a_file = open("logs_" + GAME + "/readout.txt", 'w')
    # h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # 第一个状态执行空动作，预处理图像到80x80x4大小
    # get the first state by doing nothing and pre-process the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # 堆栈，按第2维展开

    # 保存与加载网络参数
    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    choose_mode = int(input("1:restore Pre-trained NN\n2:play yourself\n"))
    if choose_mode == 1:  # 载入预训练的神经网络 # restore Pre-trained NN
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)  # 载入训练好的网络数据
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
    elif choose_mode != 2:  # 只能选择1或2
        choose_mode = 2
    # 开始训练
    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    action_index = 0
    pygame.mouse.set_visible(1)  # make cursor invisible
    while "flappy bird" != "angry bird":
        a_t = [1, 0]  # 每个回合复位一下动作为[1,0]-空动作
        for event in pygame.event.get():
            if event.type == QUIT:  # 点击关闭窗口则退出程序
                pygame.quit()
                sys.exit()
            # 选择的模式二、鼠标按下、左键
            elif choose_mode == '2' and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                a_t = [0, 1]  # 点击一次小鸟飞一次
        if choose_mode == 2:
            # choose an action epsilon greedily
            readout_t = readout.eval(feed_dict={s: [s_t]})[0]  # 网络名称.eval   [0]取第一行元素 a_t = np.zeros([ACTIONS])
            action_index = 0
            # 根据epsilon（贪心度）的大小选择一个动作
            if t % FRAME_PER_ACTION == 0:  # 每隔几帧动作一次
                a_t = [0, 0]
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS)  # action_index随机生成0或1
                    a_t[random.randrange(ACTIONS)] = 1
                else:
                    action_index = np.argmax(readout_t)  # 最大值的下标
                    a_t[action_index] = 1
            else:
                a_t[0] = 1  # do nothing # [1,0]-空动作

        # epsilon退火
        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 执行选择的动作并观察下个状态与奖励
        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)  # 原始画面、奖励、游戏终止标记、动作
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)  # 图片颜色空间变换BGR2GRAY
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)  # 二值化
        x_t1 = np.reshape(x_t1, (80, 80, 1))  # 重定义图像大小
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)  # 新的图像拼接在旧图像的上面

        # 在D中存储训练用的数据
        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))  # 旧图像[80,80,4]，动作，奖励，新图像[80,80,4]，终止标记
        if len(D) > REPLAY_MEMORY:
            D.popleft()  # 每一回合丢掉旧数据

        # 观察一段时间（t>OBSERVE）后再训练网络
        # only train if done observing
        if t > OBSERVE:
            choose_mode = ''
            # 在D中随机选取BATCH数目的数据
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # 提取这个BATCH里面的数据
            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]  # 旧图像
            a_batch = [d[1] for d in minibatch]  # 动作
            r_batch = [d[2] for d in minibatch]  # 奖励
            s_j1_batch = [d[3] for d in minibatch]  # 新图像

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})  # 新图像对应的网络输出
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # 如果游戏终止，只返回奖励（r_batch[i]）
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # 训练网络
            # perform gradient step
            train_step.run(feed_dict={
                y: y_batch,  # 奖励
                a: a_batch,  # 动作
                s: s_j_batch}  # 图像
            )

        # 更新旧数据
        # update the old values
        s_t = s_t1
        t += 1

        # 每10000步保存一下网络
        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

        # 打印信息
        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        if choose_mode == 1:  # 如果选择了载入已训练的网络
            print("TIMESTEP", t, "/ STATE", state, \
                  "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                  "/ Q_MAX %e" % np.max(readout_t))

            # # 将信息写入文件（可以注释掉不用）
            # # write info to files
            # if t % 10000 <= 100:
            #     a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            #     h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            #     cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        else:  # 如果选择了亲自训练的网络
            print("TIMESTEP", t, "/ STATE", state, \
                  "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t)


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)
