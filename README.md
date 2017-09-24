# DeepLearningFlappyBird
(Add a choice to play and train the NN by yourself)
modified from https://github.com/yenchenlin/DeepLearningFlappyBird

English：Modify:
1.Add a choice to play and train the NN by yourself.
2.Add a event to close the program window by click.
3.Add a Half-pre-trained NN(560000timesteps), which was trained after i played it myself for 3000 timesteps .(OBSERVE:3000, REPLAY_MEMORY:4000)
4.Add some Chinese annotation.

How to use:
1.python deep_q_network.py
2.After the terminal print the menu, enter 1 or 2 and press enter to choose.
(1:restore a Pre-trained NN  2:play yourself to train the NN).

How to choose different Pre-trained NN:
Modify the first line of 'checkpoint' file in the 'saved_networks' folder.

Tips:
1.Please change the value of REPLAY_MEMORY and OBSERVE by modify 'deep_q_network.py' when you use a Pre-trained NN, usually large values.(REPLAY_MEMORY :50000, OBSERVE:100000)
2.You can set them to about (REPLAY_MEMORY :5000, OBSERVE:4000) when you play and train the NN yourself, or you will feel tired.

中文：
修改部分：
1.程序开始运行时加了个选择,可以选择自己玩来训练网络或载入一个预训练的网络。
2.可以点击鼠标来关闭程序窗口。
3.加了一个我自己玩了3000步后之后，训练了一半的网络（560000步）。(OBSERVE:3000, REPLAY_MEMORY:4000)
4.加了中文注释，看起来更顺眼。

如何使用：
1.python deep_q_network.py
2.在终端显示菜单后，输入1或2，之后回车。

如何选择载入哪个网络：
修改saved_networks文件夹里面的checkpoint文件

小建议：
1.当你使用一个预训练的网络时，请修改deep_q_network.py中REPLAY_MEMORY和OBSERVE的值。(REPLAY_MEMORY :50000, OBSERVE:100000)
2.自己玩来训练可以小一些，不然会很累。(REPLAY_MEMORY :5000, OBSERVE:4000)
