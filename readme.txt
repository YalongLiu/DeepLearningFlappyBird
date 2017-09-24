# DeepLearningFlappyBird
DeepLearningFlappyBird(Add a choice to play and train the NN by yourself)
modified from https://github.com/yenchenlin/DeepLearningFlappyBird

English��
Modify:
1.Add a choice to play and train the NN by yourself.
2.Add a event to close the program window by click.
3.Add a Half-pre-trained NN(560000timesteps), which was trained after i played it myself for 3000 timesteps .(OBSERVE:3000, REPLAY_MEMORY:4000)
4.Add some Chinese annotation.

How to use:
1.python deep_q_network.py
2.After the terminal print the menu, enter 1 or 2 and press enter to choose.
(1:restore a Pre-trained NN    2:play yourself to train the NN).

How to choose different Pre-trained NN:
Modify the first line of 'checkpoint' file in the 'saved_networks' folder.

Tips:
1.Please change the value of REPLAY_MEMORY and OBSERVE by modify 'deep_q_network.py' when you use a Pre-trained NN, usually large values.(REPLAY_MEMORY :50000, OBSERVE:100000)
2.You can set them to about (REPLAY_MEMORY :5000, OBSERVE:4000) when you play and train the NN yourself, or  you will feel tired.

���ģ�
�޸Ĳ��֣�
1.����ʼ����ʱ���˸�ѡ��,����ѡ���Լ�����ѵ�����������һ��Ԥѵ�������硣
2.���Ե��������رճ��򴰿ڡ�
3.����һ�����Լ�����3000����֮��ѵ����һ������磨560000������(OBSERVE:3000, REPLAY_MEMORY:4000)
4.��������ע�ͣ���������˳�ۡ�

���ʹ�ã�
1.python deep_q_network.py
2.���ն���ʾ�˵�������1��2��֮��س���

���ѡ�������ĸ����磺
�޸�saved_networks�ļ��������checkpoint�ļ�

С���飺
1.����ʹ��һ��Ԥѵ��������ʱ�����޸�deep_q_network.py��REPLAY_MEMORY��OBSERVE��ֵ��(REPLAY_MEMORY :50000, OBSERVE:100000)
2.�Լ�����ѵ������СһЩ����Ȼ����ۡ�(REPLAY_MEMORY :5000, OBSERVE:4000)