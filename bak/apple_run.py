from appleClassification import ICLearner
import time

current_time = time.strftime("%Y%m%d-%H%M%S")
print('strat!', current_time)
'''数据集扩增到三倍，含有切割、翻转、噪声'''
Ic = ICLearner(scheduler_step=12, classes=4, batch_size=6, lr=0.01, ic_model='',
               gamma=0.1, path='data')

Ic.new_train(20)
