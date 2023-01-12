from turtle import update
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

def get_loss(file_path, num=30):
    file = open(file_path)
    lines = file.readlines()
    update = 20000
    loss_list = []
    loss_valid_list = []
    for line in lines:
        if not ':' in line:
            if 'valid' in line:
                line_split = line.split(' ')
                loss_valid_list.append(float(line_split[10]))
        else:
            line_split = line.split(' ')
            loss_list.append(float(line_split[9].split('=')[1].replace(',','')))

    train_x =  np.arange(0,update,update/len(loss_list))
    train_y = loss_list
    if len(train_y) < len(train_x):
        train_x = train_x[:len(train_y)]
    else:
        train_y = train_y[:len(train_x)]

    valid_x = np.arange(0,update,update/len(loss_valid_list))
    valid_y = loss_valid_list
    if len(valid_y) < len(valid_x):
        valid_x = valid_x[:len(valid_y)]
    else:
        valid_y = valid_y[:len(valid_x)]
    file.close()

    if num == 0: 
        step_train = len(train_x)
        step_valid = len(valid_x)
    else:
        step_train = int(len(train_x)/num)
        step_valid = int(len(valid_x)/num)

    return {'train_x':train_x[::step_train], 'train_y':train_y[::step_train], 'valid_x':valid_x[::step_valid], 'valid_y':valid_y[::step_valid]}

# style.use('ggplot')

multiple_ll = get_loss('multiple_ll_loss.txt')
multiple_cr = get_loss('multiple_cr_loss.txt')
single_ll = get_loss('single_ll_loss.txt')
single_cr = get_loss('single_cr_loss.txt')

plt.plot(multiple_ll['train_x'], multiple_ll['train_y'], marker = 'o', markersize=4, linestyle="--" ,label="Multiple_LL Train Loss", )
plt.plot(multiple_cr['train_x'], multiple_cr['train_y'], marker = 'x', markersize=4, linestyle="-.",label="Multiple_CR Train Loss")
plt.plot(single_ll['train_x'], single_ll['train_y'], marker = 'o', markersize=4, linestyle="--" ,label="Single_LL Train Loss")
plt.plot(single_cr['train_x'], single_cr['train_y'], marker = 'x', markersize=4, linestyle="-.",label="Single_CR Train Loss")


# multiple = get_loss('multiple_loss.txt')
# single =  get_loss('single_loss.txt')
# plt.plot(multiple['train_x'], multiple['train_y'], marker = 'o', markersize=4, linestyle="--" ,label="Multiple Train Loss", )
# plt.plot(multiple['valid_x'], multiple['valid_y'], marker = 'o', markersize=4, linestyle="-." ,label="Multiple Valid Loss", )
# plt.plot(single['train_x'], single['train_y'], marker = 'x', markersize=4, linestyle="--" ,label="Single Train Loss", )
# plt.plot(single['valid_x'], single['valid_y'], marker = 'x', markersize=4, linestyle="-." ,label="Single Valid Loss", )

plt.ylim(3.5, 4.0)
plt.legend(loc="best")
plt.xlabel('更新次数')
plt.ylabel('损失')
plt.show()