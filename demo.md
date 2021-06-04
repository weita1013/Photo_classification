```python
# 数据预处理
import os
import json
import shutil

name_dict = {'people': 0, 'animal': 1, 'landscape': 2,
             'vehicle': 3, 'food': 4}
data_root_path = r'dataset/photo/'
# 测试集路径
test_file_path = data_root_path + 'test.list'
# 训练集文件路径
train_file_path = data_root_path + 'train.list'
# 样本汇总文件
readme_file = data_root_path + 'readme.json'
# 记录每个类别多少张训练图片、测试图片
name_data_list = {}


def save_train_test_file(path, name):
    if name not in name_data_list:
        img_list = []
        img_list.append(path)
        name_data_list[name] = img_list
    else:
        name_data_list[name].append(path)


# 遍历目录、将图片路径存入字典，再由字典写入文件
dirs = os.listdir(data_root_path)
for d in dirs:
    full_path = data_root_path + d
    if os.path.isdir(full_path):
        imgs = os.listdir(full_path)
        for img in imgs:
            save_train_test_file(full_path + '/' + img, d)
    else:
        pass
# 分测试集和训练集
with open(test_file_path, 'w')as f:
    pass
with open(train_file_path, 'w')as f:
    pass
# 遍历字段，分配测试集
for name, img_list in name_data_list.items():
    i = 0
    num = len(img_list)
    print('{}:{}张'.format(name, num))
    for img in img_list:
        if i % 10 == 0:
            with open(test_file_path, 'a')as f:
                line = '%s\t%d\n' % (img, name_dict[name])
                f.write(line)
        else:
            with open(train_file_path, 'a')as f:
                line = '%s\t%d\n' % (img, name_dict[name])
                f.write(line)
        i += 1

# 网络搭建、模型训练/保存
import paddle
import os
import paddle.fluid as fluid
import numpy
import sys
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

paddle.enable_static()


def train_mapper(sample):
    img, label = sample
    if not os.path.exists(img):
        print('图片不存在', img)
    else:
        # 读取图片
        img = paddle.dataset.image.load_image(img)
        # 对图片进行变换，修剪，输出矩阵
        img = paddle.dataset.image.simple_transform(im=img,
                                                    resize_size=100,
                                                    crop_size=100,
                                                    is_train=True)
        # 图像归一化处理，将值压缩到0～1之间
        img = img.flatten().astype('float32') / 255.0
        return img, label


# 自定义reader，从训练集读取数据，并交给train_mapper处理
def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, 'r')as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(train_mapper,
                                      reader,
                                      cpu_count(),
                                      buffered_size)


# 搭建神经网络
# 输入层、卷积池化层dropout*3、全连接层、dropout、全连接层
def convolution_nural_network(image, type_size):
    # 第一个卷积-池化层
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=image,  # 输入数据
                                                  filter_size=3,  # 卷积核大小
                                                  num_filters=32,  # 卷积核数量，与输出通道数相同
                                                  pool_size=2,  # 池化层大小2*2
                                                  pool_stride=2,  # 池化层步长
                                                  act='relu')  # 激活函数)
    # dropout 丢弃学习，随机丢弃一些神经元的输出，防止过拟合
    drop = fluid.layers.dropout(x=conv_pool_1,  # 输出
                                dropout_prob=0.5)  # 丢弃率
    # 第二个卷积-池化层
    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=drop,  # 输入数据
                                                  filter_size=3,  # 卷积核大小
                                                  num_filters=64,  # 卷积核数量，与输出通道数相同
                                                  pool_size=2,  # 池化层大小2*2
                                                  pool_stride=2,  # 池化层步长
                                                  act='relu')  # 激活函数)
    drop = fluid.layers.dropout(x=conv_pool_2,  # 输出
                                dropout_prob=0.5)  # 丢弃率
    # 第三个卷积-池化层
    conv_pool_3 = fluid.nets.simple_img_conv_pool(input=drop,  # 输入数据
                                                  filter_size=3,  # 卷积核大小
                                                  num_filters=64,  # 卷积核数量，与输出通道数相同
                                                  pool_size=2,  # 池化层大小2*2
                                                  pool_stride=2,  # 池化层步长
                                                  act='relu')  # 激活函数)
    drop = fluid.layers.dropout(x=conv_pool_3,  # 输出
                                dropout_prob=0.5)  # 丢弃率
    # 全连接层
    fc = fluid.layers.fc(input=drop,
                         size=512,
                         act='relu')
    drop = fluid.layers.dropout(x=fc, dropout_prob=0.5)
    # 输出
    predict = fluid.layers.fc(input=drop,  # 输出层
                              size=type_size,  # 最终分类个数
                              act='softmax')  # 激活函数
    return predict


# 准备数据执行训练
BATCH_SIZE = 32
trainer_reader = train_r(train_list=train_file_path)
train_reader = paddle.batch(paddle.reader.shuffle(reader=trainer_reader,
                                                  buf_size=1200),
                            batch_size=BATCH_SIZE)
# 训练时输入数据
image = fluid.layers.data(name='image',
                          shape=[3, 100, 100],  # RGB三通道彩色图像
                          dtype='float32')
# 训练时期望输出值/真实类别
label = fluid.layers.data(name='label',
                          shape=[1],
                          dtype='int64')
# 调用函数，创建卷积神经网络
predict = convolution_nural_network(image=image,  # 输入数据
                                    type_size=5)  # 类别数量
cost = fluid.layers.cross_entropy(input=predict,  # 预测值
                                  label=label)  # 期望值
avg_cost = fluid.layers.mean(cost)  # 求损失值的平均值
# 计算预测准确率
accuracy = fluid.layers.accuracy(input=predict,  # 预测值
                                 label=label)  # 期望值
# 优化器，自适应提督下降
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)
# 执行器
place = fluid.CPUPlace()  # 在cpu上执行
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())  # 初始化系统参数
feeder = fluid.DataFeeder(feed_list=[image, label],
                          place=place)
for pass_id in range(10):
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),  # 执行默认program
                                        feed=feeder.feed(data),  # 输入数据
                                        fetch_list=[avg_cost, accuracy]  # 获取数据
                                        )
        if batch_id % 20 == 0:
            print('pass:{}, batch:{}, cost:{}, acc:{}'.format(pass_id, batch_id, train_cost[0], train_acc[0]))
print('训练完成')

# 保存模型
model_save_dir = r'model/photo/'
shutil.rmtree(model_save_dir, ignore_errors=True)
os.makedirs(model_save_dir)
fluid.io.save_inference_model(dirname=model_save_dir,
                              feeded_var_names=['image'],
                              target_vars=[predict],
                              executor=exe)
print('模型保存完成')

# 模型加载，结果预测
from PIL import Image

place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)


# 读取图片，调整尺寸归一化处理
def load_image(path):
    img = paddle.dataset.image.load_and_transform(
        path, 100, 100, False).astype('float32')
    img = img / 255.0
    return img


infer_imgs = []  # 图像数据列表
test_img = '4.jpg'  # 预测图像路径
infer_imgs.append(load_image(test_img))  # 加载图像数据，添加到列表
infer_imgs = numpy.array(infer_imgs)  # 转换成array

# 加载模型
[infer_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_save_dir, infer_exe)
# 现显示原始图片
img = Image.open(test_img)  # 打开图片
plt.imshow(img)  # 显示原始图片
plt.show()
# 执行预测
results = infer_exe.run(infer_program,
                        feed={feed_target_names[0]: infer_imgs},
                        fetch_list=fetch_targets)
print(results)  # result为数组，包含每个类别的概率
result = numpy.argmax(results[0])  # 获取最大值的索引
for k, v in name_dict.items():
    if result == v:
        print('预测结果为：', k)

```

