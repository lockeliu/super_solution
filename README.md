### Dependencies

* cv2
*  time
*  math
* numpy
* random
* argparse
*  Python 3.6.4
* multiprocessing
*  PyTorch >= 0.4.1

### TRAIN DATA
* [DIV2K](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)
### TEST DATA

* [Set5](https://link.zhihu.com/?target=http%3A//people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)
* [Set14](https://link.zhihu.com/?target=https%3A//sites.google.com/site/romanzeyde/research-interests)
* [Urban 100](https://link.zhihu.com/?target=https%3A//sites.google.com/site/jbhuang0604/publications/struct_sr)
* [BSD 100](https://link.zhihu.com/?target=https%3A//www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
* [DIV2K](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip)


**PS：Set5，Set14，Urban 100，BSD 100 在 [dataset/Eval](https://github.com/lockeliu/super_solution/tree/master/dataset/Eval)**


### TRAIN
* 下载DIV2K数据，或者自己的高清图片数据，都直接把图片放一个目录下
* 执行tool/prepare_data.py，预处理图片数据到别的目录
```
python tool/prepare_data.py -d DIV2K -o DIV2K_OUT -n 10 

-d 高清图片数据
-o 预处理后的图片数据
-n 进程数
```
* 执行main.py训练
```
python main.py -m edsr -t 'DIV2K_OUT' -v 'dataset/Eval/Set14'  -s 1 -g 1 -p weight/edsr_1.pt -r 1

-m 模型类型 edsr，mdsr
-t 预处理后的图片目录
-v 评估数据的目录
-s 训练的模型放大类型
-b 训练的batch size
-i 数据输入的长高
-r 训练数据重复几次
-e epoch
-l lr
-g gpu个数
-p 训练模型位置
-dp gan的判别模型位置
-is_gan 是否用上gan
```

### PRED
* 训练完后，执行test.py，预测图片
```
python test.py  -m edsr -S 1 -s 1 -g 2 -p weight/edsr_1.pt  -i head.jpg -o head.png

-m 模型类型 edsr，mdsr
-S 放大倍数
-s 训练的模型放大类型
-g gpu_id
-p 模型路径
-i 图片位置
-o 放大图片位置
```
