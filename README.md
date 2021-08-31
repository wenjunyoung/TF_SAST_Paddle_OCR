## TensorFlow SAST 

The repo is the tensorflow version of Paddle-OCR. At present, it only includes the function of text detection. Other functions such as text recognition will come soon.

### 快速安装

- Tensorflow==2.5
- Python==3.7

```
# 创建虚拟环境
conda create -n tf_pp_ocr python=3.7
conda activate tf_pp_ocr

# 安装Tensorflow
pip install --upgrade pip
pip install tensorflow==2.5.0

# 克隆项目
git clone https://github.com/wenjunyoung/TF_SAST_Paddle_OCR.git

# 安装必要的工具包
cd tf_paddle_ocr
pip install -r requirements.txt
```

注意，windows环境下，建议从[这里](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)下载shapely安装包完成安装，
直接通过pip安装的shapely库可能出现`[winRrror 126] 找不到指定模块的问题`。

### 注意！！！

使用本项目仓库之前，必须将Tensorflow的image_data_format默认通道顺序改为"channels_first"，请遵循以下操作：

```
# 查看通道顺顺序
$ cat ~/.keras/keras.json

默认情况如下：
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_last"
}

# 将channels顺序改为first
$ nano ~/.keras/keras.json

{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_first"
}
将内容修改后按 ctrl+o -> Enter键 -> ctrl+x退出保存

```

### 快速启动训练

**开始训练**

```
python run.py -c config/det/det_r50_vd_sast_tf_icdar15.yml 
```

**从checkpoint恢复训练**

如果训练程序中断，如果希望加载训练中断的模型从而恢复训练，可以通过指定Global.checkpoints指定要加载的模型路径：

```
python run.py -c config/det/det_r50_vd_sast_tf_icdar15.yml -o Global.checkpoints=./your/trained/model
```

### 测试

在图像上测试检测结果

For example:

```
python tools/infer_det.py -c config/det/det_r50_vd_sast_tf_icdar15.yml -o Global.infer_img="./your/image.jpg" Global.checkpoints=/your/checkpoints/

# 
python tools/infer_det.py -c config/det/det_r50_vd_sast_tf_icdar15.yml -o Global.infer_img=./assets/doc/imgs_en/img_10.jpg Global.checkpoints=./output/sast_r50_vd_ic15_tf/iter_epoch_1055/
```

### 评估

-c：配置文件          checkpoints：检查点文件

```
python tools/eval.py -c config/det/det_r50_vd_sast_tf_icdar15.yml -o Global.checkpoints= /your/checkpoint/
```

### 部署

使用OpenVINO部署于Intel CPU

待更新！

