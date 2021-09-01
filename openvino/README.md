### 使用OpenVINO 部署模型

1. 下载安装OpenVINO，安装及验证是否安装成功，请遵循官方教程的指导

2. 将 TensorFlow版本的 SAST 模型保存，并遵循OpenVINO官方教程将模型转换为 IR文件。若成功即可得到 相应的 .xml 以及.bin文件

3. 本代码运行之前，必须激活OpenVINO

4. 在命令行窗口输入一下 命令：

   Tips：电脑记得连接摄像头~

   ```
   python tf_openvino.py --xml /path/to/model/.xml  --bin /path/to/model/.bin
   ```

   注意：本代码还没经过debug，可能存在一些bug！！！

   

   