# MCNN人群计数DEMO

## 下载

使用 `git clone` 将仓库克隆至本地

## 环境准备

```bash
Python version >= 3.6
cv2
pytorch >=0.4.1
matplotlib.pyplot
numpy
```

## 使用

```bash
cd ...\MCNN_Demo
python main.py [path] commands
```

其中 `[path]` 为源图像目录或指定图像文件  
`commands` 为程序可接受的指令,均为可选指令  
|  |  |
| --- | --- |
| `-o [path]` | 输出结果到指定目录 |
| `-m [path]` | 使用指定的参数文件(由`torch.save(net.state_dict(),'filename')`建立) |
| `-s` | 在运行时展示中间结果 |

| Command  | Usage |
| ------------- | ------------- |
| `-o [path]`  | 输出结果到指定目录  |
| `-m [path]`  | 使用指定的参数文件(由`torch.save(net.state_dict(),'filename')`建立)  |
| `-s`  | 在运行时展示中间结果  |
