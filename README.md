# 行人检测+跟踪+行为识别

* 版本：0.1
* 日期：2019-11-26
* 作者：常鑫
* 框架：PyTorch 1.0+
* 说明：
本代码包含一个基于mmdetection的检测器、基于IoU-Tracker/Tracktor的两个可选行人多目标跟踪器与一个基于mmaction的行为识别网络（代码有部分修改），可读取视频并实时返回检测+跟踪+行为识别结果。
检测器可使用任何基于mmdetection的、含行人类别的检测器，无需额外训练；
IoU-Tracker和Tracktor不涉及检测器以外的训练。
* 测试指标：（行为识别）

    | Mean Class Acc |  Top-1 Acc |
    |----------------|------------|
    | 96.48%         |  96.34%    |
    
    （以上数据为latest.pth基于训练前划分出的测试集上的评估值，若重新划分子集则可能得到不同指标值，但差距不会太大）

\--------------------------------------------------------

2020.02.XX 更新

1. 修改了标注文件的格式
2. aciton_recognition.py更名为main.py，增加仅运行跟踪算法获取中间结果的功能
3. 添加了数据标注相关的文档

\--------------------------------------------------------

2019.12.17 更新

上传了几个测试用的视频和TSN模型权重文件。

\--------------------------------------------------------

## 运行环境
* python 3.7
* pytorch 1.3
* torchvision 0.4.0
* pillow 6.1.0
* opencv-python 4.1.1
* numpy 1.16.4
* matplotlib 3.1.0
* scikit-learn 0.21.2
* tqdm 4.32.1
* argparse 1.1
* mmdetection v1.0rc0
* mmaction v0.1.0

## 安装
进入`third_party`目录，分别完成mmaction和mmdetection的安装：
```
cd third_party

cd mmaction
./compile.sh
python setup.py develop

cd ..
cd mmdetection
python setup.py develop
```

## Demo
运行demo之前，请先准备mmdetection二阶段检测器(如Faster R-CNN)的权重文件，并训练行为识别器。
Cascade R-CNN在mmdetection代码实现上不属于二阶段检测器，因此这里无法使用。
```
python main.py action \
 --detector_config third_party/mmdetection/configs/faster_rcnn_r50_fpn_1x.py \
 --detector_checkpoint third_party/mmdetection/modelzoo/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth \
 --recognizer_config third_party/mmaction/configs/climbing/tsn_rgb_bninception.py \
 --recognizer_checkpoint third_party/mmaction/work_dirs/climbing_tsn_2d_rgb_bninception_seg_3_f1s1_b32_g8/epoch_80.pth \
 --tracker tracktor \
 --video_path /PATH/TO/VIDEO \
 --save_video out.mp4
```
参数说明：
- `--detector_config CONFIG`：mmdetection检测器的配置文件路径
- `--detector_checkpoint CHECKPOINT`：mmdetection检测器的权重文件路径
- `--recognizer_config CONFIG`：mmaction检测器的配置文件路径
- `--recognizer_checkpoint CHECKPOINT`：
- `--tracker TRACKER`：使用的跟踪器，可选"tracktor"或"ioutracker"，前者效果更好，后者更快
- `--video_path VIDEO_PATH`：用于测试的视频
- `--save_video SAVE_VIDEO`：若非空，则会将demo输出的视频保存

`test_samples`文件夹中包含几个较短的用于测试的视频用例，可作为`--video_path`参数的值。

## 训练（行为识别）

如准备自行准备视频并标注，请参阅[PREPARE_DATA.md](docs/PREPARE_DATA.md)。
如使用已收集的攀爬视频和现有的标注，请使用以下命令生成用于行为识别训练的格式的数据。

### 准备数据
```
# 1. 进入mmaction攀爬数据处理脚本目录
cd third_party/mmaction/data_tools/climbing

# 2. 下载视频（需安装youtube-dl，自备梯子）
bash download_videos.sh

# 3. 提取有效帧
bash extract_original_frames.sh

# 4. 剪切视频
bash crop_video_patches.sh

# 5. 随机划分子集
bash divide_splits.sh

# 6. 生成剪切帧
bash extract_frames.sh

# 7. 生成文件列表
bash generate_filelist.sh
```
   

### 开始训练
（在third_party/mmaction目录下）
```
python tools/train_recognizer.py configs/climbing/tsn_rgb_bninception.py
```
程序唯一的必选参数是训练使用的配置文件。模型选取、数据配置都在这一个文件中。

可选参数：
- `--work_dir WORK_DIR`：工作路径，用于存放日志和训好的模型。
- `--resume_from CHECKPOINT_PATH`：之前训练的权重文件，可用于迁移学习
- `--validate`：是否在训练过程中进行验证
- `--gpus GPUS`：整数，使用的GPU个数，默认为1
- `--seed SEED`：整数，自行设置的随机种子，默认为None
- `--launcher LAUNCHER`：分布式训练的启动器，可选"pytorch""slurm""mpi""none"，默认为"none"（非分布式训练）
- `--local_rank RANK`：貌似没用
   
## 评估（行为识别）
（在third_party/mmaction目录下）
```
python tools/test_recognizer.py configs/climbing/tsn_rgb_bninception.py work_dirs/climbing_tsn_2d_rgb_bninception_seg_3_f1s1_b32_g8/epoch_80.pth
```
程序的第一个参数是上述配置文件，第二个参数是训练好的权重文件。
按照配置文件中的设定，训练完成后的权重文件路径即`work_dirs/climbing_tsn_2d_rgb_bninception_seg_3_f1s1_b32_g8/epoch_80.pth`。

可选参数：
- `--gpus GPUS`：使用的GPU个数，默认为1
- `--proc_per_gpu PROC_PER_GPU`：每个GPU的进程数
- `--out OUT`：若非空，则为输出文件路径（扩展名须为.pkl或.pickle）
- `--use_softmax`：预测结果是否经过softmax层
