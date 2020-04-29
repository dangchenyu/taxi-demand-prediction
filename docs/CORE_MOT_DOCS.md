# 多目标跟踪部分核心代码文档

本项目包含两个简单的在线多目标跟踪算法——Tracktor和IoU-Tracker（见`main.py`）。二者都是仅基于检测器而不使用其他辅助工具进行的，因此在长时间跟踪上效果并不是很好（无法处理目标被遮挡后再次出现的情况），而在较短时间内跟踪效果还是可以接受的，可以用来为后续的基于分类的行为识别进行目标视频采样。

多目标跟踪代码实现了典型的在线tracking-by-detection跟踪框架。大体流程为：
1. 行人目标检测
2. 目标状态预测与建模
3. 基于二分图的数据关联
4. 目标状态更新

以下将针对核心概念与各个步骤对应的模块，对`mot`模块的核心代码进行简要介绍。

## 目标——mot.tracklet.Tracklet
Tracklet（跟踪小片段）是记录多目标跟踪过程中每个不同个体信息的类。这些信息包含ID、位置、活跃时间、特征等等。

主要属性列表：
- `id`：整数，目标的ID，一般从1递增记录，初始化后不应再修改
- `last_detection`：`mot.detect.detect.Detection`类对象，目标最后一次被检测到的位置
- `feature`：`dict`变量，目标最新的特征，用于二分图边权重的构建，每一个键值对代表一种特征，默认的`'box'`代表检测框
- `feature_history`：目标的历史特征，是上述`feature`的list
- `ttl`：整数，目标失踪后存活的时间，会随着目标被检测到而增加，随着失踪而减少
- `time_lived`：整数，目标存活时间
- `prediction`：`mot.predict.predict.Prediction`类对象，目标位置的预测
- `detected`：布尔值，目标当前是否被是检测到的（而不是通过预测得到的）

方法列表：
- `predict()`：对自身在下一帧的位置进行预测，返回一个box
- `update()`：更新
    参数：
    + `frame_id`：整数，新帧的帧号
    + `detection`：`mot.detect.detect.Detection`类对象，传入值应为检测器检测到的结果
    + `feature`：`dict`变量，与本对象的`feature`属性包含的键应一致
- `fade()`：若目标在新的一帧未被检测到，应调用该函数使其`ttl`属性值自减，同时将`last_detection`更新为位置预测值，但`detected`属性设为`False`
- `is_confirmed()`：若目标持续被跟踪的时间超过`min_time_lived`，则返回`True`
- `is_detected()`：返回`detected`属性

## 跟踪器——mot.tracker.Tracker
Tracker类完成跟踪器在视频的每一帧进行的检测、预测、建模、匹配过程。**若要实现不同的跟踪算法，所有的方法均可重写，不必拘泥于当前实现思路**。

主要属性列表：
- `detector`：检测器，须为`mot.detect.detect.Detector`类的子类对象
- `encoders`：编码器list，列表元素须为`mot.encode.encode.Encoder`类的子类对象
- `matcher`：匹配器，须为`mot.associate.matcher.Matcher`类的子类对象
- `predictor`：预测器，须为`mot.predict.predict.Predictor`类的子类对象
- `max_ttl`：整数，每个Tracklet失踪后存活的最大时间
- `max_detection_history`：整数，每个Tracklet历史检测框的最大长度
- `max_feature_history`：整数，每个Tracklet历史特征的最大长度
- `tracklets_active`：正在跟踪的目标list，列表元素须为`mot.tracklet.Tracklet`类对象
- `tracklets_finished`：已离开的目标list，列表元素须为`mot.tracklet.Tracklet`类对象
- `frame_num`：帧号，每帧处理跟踪目标之前应自增，第一帧的帧号为1

方法列表：
- `clear()`：状态清零，可用于处理完一段视频后切换另一段
- `tick()`：在线跟踪中，每一帧的处理流程
    参数：
    + `img`：`numpy.ndarray`类对象，视频的新帧
- `encode()`：对检测器的检测结果进行编码
    参数：
    + `detections`：`mot.detect.detect.Detection`类对象的list，检测器的检测结果
    + `img`：`numpy.ndarray`类对象，视频的新帧
    返回值：
    + `features`：由`dict`元素组成的`list`（目前键值至少包括`'box'`），即保存每个检测框的特征
- `predict()`：对正在跟踪的目标的状态进行预测
- `update()`：更新跟踪目标的状态
    参数：
    + `row_ind`：整数list，匹配到的目标在`tracklets_active`中的下标
    + `col_ind`：整数list，匹配到的检测框在`detections`中的下标（与`row_ind`各项一一对应）
    + `detections`：`mot.detect.detect.Detection`类对象的list，检测到的所有目标框
    + `detection_features`：任何类型数据的list，检测框的特征（与`detections`各项一一对应）
- `terminate()`：终止跟踪，`tracklets_active`的目标将被移入`tracklets_finished`
- `assignment_matrix()`：通过相似度矩阵，计算匹配矩阵
    参数：
    + `similarity_matrix`：2维的`numpy.ndarray`类对象，每一项`s(i,j)`表示`tracklets_active`中第i项与`detections`第j项的相似度
- `add_tracklet()`：将新目标添加到跟踪中的目标列表中
    参数：
    + `tracklet`：`mot.tracklet.Tracklet`类对象，新创建的目标
- `kill_tracklet()`：将离开的目标移入`tracklets_finished`中
    参数：
    + `tracklet`：要移除的目标，须保证它在`tracklets_active`中

## 目标检测——mot.detect
主要包含`mot.detect.detect.Detector`类（实际是一个接口）和`mot.detect.detect.Detection`类。要实现检测器，须继承`mot.detect.detect.Detector`类。可参考`mot.detect.mmdetection.MMDetector`的示例。

### mot.detect.detect.Detector
实际上是一个接口，需要继承并实现`__init__`函数和`__call__`函数。

方法列表：
- 构造函数
- `__call__()`：调用检测器，返回检测结果
    参数：
    + `img`：`numpy.ndarray`类对象，用于检测目标的新一帧的图像
    返回值：建议返回`mot.detect.detect.Detection`类对象

**以本项目用到的mmdetection检测器为例**：实现`MMDetector`类，继承`mot.detect.detect.Detection`类，在`__call__()`函数中直接调用mmdetection的`inference_detector()`函数，将返回值封装成`mot.detect.detect.Detection`类对象。并返回所有对象的`list`。

### mot.detect.detect.Detection
用于保存检测框的实体对象。

属性列表：
- `box`：含4个元素的`list`或`numpy.ndarray`类对象，xywh格式的检测框
- `score`：单个浮点值，检测框的置信度
- `mask`：暂未使用

## 状态预测——mot.predict

### mot.predict.predict.Predictor
状态预测器，可以用来预测目标的位置、速度等的变化，用于和检测框进行匹配。预测器可能存在状态的变化（如Kalman滤波器应保存mean和covariance），因此需要时应添加一些成员变量用于保存状态。一个Tracker只需一个预测器，每次对所有目标的状态进行预测。

方法列表：
- 构造函数
- `__call__()`：调用预测器，返回预测结果
    参数：
    + `tracklets`：`mot.tracklet.Tracklets`类对象的`list`
    + `img`：`numpy.ndarray`类对象，新一帧的图像
- `initiate()`：初始化预测器的状态，若无需状态则可pass
    参数：
    + `tracklets`：初始的`Tracklet`类对象的`list`
- `update()`：更新预测器的状态，若无需状态则可pass
    参数：
    + `tracklets`：用于更新的`Tracklet`类对象的`list`
- `predict()`：进行预测，返回一个元素为`mot.predict.predict.Prediction`类对象的`list`
    参数：
    + `tracklets`：需预测的`Tracklet`类对象的`list`
    + `img`：`numpy.ndarray`类对象，用于预测的新一帧的图像

**以本项目用到的Tracktor算法为例**：实现`mot.predict.predict.MMTwoStagePredictor`类，继承`Predictor`类。Tracktor预测目标新的位置仅需每个tracklet自带的框和新一帧的图像，因此无需保存任何状态。将mmdetection的`TwoStageDetector`进行一些修改，1.把第一阶段的网络backbone提取特征过程保留，而RPN产生RoI的过程删除；2.第二阶段原本以RoI和特征图作为输入，这里把RoI替换成所有tracklet的box；3.mmdetection的`BBoxTestMixin`涉及NMS步骤，这里不使用NMS，以免破坏检测（预测）框的顺序，保证预测的结果与输入的tracklet一一对应。

### mot.predict.predict.Prediction
状态预测值。如果要预测bbox以外的内容，可以任意添加新的属性。

属性列表：
- `box`：含4个元素的`list`或`numpy.ndarray`类对象，xywh格式的预测框
- `score`：单个浮点值，检测框的置信度

## 目标建模——mot.encode
用于对检测框detection进行编码（建模）。这个编码可以是最简单的目标xywh框，可以是CNN特征，可以是目标的人体姿态向量等等。

### mot.encode.encode.Encoder
编码器。每种不同的编码器应具有不同的名称（即`name`属性），编码器的名称将与`Tracklet`的`feature`字典中的键对应。

方法列表：
- 构造函数：初始化编码名称等属性
- `__call__()`：对检测框进行编码
    参数：
    + `detections`：`mot.detect.detect.Detection`类对象的`list`，所有需要编码的检测框
    + `img`：`numpy.ndarray`类对象，用于编码的新一帧的图像
    返回值：建议返回一个任何类型元素组成的`list`，每个元素与`detections`参数的各个元素一一对应

**以本项目行为识别部分所用到的图像patch剪切器为例**：实现`mot.encode.patch.ImagePatchEncoder`类，继承`Encoder`类。编码器仅完成对所有物体框的图像patch进行裁剪、resize的工作，组成`list`并返回。

## 二分图权重指标——mot.metric
metric指跟踪目标与检测框的相似度衡量指标。更好的metric意味着目标与检测框是同一个体时相似度更高、是不同个体时相似度更低。由于遮挡、变形、光照变化等因素，仅仅使用框重合度（IoU）、距离、姿态或ReID特征等任何单一因素都不能实现完美的匹配，因此，很多算法中会使用复杂的、考虑多种不同因素的衡量指标来提升匹配的效果。

metric不直接作用于目标或检测框，而是由`Matcher`（见下文）调用。因此，请尽量保证多种`Metric`的接口一致。

### mot.metric.metric.Metric
相似度衡量指标。

方法列表：
- 构造函数
- `__call__()`：对跟踪中的目标和检测框的特征进行密集（一对一）的相似度计算
    参数：
    + `tracklets`：`mot.tracklet.Tracklet`类对象的`list`，设长度为M
    + `detection_features`：`dict`的`list`，设长度为N
    返回值：shape为(M, N)的二维列表或`numpy.ndarray`

**以Tracktor和IoUTracker使用到的IoU指标**为例：实现`mot.metric.iou.IoUMetric`类，继承`Metric`类。使用双层循环对每个tracklet和每个detection之间的IoU进行计算。


## 二分图数据关联——mot.associate
基于相似度矩阵（可视作二分图），运行匹配算法得到关联矩阵。

### mot.associate.matcher.Matcher
二分图匹配器。每个匹配器对应一个`Metric`类对象，如果要使用多个不同指标，请在使用`Matcher`之前先定义一个用于组合多个`Metric`的`Metric`，将相似度分数进行融合。

方法列表：
- 构造函数：为匹配器分配一个`Metric`。
    参数：
    + `metric`：`Metric`类的子类对象，该匹配器使用的相似度指标
- `__call__()`：进行数据关联
    参数：
    + `tracklets`：`mot.tracklet.Tracklet`类对象的`list`，设长度为M
    + `detection_features`：`dict`的`list`，设长度为N
    返回值：
    + `row_ind`：整数`list`，每个元素是`tracklets`的一个元素的下标，理论上不允许重复
    + `col_ind`：整数`list`，每个元素是`detection_features`的一个元素的下标，理论上不允许重复，长度与`row_ind`相同，二者各项一一对应
