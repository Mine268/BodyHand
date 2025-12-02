# BodyHand 系统使用方法

```powershell
$env:Path += ";PROGRAM DIR"
$global:ROOT = "YOUR CAPTURE PATH"
```

注意不要忘了 `$env:Path` 定义中的分号。

## 0. 配置拍摄参数

可以配置的相机拍摄：

- 曝光时长（单位：us）
- 曝光增益
- 帧率

使用如下的命令可以可视化图像捕捉结果，通过调整曝光参数使得图像捕捉结果清晰均衡

```powershell
SynchronizeCapture.exe `
	--exposure=10000 `
	--fps=10 `
	--gain=17 `
	--nosave
```

## 1. 相机标定

## 1.1 双视图标定

在双视图标定阶段，使用多张同步拍摄的图像进行左右相机相对位姿的标定，**一旦完成标定之后相机就不能再动**。

手持 7x10 尺寸，格点大小 25mm 的标定板位于双视图均能拍摄到的区域内，使用之前调整好的拍摄参数运行命令：

```powershell
mkdir $global:ROOT\calib_stereo
SynchronizeCapture.exe `
	--exposure=10000 `
	--fps=2 `
	--gain=17 `
	--output_dir=$global:ROOT\calib_stereo
```

该命令将自动地进行同步图像捕捉并保存至 `output_dir` 参数指定的路径内，捕捉的图像数量数量不宜过多（小于 30 张）。捕捉的时候最好手持标定板在相机前缓慢移动，使得标定板能够覆盖图像尽可能多的范围，者能够提升标定精度。

完成捕捉之后将在目标文件夹中生成文件夹 V0 和 V1，分别保存两个视图捕捉的图像。

然后运行双视图标定命令：

```powershell
Calibration.exe $global:ROOT\calib_stereo V 6 9 25
```

第一个参数为上一条命令的 `output_dir`，随后的 `V 6 9 25` 直接照抄。如果标定成功程序将在 `E:\BodyHandCapture\2025_10_22\calib_stereo` 目录下生成 `camera_param.txt` 文件，其中包含了标定信息。如果标定失败，请删除 V0 和 V1 文件夹并将标定板的挪近一些重新捕捉。

## 1.2 单视图标定

单视图标定阶段，使用单张的标定板图像标定世界坐标系的位置，一旦完成标定之后场景中的物体不能再移动。在这一场景下，我们认为驾驶舱座椅是世界坐标系的原点，所以完成标定之后椅子不能再移动。

将标定板放置在椅子上人物膝盖所在位置偏后一些的位置，并保证标定板垂直于地面、与椅背和坐垫所称的直线平行。

```powershell
mkdir $global:ROOT\calib_single
SynchronizeCapture.exe `
	--exposure=10000 `
	--fps=2 `
	--gain=17 `
	--output_dir=$global:ROOT\calib_single
```

然后使用单视图标定命令进行标定，这一程序只需要一张图像就能完成标定。如果可视化的标定板格点紊乱或者输出

```powershell
SingleCalibration.exe $global:ROOT\calib_single\V0\000000.bmp 6 9 25
```

```
[estimatePoseFromChessboard] chessboard not found.
Calibration failed
```

则考虑将座椅挪近一些重新标定。

程序将在 `E:\BodyHandCapture\2025_10_22\calib_single\V0` 中生成 `output.txt` 文件，该文件包含了单视图标定的结果。

## 1.3 编制配置文件

使用 `gen_config.py` 生成配置文件。

```powershell
python gen_config.py `
	$global:ROOT\calib_stereo\camera_param.txt `
	$global:ROOT\calib_single\V0\output.txt
```

这个脚本将在同目录下生成 `calib.cfg` 配置文件。

打开这个生成的配置文件，将其中前三行填入正确的模型文件地址，例如

```
${MODEL_DIR}\yolov8n.onnx
${MODEL_DIR}\vitpose-s-coco.onnx
${MODEL_DIR}\handLR_480x640.onnx
${MODEL_DIR}\hand_mano.onnx
```

# 2. 捕捉运行

直接使用如下命令可以开启姿态估计系统

```powershell
MultiviewPoseEstimation2.exe $global:ROOT\calib.cfg
```

其中第一个参数为刚才生成的配置文件路径，有两个可选参数：

- `--send_tcp`：是否将姿态数据发送给 ue5 进行渲染
- `--no_write_file`：是否不要将姿态数据写入到文件

