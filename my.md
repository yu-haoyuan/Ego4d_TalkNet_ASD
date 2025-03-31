用作代码更新笔记
1 搞清楚源代码输入的内容是什么
2 整理好自己的数据集
3 改接口 接入mrope进行训练

1源代码传入的dataloader是
        audio = torch.FloatTensor(numpy.array(audioFeatures))
        faces = torch.FloatTensor(numpy.array(visualFeatures))
        labels = torch.LongTensor(numpy.array(labels))
        return audio, faces, labels

写一个脚本删除test集 只需要train和val √
查看文件夹下有多少文件保证删除正确
df -h 存储
ls -l | grep "^-" | wc -l
ls -l | grep ^d | wc -l 439 = 50+389
我需要的dataloader
0b4cacb1-970f-4ef0-85da-371d81f899e0
每15秒为一个batch
一个batch为450帧图片 对应的人脸标签 看见的标记为1 看不见的标记为0
一个batch为15s的音频 对应的音频标签 有声音的标记为1 ~为0
原始的dataloader处理逻辑
输入原始音频,根据面部数据做对应裁剪然后进入训练:根据 start 和 end 时间截取音频片段：audio[int(start*sr): int(end*sr)]。
所以这样的话,我们的15秒音频也可以拿来训练 音频实现对齐,代码可以复用
好,那么我们开始处理音视频数据
新data设置为dataset文件夹,逐步从data文件夹里面进行迁移

Ego4d_TalkNet_ASD/
├── data/
│   ├── split/
│   │   ├── train.list
│   │   └── val.list
│   ├── video_imgs/
│   │   └── 0b4cacb1-970f-4ef0-85da-371d81f899e0/
│   │       ├── img_00001.jpg
│   │       ...
│   │       └── img_09000.jpg
│   └── wav/
│       └── 0b4cacb1-970f-4ef0-85da-371d81f899e0.wav
├── dataset_clips/ # New output directory
│   ├── train/
│   │   └── {video_id_from_train_list}/
│   │       ├── clip_f000000/ # Clip starting at frame 0
│   │       │   ├── frames/
│   │       │   │   ├── img_00001.jpg # Original frame 1
│   │       │   │   ...
│   │       │   │   └── img_00450.jpg # Original frame 450
│   │       │   └── audio_f000000_f000449.wav # Audio for frames 0-449
│   │       ├── clip_f000450/ # Clip starting at frame 450
│   │       │   ├── frames/
│   │       │   │   ├── img_00451.jpg # Original frame 451
│   │       │   │   ...
│   │       │   │   └── img_00900.jpg # Original frame 900
│   │       │   └── audio_f000450_f000899.wav # Audio for frames 450-899
│   │       ...
│   │   └── {another_train_video_id}/
│   │       ...
│   └── val/
│       └── {video_id_from_val_list}/
│           ├── clip_f000000/
│           │   ├── frames/
│           │   └── audio_f000000_f000449.wav
│           ...
└── prepare_ego4d_clips_timestamps.py # Your script


new data in dataset is use to agile frame from 0001.jpg to 0000.jpg
and the genejson will use like
│   └── val/
│       └── {video_id_from_val_list}/
│           ├── clip_f000000/
│           │   ├── frames/
│           │   └── audio_f000000_f000449.wav
to generate the according file to make the full dataset

when we need to change the clip length, the newdata.py will be used
and then we use the rename to remake the name of jpgs
and then we use the genejson to get new json train.