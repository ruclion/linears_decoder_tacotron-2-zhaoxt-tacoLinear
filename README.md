# Tacotron2 model

forked from https://github.com/Rayhane-mamah/Tacotron-2

## structure

```
src
├── datasets
├── tacotron
│   ├── models
│   └── utils
```

## 步骤

预处理数据：指定数据集和输出路径

```
run.sh --start-stage 0 --stop-stage 0
```

训练：

```
run.sh --start-stage 1 --stop-stage 1

```

测试：输入音素序列，输出gl合成的语音

```
run.sh --start-stage 2 --stop-stage 2

```

## Style encoder

hparams中可指定use_style_encoder(是否使用style encoder)
当使用style encoder时，可通过style_encoder_type指定style_encoder类型： gst, vae

## 几种模型的设置

- tacotron

   use_style_encoder=False

- gst ( location sensitive attention)

  use_style_encoder=True, style_encoder_type='gst'

- vae 

  use_style_encoder=True, style_encoder_type='vae'












