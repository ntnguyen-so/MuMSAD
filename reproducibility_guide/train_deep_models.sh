#!/bin/bash

# Training Convnet
python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true

# Training Inception Time
python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=inception_time --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true

# Training Resnet
python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true

# Training SiT with Convolutional Patch
python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true


# Training SiT with Linear Patch
python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true


# Training SiT with Original Stem
python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true


# Training SiT with ReLU Stem
python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
