#!/bin/bash

# Training Convnet
echo "convnet_default 16"
#python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=20 --eval-true

echo "convnet_default 32"
#python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=20 --eval-true

echo "convnet_default 64"
#python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=20 --eval-true

echo "convnet_default 128"
#python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=20 --eval-true

echo "convnet_default 256"
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=20 --eval-true

echo "convnet_default 512"
#python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=20 --eval-true

echo "convnet_default 768"
#python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=20 --eval-true

echo "convnet_default 1024"
#python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=20 --eval-true

# Training Inception Time
echo "inception_time_default 16"
#python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=20 --eval-true

echo "inception_time_default 32"
#python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=20 --eval-true

echo "inception_time_default 64"
#python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=20 --eval-true

echo "inception_time_default 128"
#python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=20 --eval-true

echo "inception_time_default 256"
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=20 --eval-true

echo "inception_time_default 512"
#python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=20 --eval-true

echo "inception_time_default 768"
#python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=20 --eval-true

echo "inception_time_default 1024"
#python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=inception_time --params=models/configuration/resnet_default.json --batch=256 --epochs=20 --eval-true

# Training Resnet
echo "resnet 16"
#python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=20 --eval-true

echo "resnet 32"
#python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=20 --eval-true

echo "resnet 64"
#python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=20 --eval-true

echo "resnet 128"
#python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=20 --eval-true

echo "resnet 256"
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=20 --eval-true

echo "resnet 512"
#python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=20 --eval-true

echo "resnet 768"
#python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=20 --eval-true

echo "resnet 1024"
#python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=20 --eval-true

# Training SiT with Convolutional Patch
echo "sit_conv_patch 16"
#python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_conv_patch 32"
#python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_conv_patch 64"
#python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_conv_patch 128"
#python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_conv_patch 256"
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_conv_patch 512"
#python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_conv_patch 768"
#python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_conv_patch 1024"
#python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=20 --eval-true


# Training SiT with Linear Patch
echo "sit_linear_patch 16"
#python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_linear_patch 32"
#python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_linear_patch 64"
#python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_linear_patch 128"
#python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_linear_patch 256"
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_linear_patch 512"
#python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_linear_patch 768"
#python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=20 --eval-true

echo "sit_linear_patch 1024"
#python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=20 --eval-true


# Training SiT with Original Stem
echo "sit_stem_original 16"
#python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 32"
#python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 64"
#python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 128"
#python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 256"
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 512"
#python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 768"
#python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 1024"
#python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=20 --eval-true


# Training SiT with ReLU Stem
echo "sit_stem_original 16"
#python3 train_deep_model.py --path=data/OBSEA_16/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 32"
#python3 train_deep_model.py --path=data/OBSEA_32/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 64"
#python3 train_deep_model.py --path=data/OBSEA_64/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 128"
#python3 train_deep_model.py --path=data/OBSEA_128/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 256"
python3 train_deep_model.py --path=data/OBSEA_256/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 512"
#python3 train_deep_model.py --path=data/OBSEA_512/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 768"
#python3 train_deep_model.py --path=data/OBSEA_768/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=20 --eval-true

echo "sit_stem_original 768"
#python3 train_deep_model.py --path=data/OBSEA_1024/ --split=0.7 --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=20 --eval-true
