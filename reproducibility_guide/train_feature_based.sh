#!/bin/bash

# Commands to train all feature based models for all sizes, save the trained models, and evaluate them and save their results

# Nearest Neighbors
python3.8 train_feature_based.py --path=data/OBSEA_16/TSFRESH_OBSEA_16.csv --classifier=knn --split_per=0.7 --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_32/TSFRESH_OBSEA_32.csv --classifier=knn --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_32.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_64/TSFRESH_OBSEA_64.csv --classifier=knn --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_64.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_128/TSFRESH_OBSEA_128.csv --classifier=knn --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_128.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_256/TSFRESH_OBSEA_256.csv --classifier=knn --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_256.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_512/TSFRESH_OBSEA_512.csv --classifier=knn --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_512.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_768/TSFRESH_OBSEA_768.csv --classifier=knn --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_768.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_1024/TSFRESH_OBSEA_1024.csv --classifier=knn --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_1024.csv --path_save=results/weights/ --eval-true

# Linear SVM
python3.8 train_feature_based.py --path=data/OBSEA_16/TSFRESH_OBSEA_16.csv --classifier=svc_linear --split_per=0.7 --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_32/TSFRESH_OBSEA_32.csv --classifier=svc_linear --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_32.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_64/TSFRESH_OBSEA_64.csv --classifier=svc_linear --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_64.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_128/TSFRESH_OBSEA_128.csv --classifier=svc_linear --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_128.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_256/TSFRESH_OBSEA_256.csv --classifier=svc_linear --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_256.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_512/TSFRESH_OBSEA_512.csv --classifier=svc_linear --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_512.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_768/TSFRESH_OBSEA_768.csv --classifier=svc_linear --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_768.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_1024/TSFRESH_OBSEA_1024.csv --classifier=svc_linear --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_1024.csv --path_save=results/weights/ --eval-true

# Decision Tree
python3.8 train_feature_based.py --path=data/OBSEA_16/TSFRESH_OBSEA_16.csv --classifier=decision_tree --split_per=0.7 --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_32/TSFRESH_OBSEA_32.csv --classifier=decision_tree --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_32.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_64/TSFRESH_OBSEA_64.csv --classifier=decision_tree --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_64.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_128/TSFRESH_OBSEA_128.csv --classifier=decision_tree --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_128.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_256/TSFRESH_OBSEA_256.csv --classifier=decision_tree --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_256.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_512/TSFRESH_OBSEA_512.csv --classifier=decision_tree --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_512.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_768/TSFRESH_OBSEA_768.csv --classifier=decision_tree --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_768.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_1024/TSFRESH_OBSEA_1024.csv --classifier=decision_tree --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_1024.csv --path_save=results/weights/ --eval-true

# Random Forest
python3.8 train_feature_based.py --path=data/OBSEA_16/TSFRESH_OBSEA_16.csv --classifier=random_forest --split_per=0.7 --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_32/TSFRESH_OBSEA_32.csv --classifier=random_forest --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_32.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_64/TSFRESH_OBSEA_64.csv --classifier=random_forest --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_64.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_128/TSFRESH_OBSEA_128.csv --classifier=random_forest --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_128.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_256/TSFRESH_OBSEA_256.csv --classifier=random_forest --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_256.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_512/TSFRESH_OBSEA_512.csv --classifier=random_forest --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_512.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_768/TSFRESH_OBSEA_768.csv --classifier=random_forest --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_768.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_1024/TSFRESH_OBSEA_1024.csv --classifier=random_forest --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_1024.csv --path_save=results/weights/ --eval-true

# Neural Net
python3.8 train_feature_based.py --path=data/OBSEA_16/TSFRESH_OBSEA_16.csv --classifier=mlp --split_per=0.7 --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_32/TSFRESH_OBSEA_32.csv --classifier=mlp --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_32.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_64/TSFRESH_OBSEA_64.csv --classifier=mlp --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_64.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_128/TSFRESH_OBSEA_128.csv --classifier=mlp --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_128.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_256/TSFRESH_OBSEA_256.csv --classifier=mlp --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_256.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_512/TSFRESH_OBSEA_512.csv --classifier=mlp --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_512.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_768/TSFRESH_OBSEA_768.csv --classifier=mlp --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_768.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_1024/TSFRESH_OBSEA_1024.csv --classifier=mlp --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_1024.csv --path_save=results/weights/ --eval-true

# AdaBoost
python3.8 train_feature_based.py --path=data/OBSEA_16/TSFRESH_OBSEA_16.csv --classifier=ada_boost --split_per=0.7 --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_32/TSFRESH_OBSEA_32.csv --classifier=ada_boost --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_32.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_64/TSFRESH_OBSEA_64.csv --classifier=ada_boost --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_64.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_128/TSFRESH_OBSEA_128.csv --classifier=ada_boost --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_128.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_256/TSFRESH_OBSEA_256.csv --classifier=ada_boost --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_256.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_512/TSFRESH_OBSEA_512.csv --classifier=ada_boost --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_512.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_768/TSFRESH_OBSEA_768.csv --classifier=ada_boost --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_768.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_1024/TSFRESH_OBSEA_1024.csv --classifier=ada_boost --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_1024.csv --path_save=results/weights/ --eval-true

# Naive Bayes
python3.8 train_feature_based.py --path=data/OBSEA_16/TSFRESH_OBSEA_16.csv --classifier=bayes --split_per=0.7 --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_32/TSFRESH_OBSEA_32.csv --classifier=bayes --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_32.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_64/TSFRESH_OBSEA_64.csv --classifier=bayes --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_64.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_128/TSFRESH_OBSEA_128.csv --classifier=bayes --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_128.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_256/TSFRESH_OBSEA_256.csv --classifier=bayes --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_256.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_512/TSFRESH_OBSEA_512.csv --classifier=bayes --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_512.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_768/TSFRESH_OBSEA_768.csv --classifier=bayes --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_768.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_1024/TSFRESH_OBSEA_1024.csv --classifier=bayes --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_1024.csv --path_save=results/weights/ --eval-true

# QDA
python3.8 train_feature_based.py --path=data/OBSEA_16/TSFRESH_OBSEA_16.csv --classifier=qda --split_per=0.7 --file=reproducibility_guide/train_val_split.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_32/TSFRESH_OBSEA_32.csv --classifier=qda --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_32.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_64/TSFRESH_OBSEA_64.csv --classifier=qda --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_64.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_128/TSFRESH_OBSEA_128.csv --classifier=qda --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_128.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_256/TSFRESH_OBSEA_256.csv --classifier=qda --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_256.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_512/TSFRESH_OBSEA_512.csv --classifier=qda --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_512.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_768/TSFRESH_OBSEA_768.csv --classifier=qda --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_768.csv --path_save=results/weights/ --eval-true
python3.8 train_feature_based.py --path=data/OBSEA_1024/TSFRESH_OBSEA_1024.csv --classifier=qda --split_per=0.7 --file=experiments/supervised_splits/split_OBSEA_1024.csv --path_save=results/weights/ --eval-true
