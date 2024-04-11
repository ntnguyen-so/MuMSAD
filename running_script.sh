# this records the commands for experiments
python3 create_windows_dataset.py --name OBSEA --save_dir=data/ --path=data/OBSEA/data/ --metric_path=data/OBSEA/metrics/ --window_size=all --metric=Recommendation_ACC
bash ./reproducibility_guide/train_deep_models.sh