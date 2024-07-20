# this records the commands for experiments
python3.8 create_windows_dataset.py --name OBSEA --save_dir=data/ --path=data/OBSEA/data/ --metric_path=data/OBSEA/metrics/ --window_size=all --metric=PR_AUC
bash ./reproducibility_guide/train_deep_models.sh