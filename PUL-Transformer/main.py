from train_model import pre_train, fine_tune
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main(model_config=None):
    modelConfig = {
        "state": "train", # or fine_tune
        "epoch": 500,
        "batch_size": 60,
        "dropout": 0.15,
        "lr": 1e-4,
        "device": "cuda:0",
        "multiplier": 2.5,
        "data": 'Strawberry',
        "d_model": 512,
        #"save_dir": "./CheckpointsCondition/",
        "save_dir": "./Checkpoints/5",
        "finetune_dir": "./Fine_tune_model/2",
        "test_load_weight": "ckpt_500_.pt",
        #"sampled_dir": "./SampledImgs/",
        "sampled_dir": "./SampledSequences/",
        "sampledNoisyImgName": "NoisyGuidenceImg_",
        "sampledImgName": "SampledGuidenceImg_",
        "nrow": 8,
        #######################################
        "log_dir": "./log/5",
        "finetune_log_dir": "./f_log/2",
        "sampledPolicyName": "SampledGuidencePolicy_9_",
        "l2_lambda": 0.9,
        "kmeans_lambda": 1e-2,
        "kmeans_epoch": 5,
        "max_min_noral": False,
        "standard_noral": False
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        pre_train(modelConfig)
    else:
        fine_tune(modelConfig)
        pass


if __name__ == '__main__':
    main()
