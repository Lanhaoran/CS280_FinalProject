

config_dtcr = {}

# seq2seq
config_dtcr['encoder_hidden_units'] = [100,50,50] # or [50,30,30], [100,50,50]
config_dtcr['dilations'] = [1,4,16]

# classification
config_dtcr['classifier_hidden_units'] = [128,2]

# dataset setting]
config_dtcr['train_file'] = 'UCRArchive_2018/{0}/{0}.TRAIN.tsv'.format('TwoLeadECG')
config_dtcr['test_file'] = 'UCRArchive_2018/{0}/{0}.TEST.tsv'.format('TwoLeadECG')

# new dataset path
config_dtcr['motion_train_file'] = r'E:\Character Motion\Prompt_cluster\dataset\dataset\data\train_data_1.npy'
config_dtcr['motion_test_file'] = r'E:\Character Motion\Prompt_cluster\dataset\dataset\data\test_data_1.npy'

config_dtcr['training_samples_num'] = 728 # all training samples
config_dtcr['cluster_num'] = 6 # config in main file
config_dtcr['input_length'] = 250 # ctime steps
# config_dtcr['training_batch_samples'] = 100 # training samples at each batch
config_dtcr['feature_num'] = 96 # feature per step

# loss
config_dtcr['lambda'] = 0.01 # or 1, 1e-1, 1e-2, 1e-3

# train setting
config_dtcr['max_iter'] = 2000 # config
config_dtcr['alter_iter'] = 10
config_dtcr['test_every_epoch'] = 5
config_dtcr['img_path'] = 'train_img' # config
config_dtcr['learning_rate'] = 1e-3
config_dtcr['batch_size'] = 20


# performance setting
config_dtcr['indicator'] = 'NMI' # 'RI' or 'NMI'