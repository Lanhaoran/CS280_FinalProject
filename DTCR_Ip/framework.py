
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from rnns import dilated_encoder, single_layer_decoder
from classification import classifier
from kmeans import kmeans
from utils import * 
from config import config_dtcr



class DTCR():
    def __init__(self, opts):
        self.opts = opts
        
        tf.reset_default_graph()
        
        self.creat_network()
        self.init_optimizers()

    
    def creat_network(self):
        opts = self.opts
        self.encoder_input = tf.placeholder(dtype=tf.float32, shape=(None, opts['input_length'], opts['feature_num']), name='encoder_input')
        self.decoder_input = tf.placeholder(dtype=tf.float32, shape=(None, opts['input_length'], opts['feature_num']), name='decoder_input')
        self.classification_labels = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='classification_labels')
        
        
        # seq2seq
        with tf.variable_scope('seq2seq'):
            self.D_ENCODER = dilated_encoder(opts)
            self.h = self.D_ENCODER.encoder(self.encoder_input)
            
            self.S_DECOER = single_layer_decoder(opts)
            recons_input = self.S_DECOER.decoder(self.h, self.decoder_input)
            
            self.h_fake,self.h_real = tf.split(self.h, num_or_size_splits=2, axis=0)
            
        # classifier
        with tf.variable_scope('classifier'):
            self.CLS = classifier(opts)
            output_without_softmax = self.CLS.cls_net(self.h)
        
        # K-means
        with tf.variable_scope('kmeans'):
            self.KMEANS = kmeans(opts)
            # update F
            kmeans_obj = self.KMEANS.kmeans_optimalize(self.h_real)
        
        
        
        # L-reconstruction
        self.loss_reconstruction = tf.losses.mean_squared_error(self.encoder_input, recons_input)
        # L-classification
        self.loss_classification = tf.losses.softmax_cross_entropy(self.classification_labels, output_without_softmax)
        # L-kmeans
        self.loss_kmeans = kmeans_obj
        
        
    def init_optimizers(self):
        lambda_1 = self.opts['lambda']
        
        # vars
        seq2seq_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='seq2seq')
        cls_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
        end2end_vars = seq2seq_vars + cls_vars
        
        kmeans_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='kmeans')
        
        # loss
        self.loss_dtcr = self.loss_reconstruction + self.loss_classification + lambda_1 * self.loss_kmeans
        
        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.opts['learning_rate'])
        
        # update vars
        self.train_op = optimizer.minimize(self.loss_dtcr, var_list=end2end_vars)
    
    def update_kmeans_f(self, train_h):    
        new_f = truncatedSVD(train_h, self.opts['cluster_num'])
        self.KMEANS.update_f(new_f)        
        
    def train(self, train_data, train_label, test_data, test_label, INDEX):
        '''
        cls_data: shape: (2*N, timestep, dim), 前半部分是fake data
        cls_label: shape: (2*N)
        
        train_data/shape: (N, timestep, dim)
        train_label/test_label: (N)
        
        '''

        opts =self.opts
        # Get batch data
        cls_train_data_all, cls_train_label_all, batches_data, batches_label, iteration = get_Batch(train_data, train_label, opts)
        
        # session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        print('vars_num: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        best_indicator = 0
        best_epoch = -1

        train_list = []
        test_list = []
        l_list = []
        l_recons_list = []
        l_cls_list = []
        l_kmeans_list = []    

        for epoch in range(opts['max_iter']):
            print("Epoch {} / {}".format(epoch, opts['max_iter']))
            for i in range(iteration):
                # processing data and label
                real_train_x_batch = batches_data[i]
                real_train_y_batch = batches_label[i]
                cls_train_data = cls_train_data_all[i]
                cls_train_label = cls_train_label_all[i]
                
                if config_dtcr['feature_num'] == 1:
                    cls_train_data = np.expand_dims(cls_train_data, axis=2)        
                cls_label_ = np.zeros(shape=(cls_train_label.shape[0], len(np.unique(cls_train_label))))
                cls_label_[np.arange(cls_label_.shape[0]), cls_train_label] = 1

                # feed dict
                feed_d = {self.encoder_input: cls_train_data,
                        self.decoder_input: np.zeros_like(cls_train_data),
                        self.classification_labels: cls_label_}

                # init:
                train_h = sess.run(self.h_real, feed_dict=feed_d)
                self.update_kmeans_f(train_h)
                #print('init k-means vars.')
            
                # train
                # train_list = []
                # test_list = []
                l_in_epoch = []
                l_recons_in_epoch = []
                l_cls_in_epoch = []
                l_kmeans_in_epoch = []
            
                _, loss, l_recons, l_cls, l_kmeans = sess.run([self.train_op, self.loss_dtcr, self.loss_reconstruction, self.loss_classification, self.loss_kmeans], feed_dict=feed_d)
                #print('Epoch {} | iteration({}/{}): loss: {}, l_recons: {}, l_cls: {}, l_kmeans: {}'.format(epoch, i+1, iteration, loss, l_recons, l_cls, l_kmeans))
                l_in_epoch.append(loss)
                l_recons_in_epoch.append(l_recons)
                l_cls_in_epoch.append(l_cls)
                l_kmeans_in_epoch.append(l_kmeans)
            
            mean_l = np.mean(np.array(l_in_epoch))
            mean_recon = np.mean(np.array(l_recons_in_epoch))
            mean_cls = np.mean(np.array(l_cls_in_epoch))
            mean_kmeans = np.mean(np.array(l_kmeans_in_epoch))

            l_list.append(np.mean(np.array(mean_l)))
            l_recons_list.append(np.mean(np.array(mean_recon)))
            l_cls_list.append(np.mean(np.array(mean_cls)))
            l_kmeans_list.append(np.mean(np.array(mean_kmeans)))  
            print('Epoch {} | iteration({}/{}): loss: {}, l_recons: {}, l_cls: {}, l_kmeans: {}'.format(epoch, i+1, iteration, mean_l, mean_recon, mean_cls, mean_kmeans))

            if epoch % opts['alter_iter'] == 0:
                train_h = sess.run(self.h_real, feed_dict=feed_d)
                self.update_kmeans_f(train_h)
                print('update F matrix in k-means loss, epoch: {}.'.format(epoch))
                
            if epoch % opts['test_every_epoch'] == 0:
                print('Epoch {}: loss: {}, l_recons: {}, l_cls: {}, l_kmeans: {}'.format(epoch, loss, l_recons, l_cls, l_kmeans))
                train_embedding = self.test(sess, real_train_x_batch)
                test_embedding = self.test(sess, test_data)
                
                # kmeans
                pred_train = cluster_using_kmeans(train_embedding, opts['cluster_num'])
                pred_test = cluster_using_kmeans(test_embedding, opts['cluster_num'])
                
                # performance
                if opts['indicator'] == 'RI':
                    score_train = ri_score(real_train_y_batch, pred_train)
                    score_test = ri_score(test_label, pred_test)
                elif opts['indicator'] == 'NMI':
                    score_train = nmi_score(real_train_y_batch, pred_train)
                    score_test = nmi_score(test_label, pred_test)                    
                print('{}: train: {}\ttest:{}'.format(opts['indicator'], score_train, score_test))
                # performance list
                train_list.append(score_train)
                test_list.append(score_test)
                
                if score_test > best_indicator:
                    best_indicator = score_test
                    best_epoch = epoch
        # plot train and test curve
        show_train_test_curve(opts, train_list, test_list, str(INDEX), i)
        # plot loss curve
        loss_list = [l_list, l_recons_list, l_cls_list, l_kmeans_list]
        show_loss_curve(opts, loss_list, str(INDEX), i)          
        sess.close()
               
        return best_indicator, best_epoch, train_list, test_list
                    
    def test(self, sess, test_data):
        
        if config_dtcr['feature_num'] == 1:
            test_data = np.expand_dims(test_data, axis=2)
        
        feed_d = {self.encoder_input: test_data}
        
        h = sess.run(self.h, feed_dict=feed_d)
        return h