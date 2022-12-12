
import copy
import os
from datetime import datetime
import json
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
import math
from sklearn.linear_model import LogisticRegression
from cfg_config import cfg_args
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn import metrics
from sklearn.utils import shuffle
from data import CFGDataset
from model.DenseGraphMatching import HierarchicalGraphMatchNetwork
from model.DenseGraphMatching import EarlyStopping
from model.DenseGraphMatchingcross import HierarchicalGraphMatchNetworkcross
import torch.nn.functional as functional
# from DenseGraphMatching import drop_feature
# from tensorboardX import SummaryWriter
from utils import create_dir_if_not_exists, write_log_file
from utils import generate_epoch_pair
from sklearn.metrics import accuracy_score
import math


# from pytorch-tools import EarlyStopping

class CFGTrainer(object):
    def __init__(self, node_init_dims, data_dir, device, log_file, best_model_file,best_cross_file, args):
        super(CFGTrainer, self).__init__()
        # training parameters
        self.max_epoch = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.device = device
        self.patience = args.patience
        self.log_file = log_file
        self.best_model_path = best_model_file
        self.best_cross_path = best_cross_file
        self.drop_feature1 = args.drop_feature1
        self.drop_feature2 = args.drop_feature2
        self.drop_edge1 = args.drop_edge1
        self.drop_edge2 = args.drop_edge2
        self.model = HierarchicalGraphMatchNetwork(node_init_dims=node_init_dims, arguments=args, device=device).to(
            device)
        # print(self.model)
        self.modelcross = HierarchicalGraphMatchNetworkcross(node_init_dims=node_init_dims, arguments=args, device=device).to(
            device)
        write_log_file(self.log_file, str(self.model))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.earlystopping = EarlyStopping(self.patience, verbose=True)
        cfg = CFGDataset(data_dir=data_dir, batch_size=self.batch_size)
        self.graph_train = cfg.graph_train
        self.classes_train = cfg.classes_train

        self.epoch_data_valid = cfg.valid_epoch
        self.epoch_data_test = cfg.test_epoch
        self.graph_test = cfg.graph_test
        self.classes_test = cfg.classes_test
        self.global_agg = args.global_agg
        self.data_dir = data_dir

 
    def fit(self):
        best_val_auc = None

        best_LR_auc = None
        best_early_score = None
        # ----------------cosine----------

        score,lrauc = self.testing(self, model=self.model,
                                     eval_epoch_data=self.epoch_data_valid,
                                     device=self.device)
        print('Before Training: the val AUC score is: ' + str(score))
        print('Before Training: the lr AUC score is: ' + str(lrauc))
        score_, lrauc_= self.testing(self, model=self.model,
                             eval_epoch_data=self.epoch_data_test,
                             device=self.device)
        print('Before Training: the test AUC score is: ' + str(score_))
        print('Before Training: the test lrAUC score is: ' + str(lrauc_))

        

        #---------------------------------feature----------------------------------------------------
        for i in range(1, self.max_epoch + 1):  # for i in range(1, self.max_epoch + 1):
            print('Epoch {} :', i)

            loss_avg = self.train_one_epoch(self,
                                                                      model=self.model, optimizer=self.optimizer,
                                                                      graphs=self.graph_train,
                                                                      classes=self.classes_train,
                                                                      batch_size=self.batch_size,
                                                                      device=self.device,
                                                                      drop_feature1=self.drop_feature1,
                                                                      drop_feature2=self.drop_feature2,
                                                                      drop_edge1=self.drop_edge1,
                                                                      drop_edge2=self.drop_edge2,
                                                                      load_data=None)

            # validation
            valid_auc,valid_lr = self.testing(self, model=self.model, eval_epoch_data=self.epoch_data_valid, device=self.device)
            print('Now, the evaluation score is: ' + str(valid_auc))
            print('Now, the evaluation lr score is: ' + str(valid_lr))

            test_auc,test_lr = self.testing(self, model=self.model, eval_epoch_data=self.epoch_data_test, device=self.device)
            print('Now, the test score is: ' + str(test_auc))
            print('Now, the test lr score is: ' + str(test_lr))

            write_log_file(self.log_file,
                           "EPOCH {0}/{1}:\t Contrastive loss = {2} @ {3}".format(i, self.max_epoch, loss_avg,
                                                                                  datetime.now()))
            
            # ------------------------save----------------------------------
            if best_val_auc is None or best_val_auc < valid_auc:
                write_log_file(self.log_file,
                               '\tvalidation AUC increased ({} ---> {}), and saving the model ... '.format(best_val_auc,
                                                                                                           valid_auc))
                best_val_auc = valid_auc
                torch.save(self.model.state_dict(), self.best_model_path)


            if best_LR_auc is None or best_LR_auc < valid_lr:
                write_log_file(self.log_file,
                               '\tvalidation AUC with LR increased ({} ---> {}), and saving the model ... '.format(best_LR_auc,
                                                                                                           valid_lr))
                best_LR_auc = valid_lr
                

        print('best_val_auc = ', best_val_auc)
        print('best_lr_auc = ', best_LR_auc)


        score0, lrauc0 = self.testing1(self, model=self.model, eval_epoch_data=self.epoch_data_valid,
                             device=self.device)
        print('Final valid: the cosine AUC score is: ' + str(score0))
        print('Final valid: the lr AUC score is: ' + str(lrauc0))

        score1,lrauc1 = self.testing1(self, model=self.model, eval_epoch_data=self.epoch_data_test,
                             device=self.device)
        print('Final Testing: the cosine AUC score is: ' + str(score1))
        print('Final testing: the lr AUC score is: ' + str(lrauc1))

        return score0, score1
        # return 0



    @staticmethod
    def testing1(self, model, eval_epoch_data, device):
        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(self.best_model_path))

        model.eval()
        with torch.no_grad():
            tot_diff = []
            tot_truth = []
            tot_truth1 = []
            tot_diff1 = []
            for cur_data in eval_epoch_data:
                x1, x2, adj1, adj2, y = cur_data
                feature_p_init = torch.FloatTensor(x1).to(device)
                adj_p = torch.FloatTensor(adj1).to(device)
                feature_h_init = torch.FloatTensor(x2).to(device)
                adj_h = torch.FloatTensor(adj2).to(device)
                

                feature_p = model(batch_x_p=feature_p_init, batch_adj_p=adj_p)
                feature_h = model(batch_x_p=feature_h_init, batch_adj_p=adj_h)
                sim_score = self.cosine(feature_p, feature_h)

                


                tot_diff += list(sim_score.data.cpu().numpy())
                tot_truth += list(y > 0)


        diff = np.array(tot_diff) * -1
        truth = np.array(tot_truth)

        fpr, tpr, _ = roc_curve(truth, (1 - diff) / 2)
        model_auc = auc(fpr, tpr)

        aucc = self.evaluation((1 - diff) / 2, truth, ratio=0.1)

        
        return model_auc,aucc


    @staticmethod
    def testing(self, model, eval_epoch_data, device):
        model.train()
        
        model.eval()
        with torch.no_grad():
            tot_diff = []
            tot_truth = []

            for cur_data in eval_epoch_data:
                x1, x2, adj1, adj2, y = cur_data
                feature_p_init = torch.FloatTensor(x1).to(device)
                adj_p = torch.FloatTensor(adj1).to(device)
                feature_h_init = torch.FloatTensor(x2).to(device)
                adj_h = torch.FloatTensor(adj2).to(device)

                feature_p = model(batch_x_p=feature_p_init, batch_adj_p=adj_p)
                feature_h = model(batch_x_p=feature_h_init, batch_adj_p=adj_h)

                sim_score = self.cosine(feature_p, feature_h)
                tot_diff += list(sim_score.data.cpu().numpy())
                tot_truth += list(y > 0)

        diff = np.array(tot_diff) * -1
        truth = np.array(tot_truth)
        # truth = truth.astype(int)

        fpr, tpr, _ = roc_curve(truth, (1 - diff) / 2)
        model_auc = auc(fpr, tpr)


        aucc = self.evaluation((1 - diff) / 2, truth, ratio=0.1)

        
        return model_auc, aucc
        

    # def test(self,model, x, edge_index, y, final=False):
    def label_classification(self, cos, y, ratio):

        X = cos
        Y = y
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)  

        X, Y = shuffle(X, Y)

        X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                            test_size=0.9)
        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 10)
        # 一个参数

        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                           param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                           verbose=0)
        clf.fit(X_train, y_train)

        
        y_pred = logreg.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        model_auc = auc(fpr, tpr)
        
        return model_auc

    def evaluation(self, cos, y, ratio):
        y = y.astype(int)
        clf = LogisticRegression(penalty='l2', random_state=0, max_iter=2000)
        
        cos[cos < 0.94] = 0.94
        

        maxn = np.max(cos)
        minn = np.min(cos)
        avg = np.mean(cos)


        coss = (cos - minn) / (maxn - minn)

        X = np.power(10000, coss)/10000

        maxn1 = np.max(X)
        minn1 = np.min(X)
        avg1 = np.mean(X)

        
        Y = y
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)  




        train_embs, test_embs, train_labels, test_labels = train_test_split(X, Y, test_size=0.9)

        clf.fit(train_embs, train_labels)
        pred_test_labels = clf.predict(test_embs)
        acc = accuracy_score(test_labels, pred_test_labels)
        fpr, tpr, _ = roc_curve(test_labels, pred_test_labels)
        model_auc = auc(fpr, tpr)

        return model_auc



    @staticmethod  
    def partition_test_dataset(graphs, partitions, perm):
        lengraph = len(graphs)  
        st = 0.0
        ret = []
        for partition in partitions:  
            cur_g = []
            ed = st + partition * lengraph
            for cls in range(int(st), int(ed)):
                cur_g.append(graphs[cls])

            ret.append(cur_g)
            st = ed
        return ret


    @staticmethod
    def cosine(feature_h, feature_p):
        agg_h = torch.mean(feature_h, dim=1)
        agg_p = torch.mean(feature_p, dim=1)
        sim = functional.cosine_similarity(agg_p, agg_h, dim=1).clamp(min=-1, max=1)
        return sim
    @staticmethod
    def emb(self,  model,  graphs, classes, batch_size, device, load_data=None):
        self.model.load_state_dict(torch.load(self.best_model_path))

        if load_data is None:
            epoch_data = generate_epoch_pair(graphs, classes, batch_size)
        else:
            epoch_data = load_data

        perm = np.random.permutation(len(epoch_data))  
        feature_p_list = []
        feature_h_list = []
        adj_p_list = []
        adj_h_list = []


        for index in perm:
            # get feature matrix and adjacency matrix for graph_p and graph_h
            cur_data = epoch_data[index]
            x1, x2, adj1, adj2, y = cur_data
            feature_p_init = torch.FloatTensor(x1).to(device)  # graph_p feat
            adj_p = torch.FloatTensor(adj1).to(device)  # graph p adj
            feature_h_init = torch.FloatTensor(x2).to(device)  # graph h feat
            adj_h = torch.FloatTensor(adj2).to(device)  # graph h adj
            feature_p = model(batch_x_p=feature_p_init.clone(), batch_adj_p=adj_p.clone())
            feature_p1 = copy.copy(feature_p)
            feature_p_list.append(feature_p1)




            feature_h = model(batch_x_p=feature_h_init.clone(), batch_adj_p=adj_h.clone())
            feature_h1 = copy.copy(feature_h)
            feature_h_list.append(feature_h1)

            adj_p_list.append(adj_p.clone())
            adj_h_list.append(adj_h.clone())





        return feature_p_list, feature_h_list, adj_p_list, adj_h_list

    @staticmethod
    def test_emb(self,model,earlystopping,eval_epoch_data,device):
        feature_p_list = []
        feature_h_list = []
        adj_p_list = []
        adj_h_list = []
        y_list = []

        self.model.load_state_dict(torch.load(self.best_model_path))

        for cur_data in eval_epoch_data:
            x1, x2, adj1, adj2, y = cur_data
            feature_p_init = torch.FloatTensor(x1).to(device)
            adj_p = torch.FloatTensor(adj1).to(device)
            feature_h_init = torch.FloatTensor(x2).to(device)
            adj_h = torch.FloatTensor(adj2).to(device)


            feature_p = model(batch_x_p=feature_p_init.clone(), batch_adj_p=adj_p.clone())
            feature_p1 = copy.copy(feature_p)
            feature_p_list.append(feature_p1)
            # feature_p_path = os.path.join(self.data_dir, 'featurep.txt')

            feature_h = model(batch_x_p=feature_h_init.clone(), batch_adj_p=adj_h.clone())
            feature_h1 = copy.copy(feature_h)
            feature_h_list.append(feature_h1)

            adj_p_list.append(adj_p.clone())
            adj_h_list.append(adj_h.clone())
            y_list.append(y)

        return feature_p_list, feature_h_list, adj_p_list, adj_h_list,y_list




    @staticmethod
    def train_one_epoch(self, model, optimizer, graphs, classes, batch_size, device, drop_feature1,
                        drop_feature2, drop_edge1, drop_edge2, load_data=None):
        model.train()
        if load_data is None:
            epoch_data = generate_epoch_pair(graphs, classes, batch_size)
        else:
            epoch_data = load_data

        perm = np.random.permutation(len(epoch_data))  

        cum_loss = 0.0
        num = 0
        loss_p = 0.0
        loss_h = 0.0
        

        for index in perm:
            cur_data = epoch_data[index]
            x1, x2, adj1, adj2, y = cur_data
            feature_p_init = torch.FloatTensor(x1).to(device)
            adj_p = torch.FloatTensor(adj1).to(device)

            drop_feature_p1 = model.drop_feature(feature_p_init, drop_feature1)
            drop_feature_p2 = model.drop_feature(feature_p_init, drop_feature2)

            drop_edge_p1 = adj_p.clone()
            drop_edge_p2 = adj_p.clone()
            for i in range(adj_p.size()[0]):
                drop_edge_p1[i] = model.aug_random_edge(adj_p[i].cpu().numpy(), drop_edge1)
                drop_edge_p2[i] = model.aug_random_edge(adj_p[i].cpu().numpy(), drop_edge2)

            feature_h_init = torch.FloatTensor(x2).to(device)
            adj_h = torch.FloatTensor(adj2).to(device)
            drop_feature_h1 = model.drop_feature(feature_h_init, drop_feature1)
            drop_feature_h2 = model.drop_feature(feature_h_init, drop_feature2)

            drop_edge_h1 = adj_h.clone()
            drop_edge_h2 = adj_h.clone()
            for i in range(adj_h.size()[0]):

                drop_edge_h1[i] = model.aug_random_edge(adj_h[i].cpu().numpy(), drop_edge1)
                drop_edge_h2[i] = model.aug_random_edge(adj_h[i].cpu().numpy(), drop_edge2)

            
            feature_p1 = model(batch_x_p=drop_feature_p1, batch_adj_p=drop_edge_p1)
            # graph1 view2
            feature_p2 = model(batch_x_p=drop_feature_p2, batch_adj_p=drop_edge_p2)
            # graph2 view1
            feature_h1 = model(batch_x_p=drop_feature_h1, batch_adj_p=drop_edge_h1)
            # graph2 view2
            feature_h2 = model(batch_x_p=drop_feature_h2, batch_adj_p=drop_edge_h2)

            
            feature_p0 = model(batch_x_p=feature_p_init, batch_adj_p=adj_p)
            feature_h0 = model(batch_x_p=feature_h_init, batch_adj_p=adj_h)


            feature_p1, feature_p2 = model.matching_layer(feature_p1, feature_p2)

            feature_h1, feature_h2 = model.matching_layer(feature_h1, feature_h2)

            feature_p1, feature_htemp = model.matching_layer(feature_p1, feature_h0)

            feature_p2, feature_htemp = model.matching_layer(feature_p2, feature_h0)

            feature_h1, feature_ptemp = model.matching_layer(feature_h1, feature_p0)

            feature_h2, feature_ptemp = model.matching_layer(feature_h2, feature_p0)







            
            optimizer.zero_grad()
            

            loss_p = model.loss(feature_p1, feature_p2, batch_size=0)
            loss_h = model.loss(feature_h1, feature_h2, batch_size=0)  #
            loss_sum = (loss_p + loss_h) * 0.5 #

            loss_sum.backward()
            
            optimizer.step()
            cum_loss += loss_sum

            if num % int(len(perm) / 10) == 0:
                print('\t Training Contrastive: {}/{}: index = {} loss = {}'.format(num, len(epoch_data), index,
                                                                                    loss_sum))
                
            num = num + 1
        
        return cum_loss / len(perm)

    @staticmethod
    def predicting(self, model, optimizer, graphs, classes, batch_size, device, drop_feature1, drop_feature2,
                   drop_edge1, drop_edge2, load_data=None):
        

        self.model.eval()
        
        if load_data is None:
            epoch_data = generate_epoch_pair(graphs, classes, len(graphs))
        else:
            epoch_data = load_data

        perm = np.random.permutation(len(epoch_data))  # Random shuffle#把ep，perm是什么
        cum_loss = 0.0
        num = 0
        loss_p = 0.0
        loss_h = 0.0
        best_score = 0.0
        tot_diff = []
        tot_truth = []


        for index in perm:
            cur_data = epoch_data[index]
            x1, x2, adj1, adj2, y = cur_data
            feature_p_init = torch.FloatTensor(x1).to(device)
            y = torch.FloatTensor(y).to(device)
            adj_p = torch.FloatTensor(adj1).to(device)

            feature_h_init = torch.FloatTensor(x2).to(device)
            adj_h = torch.FloatTensor(adj2).to(device)
            
            feature_p1 = model(batch_x_p=feature_p_init, batch_adj_p=adj_p)
            
            feature_p2 = model(batch_x_p=feature_h_init, batch_adj_p=adj_h)

            
            auc_score = model.prediction(feature_p1, feature_p2, y)


        return auc_score

        
    @staticmethod
    def eval_auc_epoch(model, eval_epoch_data):
        model.eval()
        with torch.no_grad():
            tot_diff = []
            tot_truth = []
            for cur_data in eval_epoch_data:
                x1, x2, adj1, adj2, y = cur_data
                batch_output = model(batch_x_p=x1, batch_x_h=x2, batch_adj_p=adj1, batch_adj_h=adj2)  # simscore

                tot_diff += list(batch_output.data.cpu().numpy())
                tot_truth += list(y > 0)

        diff = np.array(tot_diff) * -1
        truth = np.array(tot_truth)

        fpr, tpr, _ = roc_curve(truth, (1 - diff) / 2)
        model_auc = auc(fpr, tpr)
        return model_auc


if __name__ == '__main__':
    d = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg_args.gpu_index)

    main_data_dir = cfg_args.data_dir
    graph_name = cfg_args.dataset
    graph_min = cfg_args.graph_size_min
    graph_max = cfg_args.graph_size_max
    graph_init_dim = cfg_args.graph_init_dim  
    # <-><-><-> only for log, delete below if open source
    title = '{}_Min{}_Max{}_InitDims{}_Task_{}_Filter_{}_Match_{}_P_{}_Agg_{}_Hidden_{}_Epoch_{}_Batch_{}_lr_{}_Dropout_{}_Global_{}_with_agg_{}' \
        .format(graph_name,
                graph_min,
                graph_max,
                graph_init_dim,  
                cfg_args.task,
                cfg_args.filters,
                cfg_args.match,  
                cfg_args.perspectives, 
                cfg_args.match_agg,  
                cfg_args.hidden_size,  
                cfg_args.epochs,
                cfg_args.batch_size,  
                cfg_args.lr,  
                cfg_args.dropout,  
                int(cfg_args.global_flag),
                cfg_args.global_agg)
    main_log_dir = cfg_args.log_path + '{}_Min{}_Max{}_InitDims{}_Task_{}/'.format(graph_name, graph_min, graph_max,
                                                                                   graph_init_dim, cfg_args.task)
    create_log_str = create_dir_if_not_exists(main_log_dir)
    best_model_dir = main_log_dir + 'BestModels_Repeat_{}/'.format(cfg_args.repeat_run)
    create_BestModel_dir = create_dir_if_not_exists(best_model_dir)
    LOG_FILE = main_log_dir + 'repeat_{}_'.format(cfg_args.repeat_run) + title + '.txt'
    BestModel_FILE = best_model_dir + title + '.BestModel'
    BestCross_FILE = best_model_dir + title + '.BestCross'
    CSV_FILE = main_log_dir + title + '.csv'

    write_log_file(LOG_FILE, create_log_str)
    write_log_file(LOG_FILE, create_BestModel_dir)
    write_log_file(LOG_FILE, str(cfg_args))
    write_log_file(LOG_FILE, title)
    # <-><-><-> only for log, delete above if open source

    sub_data_dir = '{}_{}ACFG_min{}_max{}'.format(graph_name, graph_init_dim, graph_min, graph_max)
    cfg_data_dir = os.path.join(main_data_dir, sub_data_dir) if 'ffmpeg' in sub_data_dir else os.path.join(
        main_data_dir, sub_data_dir, 'acfgSSL_6')
    assert os.path.exists(cfg_data_dir), "the path of {} is not exist!".format(cfg_data_dir)

    cfg_trainer = CFGTrainer(node_init_dims=graph_init_dim, data_dir=cfg_data_dir, device=d, log_file=LOG_FILE,
                             best_model_file=BestModel_FILE, best_cross_file= BestCross_FILE,args=cfg_args)
    ret_cross_auc, ret_best_val_auc = cfg_trainer.fit()
    print('Best_valid_auc = ', ret_cross_auc)
    print('Best_test_auc = ', ret_best_val_auc)
    print('END')
    # ret_final_test_auc = cfg_trainer.testing()
