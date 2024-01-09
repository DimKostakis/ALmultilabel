import copy
import os
import numpy as np
import warnings
from scipy.io import arff
from skmultilearn.dataset import load_from_arff
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_multilabel_classification
import random
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler
from libact.query_strategies import RandomSampling
from libact.query_strategies.multilabel import AdaptiveActiveLearning
from libact.query_strategies.multilabel import MMC, BinaryMinimization
from libact.models.multilabel import BinaryRelevance
from libact.base.dataset import import_scipy_mat
from libact.models import LogisticRegression, SVM
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,f1_score,hamming_loss,average_precision_score,recall_score
from libact.models import SklearnAdapter
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.dataset import load_from_arff
import os.path
from os import path

class KFoldCross:

  def __init__(self, k,dataset_value):
    self.k=k
    self.dataset_value=dataset_value
    self.metrics={}
    self.not_custom=True
    #data = make_multilabel_classification(
    #n_samples=300, n_classes=10, allow_unlabeled=False)
    #self.data = StandardScaler().fit_transform(data[0])
    #self.target = data[1]
    self.f1_macro=list()
    if self.dataset_value==1:
      self.dataset_name="emotions"
      data = arff.loadarff('/content/emotions.arff')
      df = pd.DataFrame(data[0])
      array=df.to_numpy()
      self.data=array[:,:-6]
      self.target=array[:,-6:]
      self.target=np.where(self.target==b'0',0,self.target)
      self.target=np.where(self.target==b'1',1,self.target)
    elif self.dataset_value==2:
      self.dataset_name="genbase"
      data = arff.loadarff('/content/genbase.arff')
      df = pd.DataFrame(data[0])
      array=df.to_numpy()
      self.data=array[:,1:-27]
      self.target=array[:,-27:]
      self.data=np.where(self.data==b'NO',0,self.data)
      self.data=np.where(self.data==b'YES',1,self.data)
      self.target=np.where(self.target==b'0',0,self.target)
      self.target=np.where(self.target==b'1',1,self.target)
    elif self.dataset_value==3:
      self.dataset_name="yeast"
      data = arff.loadarff('/content/yeast.arff')
      df = pd.DataFrame(data[0])
      array=df.to_numpy()
      self.data=array[:,:-14]
      self.target=array[:,-14:]
      self.target=np.where(self.target==b'0',0,self.target)
      self.target=np.where(self.target==b'1',1,self.target)
    elif self.dataset_value==4:
      self.dataset_name="bibtex"
      self.data,self.target=load_from_arff('/content/bibtex.arff',label_count=159,label_location="end",input_feature_type='int', encode_nominal=False, load_sparse=False,
      return_attribute_definitions=False)
      self.data=self.data.toarray()
      self.target=self.target.toarray()
    elif self.dataset_value==5:
      self.dataset_name="medical"
      self.data,self.target=load_from_arff('/content/medical.arff',label_count=45,label_location="end",input_feature_type='int', encode_nominal=False, load_sparse=False,
      return_attribute_definitions=False)
      self.data=self.data.toarray()
      self.target=self.target.toarray()


    self.kfold=KFold(k, True,random_state=1)



  def get_qs(self,train_ds):
    if self.strategy_value==1:
      self.strategy_name="MMC"
      qs = MMC(train_ds, br_base=LogisticRegression())
      return qs
    elif self.strategy_value==2:
      self.strategy_name="RandS"
      qs = RandomSampling(train_ds)
      return qs
    elif self.strategy_value==3:
      self.strategy_name="AdaptiveAl"
      qs=AdaptiveActiveLearning(train_ds, base_clf=LogisticRegression())
      return qs
    elif self.strategy_value==4:
      self.strategy_name="BinMin"
      qs=BinaryMinimization(train_ds, LogisticRegression())
      return qs


  def custom_run(self,strategy_value,clf_value,n_iterations,n_queries,percentage):
    self.not_custom=False
    self.clf_value=clf_value
    self.strategy_value=strategy_value
    self.n_iters=n_iterations
    self.n_q=n_queries
    self.percentage=percentage
    if self.n_iters==0:
      len_split=len(self.data)-len(self.data)//self.k
      n_labelled=int(round(self.percentage*len_split/100))
      self.n_iters=10
      self.n_q=int(n_labelled/10)
    if self.strategy_value==1:
      self.strategy_name="MMC"
    elif self.strategy_value==2:
      self.strategy_name="RandS"
    elif self.strategy_value==4:
      self.strategy_name="BinMin"
    if self.clf_value==1:
      self.clf_name="LR_"
    elif self.clf_value==2:
      self.clf_name="DecisionTree_"
    elif self.clf_value==3:
      self.clf_name="GausianNB_"

    filename="al_"+self.dataset_name+"_"+self.clf_name+self.strategy_name+"_it="+str(self.n_iters)+"_qr="+str(self.n_q)+"_lratio="+str(self.percentage)+".pkl"
    return filename



  def perform_validation(self):
    if self.not_custom:
      self.strategy_value=int(input("Type the number corresponding to the query strategy you would like to use:\n1.MMC\n2.Random\n3.AdaptiveActiveLearning\n4.BinaryMinimazition"))
      self.percentage=int(input("What percentage of the pool would you like to be labelled? "))
      self.n_iters=int(input("How many iterations should be made?"))
      self.n_q=int(input("How many queries should be made per iterations?"))
    len_split=len(self.data)-len(self.data)//self.k
    n_labelled=int(round(self.percentage*len_split/100))
    n_unlabelled=len_split-n_labelled
    self.train_idx=random.sample(range(0,len_split),n_labelled)
    if self.not_custom:
      print("L =",n_labelled,"and U =",n_unlabelled)
    fold=0
    for train, test in self.kfold.split(self.data,self.target):
       scores=list()
       X_train = self.data[train]
       y_train = self.target[train]
       train_ds = Dataset(X_train, y_train[:n_labelled].tolist() + [None] * (len(y_train) - n_labelled))
       test_ds= Dataset(self.data[test],self.target[test].tolist())
       x,y=test_ds.format_sklearn()
       self.qs=self.get_qs(train_ds)
       fully_labeled_trn_ds = Dataset(X_train, y_train)
       lbr = IdealLabeler(fully_labeled_trn_ds)
       print(len(train_ds))
       if self.clf_value==1:
        self.clf_name="LR_"
        model = BinaryRelevance(LogisticRegression())
       elif self.clf_value==2:
        self.clf_name="DecisionTree_"
        adapter=SklearnAdapter(DecisionTreeClassifier())
        model = BinaryRelevance(adapter)
       elif self.clf_value==3:
        self.clf_name="GausianNB_"
        adapter=SklearnAdapter(GaussianNB())
        model= BinaryRelevance(adapter)
       fold+=1
       for iter in range(self.n_iters):
         for quer in range(self.n_q):
          ask_id=self.qs.make_query()
          X = X_train[ask_id]
          lb = lbr.label(X)
          train_ds.update(ask_id, lb)
         model.train(train_ds)
       self.f1_macro.append(round(f1_score(y,model.predict(self.data[test]),average="macro"),4))
       #print("Accuracy score after iteration "+str(iter)+" :"+str(accuracy_score(self.target[test],model.predict(self.data[test]))))
       scores.append(round(model.score(test_ds,criterion='hamming'),3))
       scores.append(round(f1_score(y,model.predict(self.data[test]),average="micro"),3))
       scores.append(round(f1_score(y,model.predict(self.data[test]),average="macro"),3))
       scores.append(round(average_precision_score(y,model.predict(self.data[test])),3))
       scores.append(round(recall_score(y,model.predict(self.data[test]),average="macro"),3))
       self.metrics[fold]=[]
       for score in scores:
         self.metrics[fold].append(score)
       print("Done fold:"+str(fold))
       if self.not_custom:
        print("Hamming loss score after fold "+str(fold)+" :"+str(hamming_loss(y,model.predict(self.data[test]))))
        print("Micro f1 score after fold "+str(fold)+" :"+str(f1_score(y,model.predict(self.data[test]),average="micro")))
        print("Macro f1 score after fold "+str(fold)+" :"+str(f1_score(y,model.predict(self.data[test]),average="macro")))
        print("Average precision score after fold "+str(fold)+" :"+str(average_precision_score(y,model.predict(self.data[test]))))
        print("----------------------------")
    column_names=["HammingLoss","f1-micro","f1-macro","AveragePrec","Recall-macro"]
    data_frame = pd.DataFrame.from_dict(self.metrics,orient="index",columns= column_names)
    filename="al_"+self.dataset_name+"_"+self.clf_name+self.strategy_name+"_it="+str(self.n_iters)+"_qr="+str(self.n_q)+"_lratio="+str(self.percentage)+".pkl"

    data_frame.to_pickle('/content/drive/MyDrive/missingmulti/%s'%(filename))

    print(filename+"done")
    #print("Mean of f1 macro scores is:",np.mean(self.f1_macro))
    #print(self.metrics)



def complete_experiment():
  cases=0
  count=0
  for qs in range(1,5):
    for clf in range(1,4):
      for dataset in range(1,6):
        for lratio in range(5,25,5):
          for diff_iter_quer in range (1,7):
            if qs!=3 and dataset!=4 and diff_iter_quer!=1 and diff_iter_quer!=2 and diff_iter_quer!=4 and diff_iter_quer!=6:
              new_kfold_cross=KFoldCross(10,dataset)
              if diff_iter_quer==1:
                filename=new_kfold_cross.custom_run(strategy_value=qs,clf_value=clf,n_iterations=100,n_queries=1,percentage=lratio)
                if path.exists('/content/drive/MyDrive/multiresults/'+filename):
                  print("It exists!")
                else:
                  new_kfold_cross.perform_validation()
              elif diff_iter_quer==2:
                filename=new_kfold_cross.custom_run(strategy_value=qs,clf_value=clf,n_iterations=10,n_queries=10,percentage=lratio)
                if path.exists('/content/drive/MyDrive/multiresults/'+filename):
                  print("It exists!")
                else:
                  new_kfold_cross.perform_validation()
              elif diff_iter_quer==3:
                filename=new_kfold_cross.custom_run(strategy_value=qs,clf_value=clf,n_iterations=10,n_queries=20,percentage=lratio)
                if path.exists('/content/drive/MyDrive/multiresults/'+filename):
                  print("It exists!")
                else:
                  new_kfold_cross.perform_validation()
              elif diff_iter_quer==4:
                filename=new_kfold_cross.custom_run(strategy_value=qs,clf_value=clf,n_iterations=5,n_queries=20,percentage=lratio)
                if path.exists('/content/drive/MyDrive/multiresults/'+filename):
                  print("It exists!")
                else:
                  new_kfold_cross.perform_validation()
              elif diff_iter_quer==5:
                filename=new_kfold_cross.custom_run(strategy_value=qs,clf_value=clf,n_iterations=0,n_queries=0,percentage=lratio)
                if path.exists('/content/drive/MyDrive/multiresults/'+filename):
                    print("It exists and its adaptive!")
                else:
                  new_kfold_cross.perform_validation()
              elif diff_iter_quer==6:
                filename=new_kfold_cross.custom_run(strategy_value=qs,clf_value=clf,n_iterations=1,n_queries=0,percentage=lratio)
                if path.exists('/content/drive/MyDrive/multiresults/'+filename):
                  print("It exists!")
                else:
                  new_kfold_cross.perform_validation()
            if dataset!=4 and qs!=3:
             cases+=1

  print("All experiments where successful!Total cases run: ",cases)

#complete_experiment()