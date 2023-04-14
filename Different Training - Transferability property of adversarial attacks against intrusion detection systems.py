# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:07:06 2021

@author: Adam
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:45:26 2020

@author: Adam
"""
#%%

import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, CarliniL2Method,CarliniLInfMethod 
from art.attacks.evasion import BasicIterativeMethod,ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%%

def accuracyPerso(classifier, X_test_, Y_test):
    
    Y_pred=np.argmax(classifier.predict(X_test_), axis=1)
    label = np.argmax(Y_test, axis = 1)
    nb_correct_pred = np.sum(Y_pred == label)
    #print('\nAccuracy of the model on test data: {:4.2f}%'.format(nb_correct_pred/label.shape[0] * 100))
    
    from sklearn.metrics import precision_recall_fscore_support as score
    from sklearn.metrics import confusion_matrix
    
    precision, recall, fscore, support = score(label, Y_pred)
    CM=confusion_matrix(label, Y_pred)
    
    print('confusion_matrix:\n {}'.format(CM))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return nb_correct_pred, nb_correct_pred/label.shape[0]
#%%

def accuracyPerso_Linear(classifier, X_test_, Y_test):
    
    Y_pred=classifier.predict(X_test_)
    label = np.argmax(Y_test, axis = 1)
    nb_correct_pred = np.sum(Y_pred == label)
    #print('\nAccuracy of the model on test data: {:4.2f}%'.format(nb_correct_pred/label.shape[0] * 100))
    
    from sklearn.metrics import precision_recall_fscore_support as score
    from sklearn.metrics import confusion_matrix
    
    precision, recall, fscore, support = score(label, Y_pred)
    CM=confusion_matrix(label, Y_pred)
    
    print('confusion_matrix:\n {}'.format(CM))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return nb_correct_pred, nb_correct_pred/label.shape[0]
#%%
filename = "D:/Python/Spyder/NSL KDD dataset/KDDTrain+.dat"
df = pd.read_csv(filename,low_memory=False,header=None)


n_features = 41
n_classes = 2

X = df[df.columns[:-1]] 
# 0 non attack ----- 1 attack
y = df[df.columns[-1]]-1 # instead of 1 and 2 --> 0 normal , 1 attack

# One hot encoding
yHEC= tf.keras.utils.to_categorical(y, n_classes)


# scaling is importance for convergence of the neural network
scaler = StandardScaler() # Scale data to have mean 0 and variance 1 
#scaler = MinMaxScaler()#scale between 0 and 1 

X_scaled = scaler.fit_transform(X)
# =============================================================================
# #saving a normalized version of the data as a csv file
# data_scaled = scaler.fit_transform(data)
# np.savetxt("data_scaled.csv", data_scaled, delimiter=",")
# =============================================================================
#%%
eps_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]#, 1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9, 2]
clf_list_names=['classifier', 'DecisionTree_Classifier','SVM_Classifier','LogisticRegression_clf','RandomForest_clf','LinearDiscriminantAnalysis_clf']
NumberOfClassifers=len(clf_list_names)+1 # ensemble classifier +1
NumberOfTestesClassifers= len(clf_list_names)-1 # ensemble classifier and DNN are removed : important


nb_correct_attack_fgsm_global = np.zeros((NumberOfClassifers,len(eps_range))) 
nb_correct_attack_pgd_global = np.zeros((NumberOfClassifers,len(eps_range)))

rangeN=1
for g in range(rangeN):
    
    # Split the data set into training and testing
    X_train_Init, X_test, Y_train_Init, Y_test = train_test_split(
        X_scaled, yHEC, test_size=0.2, random_state=2)
    
    X_train, X_train_DNN, Y_train, Y_train_DNN = train_test_split(
        X_train_Init, Y_train_Init, test_size=0.5, random_state=2)
    
    
    
    model = tf.keras.models.Sequential(
        [   
            tf.keras.layers.Dense(n_features, activation=tf.nn.relu,input_dim = n_features),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)
        ]
    )
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    
    
    model.fit(X_train_DNN, Y_train_DNN, epochs=50,verbose=1)
    classifier = KerasClassifier( model=model) 
    
    
    
    # =============================================================================
    # x_test_pred = np.argmax(classifier.predict(X_test), axis=1)
    # nb_correct_pred = np.sum(x_test_pred == np.argmax(Y_test, axis=1))
    # 
    # print("Original test data :")
    # print("Correctly classified: {}".format(nb_correct_pred))
    # print("Incorrectly classified: {}".format(Y_test.shape[0]-nb_correct_pred))
    # 
    # 
    # =============================================================================
    
    
    Y_train_Linear = np.argmax(Y_train, axis = 1)
    Y_test_Linear = np.argmax(Y_test, axis=1)
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    LinearDiscriminantAnalysis_clf = LinearDiscriminantAnalysis()
    LinearDiscriminantAnalysis_clf.fit(X_train, Y_train_Linear)
    
    
    from sklearn.ensemble import RandomForestClassifier
    RandomForest_clf = RandomForestClassifier(random_state=0)
    RandomForest_clf.fit(X_train, Y_train_Linear)
    
    from sklearn.linear_model import LogisticRegression
    LogisticRegression_clf = LogisticRegression(random_state=0)
    LogisticRegression_clf.fit(X_train, Y_train_Linear)
    
    
    from sklearn.svm import SVC
    SVM_Classifier = SVC(gamma='auto')
    SVM_Classifier.fit(X_train,Y_train_Linear)
    
    
    from sklearn.tree import DecisionTreeClassifier
    DecisionTree_Classifier = DecisionTreeClassifier()
    DecisionTree_Classifier.fit(X_train,Y_train_Linear)
 
    
    attack_fgsm = FastGradientMethod(estimator=classifier)
    attack_pgd = ProjectedGradientDescent(estimator=classifier)
     

    
    nb_correct_attack_fgsm = np.zeros((NumberOfClassifers,len(eps_range))) 
    nb_correct_attack_pgd = np.zeros((NumberOfClassifers,len(eps_range))) 
    clf_list=[classifier, DecisionTree_Classifier,SVM_Classifier,LogisticRegression_clf,RandomForest_clf,LinearDiscriminantAnalysis_clf]
    #the case where there is NO ATTACK ( eps = 0 )
    tmp = np.zeros((Y_test.shape[0]))
    for j, clf in enumerate(clf_list):
        if j!=0:
            nb_correct_attack_fgsm[j,0] = np.sum(clf.predict(X_test) == Y_test_Linear)/Y_test.shape[0]
            nb_correct_attack_pgd[j,0] = nb_correct_attack_fgsm[j,0]
            tmp=tmp+clf.predict(X_test)
    ensemble_clf_prediction=[0 if x<NumberOfTestesClassifers/2 else 1 for x in tmp] #majority vote
    nb_correct_attack_fgsm[-1,0] = np.sum(ensemble_clf_prediction == Y_test_Linear)/Y_test.shape[0]
    nb_correct_attack_pgd[-1,0] = nb_correct_attack_fgsm[-1,0]
    
    #the case where there is an attack (eps != 0)
    for i,eps in enumerate(eps_range):
        if i!=0:
            print("eps=",eps,"   g =",g )
            attack_fgsm.set_params(**{'eps': eps})
            attack_pgd.set_params(**{'eps': eps})
            X_test_adv_fgsm = attack_fgsm.generate(X_test)
            X_test_adv_pgd = attack_pgd.generate(X_test)
            
            tmp = np.zeros((Y_test.shape[0],2))
            for j, clf in enumerate(clf_list):
                if j!=0:
                    nb_correct_attack_fgsm[j,i] = np.sum(clf.predict(X_test_adv_fgsm) == Y_test_Linear)/Y_test.shape[0]
                    tmp[:,0]=tmp[:,0]+ clf.predict(X_test_adv_fgsm)
                    nb_correct_attack_pgd[j,i] = np.sum(clf.predict(X_test_adv_pgd) == Y_test_Linear)/Y_test.shape[0]
                    tmp[:,1]=tmp[:,1]+ clf.predict(X_test_adv_pgd)
                
            ensemble_clf_prediction=[0 if x<3 else 1 for x in tmp[:,0]]
            nb_correct_attack_fgsm[-1,i] = np.sum(ensemble_clf_prediction == Y_test_Linear)/Y_test.shape[0]
            ensemble_clf_prediction=[0 if x<3 else 1 for x in tmp[:,1]]
            nb_correct_attack_pgd[-1,i] = np.sum(ensemble_clf_prediction == Y_test_Linear)/Y_test.shape[0] 
            

    nb_correct_attack_fgsm_global = nb_correct_attack_fgsm_global + nb_correct_attack_fgsm
    nb_correct_attack_pgd_global =  nb_correct_attack_pgd_global +  nb_correct_attack_pgd  

nb_correct_attack_fgsm_global = nb_correct_attack_fgsm_global/rangeN
nb_correct_attack_pgd_global =  nb_correct_attack_pgd_global/rangeN

#%%


#Resultat de code NSD-KDD adversarial attack and training.py
nb_correct_attack_fgsm_global[0,:]=[	0.996070649,	0.762214725,	0.37892439,	0.138837071,	0.139353046,	0.140107164,	0.140504068,	0.141774162,	0.1416154,	0.141139115,	0.141297877]#,	0.141337567,	0.1416154,	0.141774162,	0.141813852,	0.142250447,	0.142528279,	0.14256797,	0.142647351,	0.142687041,	0.143123636]
    
nb_correct_attack_pgd_global[0,:]=[	0.996070649,	0.702877555,	0.214090097,	0.089343124,	0.088787458,	0.088589006,	0.088509625,	0.088509625,	0.088509625,	0.088509625,	0.088509625]#,	0.088509625,	0.088509625,	0.088509625,	0.088509625,	0.088509625,	0.088509625,	0.088509625,	0.088509625,	0.088509625,	0.088509625]

    
#%%
eps_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]#, 1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9, 2]
clf_list_name =['DNN', 'Decision Tree', 'SVM','Logistic Regression','Random Forest', 'Linear Discriminant Analysis','ensemble']

import matplotlib.pyplot as plt
for c,clf_name in enumerate(clf_list_name):
    
    
    # ax.plot(np.array(eps_range), np.array(nb_correct_original[r,:]), 'g--', label='Clean test data')
    
    
    fig, ax = plt.subplots()
    ax.plot(np.array(eps_range), np.array(nb_correct_attack_fgsm_global[c,:]), 'r', label= 'Transferability of FGSM \n to '+str(clf_name))
    ax.plot(np.array(eps_range), np.array(nb_correct_attack_fgsm_global[0,:]), 'r--', label= 'FGSM against DNN [ref]')
    ax.plot(np.array(eps_range), np.array(nb_correct_attack_pgd_global[c,:]), 'b', label='Transferability of PGD\n to '+str(clf_name))
    ax.plot(np.array(eps_range), np.array(nb_correct_attack_pgd_global[0,:]), 'b--', label='PGD against DNN [ref]')
        
    # ax.plot(np.array(eps_range), np.array(nb_correct_attack_pgd_std), 'k--', label='PGD attack \nwithout adv training')
    legend = ax.legend(loc='best', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#00FFCC')
    ax.legend(bbox_to_anchor=(1, 1))
    title_name='Transferability of Adversarial Attacks between DNN and '+str(clf_name)
    plt.title(title_name)
    plt.xlabel(r"Attack strength ($\epsilon_{attack}$)")
    plt.ylabel('Accuracy')
    plt.show()
    
#%%
#print(RandomForest_clf.predict_proba(X_test_adv_fgsm[0:5,:]))

#%%
# =============================================================================
# #Create a ART FastGradientMethod attack.
# attack_fgsm = FastGradientMethod(estimator=classifier, eps=10 )
# 
# X_test_adv = attack_fgsm.generate(X_test)
# 
# #loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
# nb_correct_pred_fgsm,accuracy_test_fgsm=accuracyPerso(classifier, X_test_adv, Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data: {:4.2f}%'.format(accuracy_test_fgsm * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# #%%
# 
# #Create a ART ProjectedGradientDescent attack.
# #ProjectedGradientDescent
# attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=10)#, eps_step=0.01, max_iter=250)
# 
# X_test_adv =attack_pgd.generate(X_test)
# 
# #loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
# nb_correct_pred_pgd,accuracy_test_pgd=accuracyPerso(classifier, X_test_adv, Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data: {:4.2f}%'.format(accuracy_test_pgd * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# =============================================================================
#%%
# =============================================================================
# 
# #DecisionTreeClassifier
# 
# #  X_train , X_test , Y_train , Y_test
# 
# from sklearn.tree import DecisionTreeClassifier
# 
# DecisionTree_Classifier = DecisionTreeClassifier()
# DecisionTree_Classifier.fit(X_train,Y_train)
# 
# 
# 
# nb_correct_pred,accuracy_test=accuracyPerso(DecisionTree_Classifier, X_test, Y_test)
# 
# print('\nAccuracy on CLEAN test data: {:4.2f}%'.format(accuracy_test * 100))
# 
# 
# 
# #FGSM ATTACK
# attack_fgsm = FastGradientMethod(estimator=classifier, eps=1 )# DNN Classifier
# 
# X_test_adv_fgsm = attack_fgsm.generate(X_test)
# 
# nb_correct_pred_fgsm,accuracy_test_fgsm=accuracyPerso(DecisionTree_Classifier, X_test_adv_fgsm, Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data - FGSM ATTACK: {:4.2f}%'.format(accuracy_test_fgsm * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# 
# 
# #PGD ATTACK
# attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=1)#, eps_step=0.01, max_iter=250)
# 
# X_test_adv_pgd =attack_pgd.generate(X_test)
# 
# #loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
# nb_correct_pred_pgd,accuracy_test_pgd=accuracyPerso(DecisionTree_Classifier, X_test_adv_pgd, Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data - PGD ATTACK: {:4.2f}%'.format(accuracy_test_pgd * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# 
# 
# 
# #%%
# 
# # SVM Classifier
# 
# #  X_train , X_test , Y_train , Y_test
# 
# 
# from sklearn.svm import SVC
# 
# SVM_Classifier = SVC(gamma='auto')
# SVM_Classifier.fit(X_train,np.argmax(Y_train, axis = 1))
# 
# 
# 
# nb_correct_pred,accuracy_test=accuracyPerso_Linear(SVM_Classifier, X_test, Y_test)
# 
# print('\nAccuracy on CLEAN test data: {:4.2f}%'.format(accuracy_test * 100))
# 
# 
# #FGSM ATTACK
# attack_fgsm = FastGradientMethod(estimator=classifier, eps=1 )# DNN Classifier
# 
# X_test_adv_fgsm = attack_fgsm.generate(X_test)
# 
# nb_correct_pred_fgsm,accuracy_test_fgsm=accuracyPerso_Linear(SVM_Classifier, X_test_adv_fgsm,Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data - FGSM ATTACK: {:4.2f}%'.format(accuracy_test_fgsm * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# 
# 
# #PGD ATTACK
# attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=1)#, eps_step=0.01, max_iter=250)
# 
# X_test_adv_pgd =attack_pgd.generate(X_test)
# 
# #loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
# nb_correct_pred_pgd,accuracy_test_pgd=accuracyPerso_Linear(SVM_Classifier, X_test_adv_pgd, Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data - PGD ATTACK: {:4.2f}%'.format(accuracy_test_pgd * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# 
# #%%
# 
# # Logistic Regression Classifier
# 
# #  X_train , X_test , Y_train , Y_test
# 
# from sklearn.linear_model import LogisticRegression
# 
# 
# # we create an instance of LogisticRegression Classifier and fit the data.
# LogisticRegression_clf = LogisticRegression(random_state=0)
# LogisticRegression_clf.fit(X_train, np.argmax(Y_train, axis = 1))
# 
# 
# 
# 
# 
# nb_correct_pred,accuracy_test=accuracyPerso_Linear(LogisticRegression_clf, X_test, Y_test)
# 
# print('\nAccuracy on CLEAN test data: {:4.2f}%'.format(accuracy_test * 100))
# 
# 
# #FGSM ATTACK
# attack_fgsm = FastGradientMethod(estimator=classifier, eps=1 )# DNN Classifier
# 
# X_test_adv_fgsm = attack_fgsm.generate(X_test)
# 
# nb_correct_pred_fgsm,accuracy_test_fgsm=accuracyPerso_Linear(LogisticRegression_clf, X_test_adv_fgsm,Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data - FGSM ATTACK: {:4.2f}%'.format(accuracy_test_fgsm * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# 
# 
# #PGD ATTACK
# attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=1)#, eps_step=0.01, max_iter=250)
# 
# X_test_adv_pgd =attack_pgd.generate(X_test)
# 
# #loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
# nb_correct_pred_pgd,accuracy_test_pgd=accuracyPerso_Linear(LogisticRegression_clf, X_test_adv_pgd, Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data - PGD ATTACK: {:4.2f}%'.format(accuracy_test_pgd * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# 
# #%%
# 
# # Random Forest Classifier
# 
# #  X_train , X_test , Y_train , Y_test
# 
# from sklearn.ensemble import RandomForestClassifier
# 
# 
# # we create an instance of LogisticRegression Classifier and fit the data.
# RandomForest_clf = RandomForestClassifier(random_state=0)
# RandomForest_clf.fit(X_train, np.argmax(Y_train, axis = 1))
# #The default values for the parameters controlling the size of the trees (e.g. max_depth, min_samples_leaf, etc.) 
# #lead to fully grown and unpruned trees which can potentially be very large on some data sets. 
# #To reduce memory consumption, the complexity and size of the trees should be controlled by setting those parameter values.
# 
# 
# 
# 
# nb_correct_pred,accuracy_test=accuracyPerso_Linear(RandomForest_clf, X_test, Y_test)
# 
# print('\nAccuracy on CLEAN test data: {:4.2f}%'.format(accuracy_test * 100))
# 
# 
# #FGSM ATTACK
# attack_fgsm = FastGradientMethod(estimator=classifier, eps=1 )# DNN Classifier
# 
# X_test_adv_fgsm = attack_fgsm.generate(X_test)
# 
# nb_correct_pred_fgsm,accuracy_test_fgsm=accuracyPerso_Linear(RandomForest_clf, X_test_adv_fgsm,Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data - FGSM ATTACK: {:4.2f}%'.format(accuracy_test_fgsm * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# 
# 
# #PGD ATTACK
# attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=1)#, eps_step=0.01, max_iter=250)
# 
# X_test_adv_pgd =attack_pgd.generate(X_test)
# 
# #loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
# nb_correct_pred_pgd,accuracy_test_pgd=accuracyPerso_Linear(RandomForest_clf, X_test_adv_pgd, Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data - PGD ATTACK: {:4.2f}%'.format(accuracy_test_pgd * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# 
# 
# #%%
# 
# # Linear Discriminant Analysis Classifier
# 
# #  X_train , X_test , Y_train , Y_test
# 
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# #from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# 
# # we create an instance of LogisticRegression Classifier and fit the data.
# LinearDiscriminantAnalysis_clf = LinearDiscriminantAnalysis()
# LinearDiscriminantAnalysis_clf.fit(X_train, np.argmax(Y_train, axis = 1))
# 
# 
# nb_correct_pred,accuracy_test=accuracyPerso_Linear(LinearDiscriminantAnalysis_clf, X_test, Y_test)
# 
# print('\nAccuracy on CLEAN test data: {:4.2f}%'.format(accuracy_test * 100))
# 
# 
# #FGSM ATTACK
# attack_fgsm = FastGradientMethod(estimator=classifier, eps=1 )# DNN Classifier
# 
# X_test_adv_fgsm = attack_fgsm.generate(X_test)
# 
# nb_correct_pred_fgsm,accuracy_test_fgsm=accuracyPerso_Linear(LinearDiscriminantAnalysis_clf, X_test_adv_fgsm,Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data - FGSM ATTACK: {:4.2f}%'.format(accuracy_test_fgsm * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# 
# 
# #PGD ATTACK
# attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=1)#, eps_step=0.01, max_iter=250)
# 
# X_test_adv_pgd =attack_pgd.generate(X_test)
# 
# #loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
# nb_correct_pred_pgd,accuracy_test_pgd=accuracyPerso_Linear(LinearDiscriminantAnalysis_clf, X_test_adv_pgd, Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data - PGD ATTACK: {:4.2f}%'.format(accuracy_test_pgd * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# =============================================================================