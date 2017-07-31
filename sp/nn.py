# -*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
import sys
from datetime import datetime

train_prob = 0.95

print("Read train data")
train_df = pd.read_csv("../data/num_all.pd",sep='\t').sample(frac=1).reset_index(drop=True)

train_idx = int(train_df.shape[0]*train_prob)

x_train = train_df.drop(['target'], axis=1).loc[:train_idx]
y_train = train_df.target.loc[:train_idx]
x_test = train_df.drop(['target'], axis=1).loc[train_idx:]
y_test = train_df.target.loc[train_idx:]
y_last = train_df['last'].loc[train_idx:]
y_log10 = train_df.log10.loc[train_idx:]
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
print('Shape train label: {}\nShape test label: {}\nShape test last: {}'.format(y_train.shape, y_test.shape, y_last.shape))

y_mean = np.mean(y_train)

split = int(x_train.shape[0]*0.95)
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]


d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)


params = {}
params['learning_rate'] = 0.05
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'
params['sub_feature'] = 0.9
params['num_leaves'] = 128
params['is_unbalance'] = 'true'
params['min_data'] = 100
params['min_hessian'] = 1
params['early_stopping_round'] = 100

#params['bagging_freq'] = 50
#params['bagging_fraction'] = 0.8

print("Start training")
watchlist = [d_valid]
clf = lgb.train(params,d_train,2500,watchlist)
# clf2 = lgb.Booster(model_file='test.model')

print("Make prediction")
#clf.reset_parameter({"num_threads":4})
p_test = clf.predict(x_test)

print("Start correction")
if p_test.shape[0]!=y_test.shape[0]:
    print("WARNING!!!\nThe prediction should match real label")
    sys.exit()

#if p_test[i] > 50 and abs(p_test[i]-y_test[i])>y_test[i]*1:

k = 0
'''
for i in xrange(p_test.shape[0]):
    multi = 1 if y_log10.iloc[i]<2.7 else 2.7/y_log10.iloc[i] # log10(500)=2.7
    if abs(p_test[i]-y_test.iloc[i])>abs(y_test.iloc[i])*multi:
        k += 1
        p_test[i] = (p_test[i]+y_last.iloc[i])/2 + 0.1*np.random.random()-0.05

for i in xrange(p_test.shape[0]):
    multi = 1 if y_log10.iloc[i]<2.7 else 2.7/y_log10.iloc[i] # log10(500)=2.7
    times = abs(p_test[i]-y_test.iloc[i])/abs(y_test.iloc[i]+1e-20)
    if times > multi:
        k += 1
        p_test[i] = (p_test[i]+y_last.iloc[i]*times)/(times+1) + 0.1*np.random.random()-0.05

for i in xrange(p_test.shape[0]):
    k += 1
    times = abs(p_test[i]-y_last.iloc[i])/abs(y_last.iloc[i]+1e-20)
    p_test[i] = (p_test[i]+y_last.iloc[i])/2 + 0.1*np.random.random()-0.05
'''
print("{}/{}".format(k,p_test.shape[0]))


mse = (np.sqrt((p_test - y_test)**2)).sum()/p_test.shape[0]

compare_mse = (np.sqrt((y_last - y_test)**2)).sum()/p_test.shape[0]

print("Test mse is: {}\tCompareson mse is: {}".format(mse,compare_mse))

file_name = "lightgbm_{}_{}".format(datetime.now().strftime('%Y%m%d_%H%M%S'), int(100000*mse))

print("Save model")
clf.save_model('../model/'+file_name+'.model')

print("Generate data")
output = pd.DataFrame({'goods_id': x_test.goods_id,'channel_id':x_test.channel_id,'year':x_test.year,
            'weeknum':x_test.weeknum,'thisweek':x_test.value,'real':y_test,'prediction': p_test})

print("Save result as: "+file_name+'.result')
output.to_csv('../result/'+file_name+'.result', index=False, float_format='%.4f')
