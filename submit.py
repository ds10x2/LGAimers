#-*- coding: utf-8 -*-

import pandas as pd
import random
import os
import numpy as np 

from sklearn.preprocessing import LabelEncoder #카테고리형 데이터를 수치형으로 변환
from sklearn.ensemble import GradientBoostingClassifier


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(37) # Seed 고정


#===데이터 로드===
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

train_x = train_df.drop(columns=['PRODUCT_ID', 'TIMESTAMP', 'Y_Class', 'Y_Quality']) #drop: 칼럼 제거
train_y = train_df['Y_Class'] #0: 적정 기준 미달, 1: 적합, 2: 적정 기준 초과

test_x = test_df.drop(columns=['PRODUCT_ID', 'TIMESTAMP'])

#===데이터 pre-processing===
train_x = train_x.fillna(0) #fillna(0)은 NaN를 0으로 대체
test_x = test_x.fillna(0)
# qualitative to quantitative
qual_col = ['LINE', 'PRODUCT_CODE']

for i in qual_col:
    #학습 데이터
    le = LabelEncoder()
    le = le.fit(train_x[i]) #fit하고 라벨 숫자로 transform(변환)
    train_x[i] = le.transform(train_x[i]) 
    
    #테스트 데이터
    for label in np.unique(test_x[i]):  #np.unique: 고유한 값만 남기고 정렬
        if label not in le.classes_: 
            le.classes_ = np.append(le.classes_, label) #라벨 추가
    test_x[i] = le.transform(test_x[i])
print('Done.')


#===Classification Model Fit=== 분류 모델 fit
RF = GradientBoostingClassifier(random_state=37).fit(train_x, train_y)
print('Done')




#==Inference=== 추론
preds = RF.predict(test_x)
print('Done.')


#===submit===
submit = pd.read_csv('./sample_submission.csv')
submit['Y_Class'] = preds
submit.to_csv('./result.csv', index=False)