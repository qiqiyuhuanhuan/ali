import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# 数据加载
train = pd.read_csv('C:/Users/小胖墩/Desktop/train.csv')
test = pd.read_csv('C:/Users/小胖墩/Desktop/test.csv')

# 特征编码
# 使用LabelEncoder对'job'列进行编码
job_le = LabelEncoder()
train['job'] = job_le.fit_transform(train['job'])
test['job'] = job_le.transform(test['job'])  # 使用同一个LabelEncoder对象对测试集编码

# 使用map方法对'marital'列进行编码
train['marital'] = train['marital'].map({'unknown': 0, 'single': 1, 'married': 2, 'divorced': 3})
test['marital'] = test['marital'].map({'unknown': 0, 'single': 1, 'married': 2, 'divorced': 3})

# 使用map方法对'education'列进行编码
train['education'] = train['education'].map({'unknown': 0, 'illiterate': 1, 'basic.4y': 2, 'basic.6y': 3,
                                             'basic.9y': 4, 'high.school': 5, 'university.degree': 6,
                                             'professional.course': 7})
test['education'] = test['education'].map({'unknown': 0, 'illiterate': 1, 'basic.4y': 2, 'basic.6y': 3,
                                           'basic.9y': 4, 'high.school': 5, 'university.degree': 6,
                                           'professional.course': 7})

# 使用map方法对'housing'列进行编码
train['housing'] = train['housing'].map({'unknown': 0, 'no': 1, 'yes': 2})
test['housing'] = test['housing'].map({'unknown': 0, 'no': 1, 'yes': 2})

# 使用map方法对'loan'列进行编码
train['loan'] = train['loan'].map({'unknown': 0, 'no': 1, 'yes': 2})
test['loan'] = test['loan'].map({'unknown': 0, 'no': 1, 'yes': 2})

# 使用map方法对'contact'列进行编码
train['contact'] = train['contact'].map({'cellular': 0, 'telephone': 1})
test['contact'] = test['contact'].map({'cellular': 0, 'telephone': 1})

# 使用map方法对'day_of_week'列进行编码
train['day_of_week'] = train['day_of_week'].map({'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4})
test['day_of_week'] = test['day_of_week'].map({'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4})

# 使用map方法对'poutcome'列进行编码
train['poutcome'] = train['poutcome'].map({'nonexistent': 0, 'failure': 1, 'success': 2})
test['poutcome'] = test['poutcome'].map({'nonexistent': 0, 'failure': 1, 'success': 2})

# 使用map方法对'default'列进行编码
train['default'] = train['default'].map({'unknown': 0, 'no': 1, 'yes': 2})
test['default'] = test['default'].map({'unknown': 0, 'no': 1, 'yes': 2})

# 使用map方法对'month'列进行编码
train['month'] = train['month'].map({'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                                     'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})
test['month'] = test['month'].map({'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                                   'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})

# 使用map方法对'subscribe'列进行编码
train['subscribe'] = train['subscribe'].map({'no': 0, 'yes': 1})

# 去掉训练集和测试集中的'id'列，保存测试集的id
y_id = test['id']
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

# 使用LightGBM进行模型训练
model_lgb = lgb.LGBMClassifier(
            num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='binary',
            max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=2022,
            n_estimators=2000, subsample=1, colsample_bytree=1,
        )
model_lgb.fit(train.drop('subscribe', axis=1), train['subscribe'])  # 在训练集上训练模型

# 在测试集上进行预测
y_pred = model_lgb.predict(test)

# 将预测结果映射回'yes'和'no'
result_map = {0: 'no', 1: 'yes'}
result = pd.DataFrame({'id': y_id, 'subscribe': y_pred.astype(np.int32)})
result['subscribe'] = result['subscribe'].map(result_map)

# 将结果保存为CSV文件
result.to_csv("predict1.csv", index=False)

# 打印预测结果的数量统计
print("Prediction Counts:")
print(result['subscribe'].value_counts())
