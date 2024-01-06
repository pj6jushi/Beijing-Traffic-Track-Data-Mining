# _*_coding:utf_8 _*_
# @Time: 2023/11/21 2:23
# @Author: li zhuoyuan
# @File: main.py
# @Contact: 21377153@buaa.edu.cn
# @Software: PyCharm

import random
import numpy as np
import pandas as pd
import pytz
from datetime import datetime

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost.callback import EarlyStopping
import xgboost as xgb
import optuna
import warnings
import os

# 参数设置
kfold = 'kf'
n_splits = 10
n_reapts = 1
early_stopping_rounds = 300

# 随机种子
random_state = 42
random.seed(random_state)
random_state_list = random.sample(range(9999), n_reapts)

# 交叉验证分割类
class Splitter:
    def __init__(self, kfold='skf', n_splits=5, cat_df=pd.DataFrame(), test_size=0.3):
        self.n_splits = n_splits
        self.kfold = kfold
        self.cat_df = cat_df
        self.test_size = test_size

    def split_data(self, X, y, random_state_list):
        if self.kfold == 'skf':
            for random_state in random_state_list:
                kf = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, self.cat_df):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val, val_index
        else:  # kf
            for random_state in random_state_list:
                kf = KFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, self.cat_df):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val, val_index


def objective(trial):
    # 1. 定义模型的超参数
    param = {
        'objective': 'reg:squarederror',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        'subsample': trial.suggest_float('subsample', 0.1, 1),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 10.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'n_estimators': trial.suggest_int('n_estimators', 3000, 10000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20)
    }

    # 2. 使用这些参数训练模型并计算损失
    splitter = Splitter(kfold=kfold, n_splits=n_splits, cat_df=y)
    rmse_list = []
    for X_train_, X_val, y_train_, y_val, _ in splitter.split_data(X, y, random_state_list=[random_state]):
        dtrain = xgb.DMatrix(X_train_, label=y_train_)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        model = xgb.train(param, dtrain, num_boost_round=param['n_estimators'], evals=watchlist,
                          early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        preds = model.predict(dvalid)
        rmse_list.append(mean_squared_error(y_val, preds, squared=False))

    # 3. 返回平均 RMSE
    return np.mean(rmse_list)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pd.set_option('display.width', 1200)
    pd.set_option('display.max_columns', 30)

    if not os.path.exists('model'):
        os.makedirs('model')

    df = pd.read_csv('./data/data.csv')
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # train test split
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=random_state)

    continuous_cols = ['distance', 'start_x', 'start_y', 'target_x', 'target_y', 'start_speed', 'target_speed', 'hour']

    discrete_cols = ['holidays']

    target_cols = ['time']

    # 预测数据
    data = pd.read_csv('./data/eta_task.csv')
    data_predict = pd.read_csv('./data/data_predict.csv')
    # data_predict = pd.DataFrame(
    #     columns=['distance', 'start_x', 'start_y', 'target_x', 'target_y', 'start_speed', 'target_speed', 'holidays',
    #              'hour'], index=range(4400))
    # min_x = 116.24
    # max_x = 116.5
    # min_y = 39.75
    # max_y = 40.02
    # for i in range(len(data)):
    #     if i % 2 == 0:
    #         data_predict['distance'][i // 2] = data['current_dis'][i + 1]
    #         data_predict['start_x'][i // 2] = (float(data['coordinates'][i].strip('[]').split(',')[0]) - min_x) / (
    #                     max_x - min_x)
    #         data_predict['start_y'][i // 2] = (float(data['coordinates'][i].strip('[]').split(',')[1]) - min_y) / (
    #                     max_y - min_y)
    #         data_predict['target_x'][i // 2] = (float(data['coordinates'][i + 1].strip('[]').split(',')[0]) - min_x) / (
    #                     max_x - min_x)
    #         data_predict['target_y'][i // 2] = (float(data['coordinates'][i + 1].strip('[]').split(',')[1]) - min_y) / (
    #                     max_y - min_y)
    #         data_predict['start_speed'][i // 2] = data['speeds'][i]
    #         data_predict['target_speed'][i // 2] = data['speeds'][i + 1]
    #         data_predict['holidays'][i // 2] = data['holidays'][i]
    #         data_predict['hour'][i // 2] = int(data['time'][i][11:13])
    # data_predict.to_csv('./data/data_predict.csv', index=False)

    # 数据预处理
    # 最大最小归一化
    scaler = MinMaxScaler()
    scaler.fit(train_df[continuous_cols])
    train_df[continuous_cols] = scaler.transform(train_df[continuous_cols])
    test_df[continuous_cols] = scaler.transform(test_df[continuous_cols])
    data_predict[continuous_cols] = scaler.transform(data_predict[continuous_cols])


    X, y = train_df.drop(columns=target_cols, axis=1), train_df[target_cols]
    test_df1, test_result = test_df.drop(columns=target_cols, axis=1), test_df[target_cols]

    feature_importances_ = pd.DataFrame(index=X.columns)
    eval_results_ = {}
    oof = np.zeros((X.shape[0]))
    test_predss = np.zeros((test_df1.shape[0]))
    test_predss2 = np.zeros((data_predict.shape[0]))

    # 模型参数
    xgb_params = {
        'learning_rate': 0.0538076577345207,
        'colsample_bytree': 0.9625456191950508,
        'colsample_bylevel': 1,
        'subsample': 0.7276302931196216,
        'reg_alpha': 4.703147620659009e-05,
        'reg_lambda': 4.703333667593938,
        'max_depth': 11,
        'n_estimators': 30000,
        'min_child_weight': 18,
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'n_jobs': -1,
        'verbosity': 0
    }

    # # 创建一个研究并优化
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=100)  # 你可以调整 n_trials 的值
    #
    # # 获取最佳参数
    # print('Best Parameters:', study.best_params)

    # 多折交叉验证
    splitter = Splitter(kfold=kfold, n_splits=n_splits, cat_df=y)
    for i, (X_train_, X_val, y_train_, y_val, val_index) in \
            enumerate(splitter.split_data(X, y, random_state_list=[random_state])):
        fold = i % n_splits
        fit_set = xgb.DMatrix(X_train_, y_train_)
        val_set = xgb.DMatrix(X_val, y_val)
        watchlist = [(fit_set, 'fit'), (val_set, 'val')]
        # 存储验证集损失结果
        eval_results_[fold] = {}

        xgb_params['objective'] = 'reg:squarederror'

        # 加载模型
        # model = xgb.Booster()
        # model.load_model(f'./model/xgb_{fold}.json')

        # 训练模型
        model = xgb.train(
            num_boost_round=xgb_params['n_estimators'],
            params=xgb_params,
            dtrain=fit_set,
            evals=watchlist,
            evals_result=eval_results_[fold],
            verbose_eval=False,
            callbacks=[EarlyStopping(early_stopping_rounds, data_name='val', save_best=True)])

        # 保存模型
        model.save_model(f'./model/xgb_{fold}.json')

        # 验证集与测试集的预测结果
        val_preds = model.predict(val_set)
        test_predss += model.predict(xgb.DMatrix(test_df1)) / n_splits
        test_predss2 += model.predict(xgb.DMatrix(data_predict)) / n_splits
        # 保存验证集的预测结果
        oof[val_index] = val_preds

        # 计算评估指标，例如 RMSE
        val_score = mean_squared_error(y_val, val_preds, squared=False)
        best_iter = model.best_iteration
        print(f'Fold: {fold:>3}| RMSE: {val_score:.5f}' f' | Best iteration: {best_iter:>4}')

        # 存储特征重要度
        feature_importances_[f'gain_{fold}'] = feature_importances_.index.map(model.get_score(importance_type='gain'))
        feature_importances_[f'split_{fold}'] = feature_importances_.index.map(
            model.get_score(importance_type='weight'))

    # 查看整体评估指标
    mean_cv_score_full = mean_squared_error(y, oof, squared=False)
    print(f'{"*" * 50}\nMean RMSE : {mean_cv_score_full:.5f}')



    print('feature_importances_:')
    print(feature_importances_)

    # 结果对比
    # for i in range(len(test_result)):
    #     print(test_result.iloc[i]['time'], test_predss[i])
    for i in range(15):
        print(test_predss2[i])
    # 计算测试集的RMSE
    test_score = mean_squared_error(test_result, test_predss, squared=False)
    print(f'{"*" * 50}\nTest RMSE : {test_score:.5f}')


    for i in range(len(data)):
        if i % 2 == 1:
            time = int(test_predss2[i // 2])
            timestamp_dt = datetime.fromisoformat(data['time'].iloc[i - 1].replace("Z", "+00:00"))
            unix_timestamp = timestamp_dt.replace(tzinfo=pytz.UTC).timestamp()
            unix_timestamp += time
            # 再转换回去
            timestamp_dt = datetime.utcfromtimestamp(unix_timestamp)
            data['time'][i] = timestamp_dt.isoformat() + 'Z'
    print(data)
    data.to_csv('eta_task_pred.csv', index=False)


