# -*- coding: utf-8 -*-
"""
@author:
"""

import numpy as np
import pandas as pd
# import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale, PolynomialFeatures, MinMaxScaler

# cross_val_score：交叉检验
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, KFold, StratifiedKFold, \
    ShuffleSplit, cross_val_score
from keras.models import load_model
from sklearn.decomposition import PCA
# 指标算法
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ARDRegression, BayesianRidge, RidgeCV, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # 集成算法，随机森林
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, PReLU
from keras.layers import Input, add, multiply, maximum
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers  # 正则化


# 构建机器学习模型
def Model_1(train_x, train_y):
    '''
    输入：训练数据的特征数据，和标签数据
    输出：一个深度神经网络模型
    :param train_x:
    :param train_y:
    :return:
    '''
    # 建立一个三层的神经网络（不带输出层）
    model = Sequential()
    model.add(Dense(500, input_shape=(train_x.shape[1],)))
    model.add(Activation('sigmoid'))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('tanh'))

    # 输出层
    model.add(Dense(1))
    model.add(Activation('linear'))

    # 三种优化器：SGD Adam RMSprop
    # 用于配置训练模型
    model.compile(optimizer='sgd',
                  loss='mean_squared_error')

    '''
    ReduceLROnPlateau
    monitor:被监测的数据
    factor：学习速率被降低的因素。新的学习速率 = 学习速率 * 因数
    patience：没有进步的训练轮数，在这之后训练速率会被降低。
    verbose:整数。0：安静，1：更新信息。
    mode： {auto, min, max} 其中之一。如果是 min 模式，学习速率会被降低如果被监测的数据已经停止下降； 
                                     在 max 模式，学习塑料会被降低如果被监测的数据已经停止上升； 
                                     在 auto 模式，方向会被从被监测的数据中自动推断出来
    min_delta: 对于测量新的最优化的阀值，只关注巨大的改变。
    cooldown:在学习速率被降低之后，重新恢复正常操作之前等待的训练轮数量。
    min_lr:学习速率的下边界。
    '''
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                                  patience=10, verbose=0,
                                  mode='auto', min_delta=0.001,
                                  cooldown=0, min_lr=0)
    epochs = 50  # 迭代次数
    model.fit(train_x, train_y, epochs=epochs,
              batch_size=20, validation_split=0.0,
              callbacks=[reduce_lr],
              verbose=0)
    return model


def kfold_loss(df_x, df_y):
    '''
    输入：特征数据，和标签数据(dataframe类型的)
    输出：利用交叉验证划分数据，得到mean_loss
    :param df_x:
    :param df_y:
    :return:
    '''
    loss_list = []
    df_x = pd.DataFrame(df_x, index=None)
    df_y = pd.DataFrame(df_y, index=None)
    sfloder = KFold(n_splits=5, shuffle=False)

    for train_id, test_id in sfloder.split(df_x, df_y):
        model = Model_1(df_x.iloc[train_id], df_y.iloc[train_id])
        loss = model.evaluate(df_x.iloc[test_id], df_y.iloc[test_id], verbose=0)
        loss_list.append(loss)
    return np.array(loss_list).mean()


# def HeatGrape(df):
#     '''
#     输入dataframe数据
#     输出特征之间的热力相关图
#     :param df:
#     :return:
#     '''
#     # 找出相关程度
#     plt.figure(figsize=(20, 16))  # 指定绘图对象宽度和高度
#     colnm = df.columns.tolist()  # 列表头
#     mcorr = df[colnm].corr()  # 相关系数矩阵，即给出了任意两个变量之间的相关系数
#     mask = np.zeros_like(mcorr, dtype=np.bool)  # 构造与mcorr同维数矩阵 为bool型
#     mask[np.triu_indices_from(mask)] = True  # 角分线右侧为True
#     cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matplotlib colormap对象
#     g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
#     plt.show()


# 创建多项式模型的函数
def Polynomial_model(degree=1):
    '''
    输入：一个维度
    输出：一个对应维度的多项式机器学习模型
    :param degree:
    :return:
    '''
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    return pipeline


# 画出学习曲线->判断过拟合和欠拟合
def plot_learning_curve(estimator, x, y):
    '''
    输入：model, x, y
    输出：画出学习曲线，来判断是否过拟合或者欠拟合
    :param estimator:
    :param x:
    :param y:
    :return:
    '''
    train_sizes = np.linspace(.1, 1.0, 5)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plt.figure()
    plt.title('learning curve')

    plt.xlabel("Training examples")
    plt.ylabel("score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_betweenx(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def Validation_curve_demo(x, y, model, param_name, param_range):
    '''
    输入：x，y, model, param_name, param_range
    输出：输出一个图像，用来选取最佳参数值
    :param x:
    :param y:
    :param model:
    :param param_name:
    :param param_range:
    :return:
    '''
    train_loss, test_loss = validation_curve(
        model, x, y, param_name=param_name,
        param_range=param_range, cv=5, scoring='neg_mean_squared_error')
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)

    plt.plot(param_range, train_loss_mean, 'o-', color='r',
             label='Training')
    plt.plot(param_range, test_loss_mean, 'o-', color='g',
             label='Cross-validation')

    plt.xlabel(param_name)
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()


def Get_feature_std(std, feature_name, limit_std):
    '''
    输入：方差，特征名称列表，限制方差
    输出：大于限制方差值的对应特征名称列表
    :param std:
    :param feature_name:
    :param limit_std:
    :return:
    '''
    feature = []
    for i, j in zip(std, feature_name):
        if i < limit_std:
            feature.append(j)

    return feature


def Pre_data_process(df_train, df_test):
    '''
    输入：训练特征数据，和测试特征数据
    输出：预处理之后的，训练特征数据，和测试特征数据
    :param df_train:
    :param df_test:
    :return:
    '''
    scale_column = ['V0', 'V1', 'V6', 'V30']
    # 训练数据和测试数据特征分布差异较大的feature
    drop_list_1 = ['V9', 'V17', 'V22', 'V28']

    mcorr = df_train.corr()
    # 和target相关性较小的feature
    drop_list_2 = [c for c in mcorr['target'].index
                   if abs(mcorr['target'][c]) < 0.15]
    # 方差较小的feature
    drop_list_3 = Get_feature_std(df_train.std(), df_train.columns, 0.6)

    drop_label = list(set(drop_list_3 + drop_list_2))
    df_train = df_train.drop(drop_label, axis=1)
    df_test = df_test.drop(drop_label, axis=1)

    # # 将部分特征数据标准
    # for column in scale_column:
    #     df_train[column] = scale(df_train[column])
    #     df_test[column] = scale(df_test[column])

    return (df_train, df_test)


def Cross_validation(x, y, model):
    '''
    输入：x, y, model
    输出：交叉验证后的误差均值
    :param x: train_x
    :param y: train_y
    :param model:
    :return:
    '''
    loss_list = cross_val_score(model, x, y, cv=5,
                                scoring='neg_mean_squared_error')
    return -loss_list.mean()


def Model_stack(df_train_x, df_train_y, df_test):
    # kernel has 'linear'/'poly'/'rbf'/'sigmoid'/'precomputed'/'callable' 如果没有给出，默认'rbf' callable 预先计算内核矩阵
    svr_ = SVR(kernel='linear', degree=3, coef0=0.0, tol=0.001,
               C=1.0, epsilon=0.1, shrinking=True, cache_size=20)
    lgb_ = lgb.LGBMModel(boosting_type='gbdt', num_leaves=35,
                         max_depth=20, max_bin=255, learning_rate=0.03, n_estimator=10, subsample_for_bin=2000,
                         objective='regression', min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20,
                         subsample=1.0, verbose=0, subsample_freq=1, colsample_bytree=1.0, reg_alpha=0.0,
                         reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True)
    RF_model = RandomForestRegressor(n_estimators=50, max_depth=25, min_samples_split=20, min_samples_leaf=10,
                                     max_features='sqrt', oob_score=True, random_state=10)
    # 贝叶斯岭回归
    BR_model = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True, fit_intercept=True,
                             lambda_1=1e-06, lambda_2=1e-06, n_iter=300, normalize=False, tol=0.0000001, verbose=False)
    linear_model = LinearRegression()
    ls = Lasso(alpha=0.00375)
    x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y,
                                                        test_size=0.6)
    rg = RidgeCV(cv=5)
    stack = pd.DataFrame()
    stack_test = pd.DataFrame()

    ls.fit(x_train, y_train)
    lgb_.fit(x_train, y_train)
    RF_model.fit(x_train, y_train)
    svr_.fit(x_train, y_train)
    linear_model.fit(x_train, y_train)
    BR_model.fit(x_train, y_train)

    stack['rf'] = ls.predict(x_test)
    stack['adaboost'] = lgb_.predict(x_test)
    stack['gbdt'] = RF_model.predict(x_test)
    stack['lightgbm'] = svr_.predict(x_test)
    stack['linear_model'] = linear_model.predict(x_test)
    stack['BR'] = BR_model.predict(x_test)
    # print('stacking_model: ',Cross_validation(stack, y_test, rg))

    rg.fit(stack, y_test)
    stack_test['rf'] = ls.predict(df_test)
    stack_test['adaboost'] = lgb_.predict(df_test)
    stack_test['gbdt'] = RF_model.predict(df_test)
    stack_test['lightgbm'] = svr_.predict(df_test)
    stack_test['linear_model'] = linear_model.predict(df_test)
    stack_test['BR'] = BR_model.predict(df_test)

    final_ans = rg.predict(stack_test)
    pd.DataFrame(final_ans).to_csv('predict_drop+3.txt', index=False, header=False)


# 数据降维(主成分分析法)
# n_components = 'mle' and svd_solver = 'full' to guess the dimension

df_train = pd.read_csv(r'C:\Users\93767\Desktop\python\Ali-steam-master\zhengqi_train.txt', sep="\t")  # 获取训练数据
df_test = pd.read_csv(r'C:\Users\93767\Desktop\python\Ali-steam-master\zhengqi_test.txt', sep="\t")  # 得到预测的数据

df_train, df_test = Pre_data_process(df_train, df_test)

df_train_x = df_train.drop(['target'], axis=1)
df_train_y = df_train['target']

# pca = PCA(n_components=0.95)
pca = PCA(n_components='mle', svd_solver='full')
pca.fit(df_train_x)
df_train_x = pca.transform(df_train_x)
print(r'PCA降维后特征数：', pd.DataFrame(df_train_x).shape[1])
df_test = pca.transform(df_test)

# LinearRegression
# line_R_model = LinearRegression()

ARD_model = ARDRegression(alpha_1=1e-06, alpha_2=1e-06,
                          compute_score=False, copy_X=True, fit_intercept=True,
                          lambda_1=1e-06, lambda_2=1e-06, n_iter=300, normalize=False, threshold_lambda=10000.0,
                          tol=0.001, verbose=False)
BR_model = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
                         copy_X=True, fit_intercept=True, lambda_1=1e-06,
                         lambda_2=1e-06, n_iter=30, normalize=False, tol=0.0000001,
                         verbose=True)
myGBR = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None, learning_rate=0.01, loss='huber',
                                  max_depth=14, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0,
                                  min_impurity_split=None, min_samples_leaf=10, min_samples_split=40,
                                  min_weight_fraction_leaf=0.0, n_estimators=300, presort='auto', random_state=10,
                                  subsample=0.8, verbose=0, warm_start=False
                                  )
RF_model = RandomForestRegressor(n_estimators=50, max_depth=25, min_samples_split=20, min_samples_leaf=10,
                                 max_features='sqrt', oob_score=True, random_state=10)

linear_model = LinearRegression()

# Model_stack(df_train_x, df_train_y, df_test)

# loss = kfold_loss(df_train_x, df_train_y)
# #print('keras_model : ', loss)
# #print('linearRegression: ',Cross_validation(df_train_x,df_train_y, linear_model))
# #print('BayesianRidge : ',Cross_validation(df_train_x,df_train_y, BR_model))
# #print('GradientBosting : ',Cross_validation(df_train_x,df_train_y, myGBR))
# #print('ARDRegression : ',Cross_validation(df_train_x,df_train_y, ARD_model))
# #print('RandomForest : ',Cross_validation(df_train_x,df_train_y, RF_model))

# 将最终预测结果保存到文件当中

# 贝叶斯岭回归 当数据有很多缺失或者矛盾的病态数据，可以考虑，对病态数据鲁棒性很高，也不用交叉验证选择超参数。
# 但是极大化似然函数的推断过程比较耗时，一般情况不推荐使用。 θ为球形分布，自由度一个
# BR_model.fit(df_train_x, df_train_y)
# final_ans = BR_model.predict(df_test)
# pd.DataFrame(final_ans).to_csv('______predict_BR_drop_3.txt', index=False, header=False)
# # θ为椭圆形高斯分布，自由度n个
# ARD_model.fit(df_train_x, df_train_y)
# final_ans = ARD_model.predict(df_test)
# pd.DataFrame(final_ans).to_csv('______predict_ARD_drop_3.txt', index=False, header=False)
# #
# myGBR.fit(df_train_x, df_train_y)
# final_ans = myGBR.predict(df_test)
# pd.DataFrame(final_ans).to_csv('______predict_myGBR_drop_3.txt', index=False, header=False)
