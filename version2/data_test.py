import scanpy as sc

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import random


def dataread_sciplex_test(file,inputsize=20000):
    adata = sc.read_h5ad(filename=file)

    # 查看数据
    # print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # X = (X-X.mean())/X.std()
    # X = (X - X.min()) / (X.max() - X.min())
    # print(adata.to_df())

    # print(adata.obs)
    df = adata.obs
    unique_values = df["condition"].unique()

    # 替换值
    for i, value in enumerate(unique_values):
        df["condition"].replace(value, i, inplace=True)

    # 查看结果
    # print(df)
    dd = df["condition"]
    dd = dd.astype("int")
    # max_value = dd.max()
    # min_value = df["perturbation"].min()

    # print(max_value)

    datainput = pd.concat([X, dd], axis=1)
    print(datainput.shape)
    ###onehot编码
    # one_hot_encoder = OneHotEncoder(sparse=False)
    # datainput.loc['condition'] = one_hot_encoder.fit_transform(datainput.loc['condition'].reshape(-1, 1))

    one_hot_encoder_df =pd.get_dummies(datainput['condition'],prefix="condition")
    datainput.drop("condition",axis=1,inplace=True)
    datainput =pd.concat([datainput,one_hot_encoder_df],axis=1)
    # print(datainput)
    # exit()
    # print(datainput.iloc[:,:-1])
    ###变小
    # exit()
    # indices = np.random.randint(0, high=datainput.shape[0], size = inputsize,random=42)
    # datainput =datainput.iloc[indices,:]
    datainput =datainput.sample(n= inputsize, random_state= 42)
    ###变小结束

    ###数据取CSE RV MOCK
    # top3_targets = [0, 1, 2]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[datainput['batch'].isin(top3_targets)]

    ##数据去除"Basal", "Brush+PNEC", "Cycling basal"
    top3_targets = ['A549']  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    Train = datainput[~df['cell_type'].isin(top3_targets)]
    # print(df['cell_type'].isin(top3_targets).shape,'A549')
    print(Train.shape,'Train.shape')
    ##打乱顺序
    # shuffled_Train = Train.sample(frac=1, random_state=42)
    # print(shuffled_Train)

    # top1_targets = [3]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Test = datainput[datainput['batch'].isin(top1_targets)]

    # print(Test)
    ###把4类数据删掉
    '''
    X_test = Test.iloc[:,:-1]
    y_test = Test['batch']
    X_train = shuffled_Train.iloc[:,:-1]
    y_train = shuffled_Train['batch']
    '''


    # print(min_value)
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-188], Train.iloc[:,-188:],
                                                        test_size=0.2, random_state=42)
    # print(X_train)
    X_train = np.array(X_train.values)
    X_test = np.array(X_test.values)
    y_train = np.array(y_train.values)
    y_test = np.array(y_test.values)

    # indices = random.randint(0, X_train.shape[0], size =inputsize)

    # train_x = data_inputx[indices,:]
    # train_y = data_inputy[indices]
    # test_x = data_inputx[indicesy,:]
    # test_y = data_inputy[indicesy]
    # print(X_train.shape)

    ###onehot
    # one_hot_encoder = OneHotEncoder(sparse=False)

    # y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    # y_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))
    ###区分实验后续使用

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                        test_size=0.2, random_state=42)

    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    y1_value = X_test.min()

    print(yy_value, y1_value, X_train.max(),X_test.max())
    # exit()
    print(X_train.shape,'_train.shape')
    print(y_train.shape,'y_train.shape')
    print(X_test.shape,"X_test.shape")
    print(y_test.shape,'y_test.shape')
    print('read file finish')

    return X_train, X_val, y_train, y_val


def dataread_sciplex_other(file,inputsize=20000):
    adata = sc.read_h5ad(filename=file)

    # 查看数据
    # print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # X = (X-X.mean())/X.std()
    # X = (X - X.min()) / (X.max() - X.min())
    # print(adata.to_df())

    # print(adata.obs)
    df = adata.obs
    unique_values = df["condition"].unique()

    # 替换值
    for i, value in enumerate(unique_values):
        df["condition"].replace(value, i, inplace=True)

    # 查看结果
    # print(df)
    dd = df["condition"]
    dd = dd.astype("int")
    # max_value = dd.max()
    # min_value = df["perturbation"].min()

    # print(max_value)

    datainput = pd.concat([X, dd], axis=1)
    print(datainput.shape)
    ###onehot编码
    # one_hot_encoder = OneHotEncoder(sparse=False)
    # datainput.loc['condition'] = one_hot_encoder.fit_transform(datainput.loc['condition'].reshape(-1, 1))

    one_hot_encoder_df =pd.get_dummies(datainput['condition'],prefix="condition")
    datainput.drop("condition",axis=1,inplace=True)
    datainput =pd.concat([datainput,one_hot_encoder_df],axis=1)
    # print(datainput)
    # exit()
    # print(datainput.iloc[:,:-1])
    ###变小
    # exit()
    # indices = np.random.randint(0, high=datainput.shape[0], size = inputsize,random=42)
    # datainput =datainput.iloc[indices,:]
    datainput =datainput.sample(n= inputsize, random_state= 42)
    ###变小结束

    ###数据取CSE RV MOCK
    # top3_targets = [0, 1, 2]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[datainput['batch'].isin(top3_targets)]

    ##数据去除"Basal", "Brush+PNEC", "Cycling basal"
    top3_targets = ['A549']  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    Train = datainput[~df['cell_type'].isin(top3_targets)]
    Test = datainput[df['cell_type'].isin(top3_targets)]
    # Test =
    print(Test.shape,'Test.shape')
    # print(df['cell_type'].isin(top3_targets).shape,'A549')
    print(Train.shape,'Train.shape')
    ##打乱顺序
    # shuffled_Train = Train.sample(frac=1, random_state=42)
    # print(shuffled_Train)

    # top1_targets = [3]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Test = datainput[datainput['batch'].isin(top1_targets)]

    # print(Test)
    ###把4类数据删掉
    '''
    X_test = Test.iloc[:,:-1]
    y_test = Test['batch']
    X_train = shuffled_Train.iloc[:,:-1]
    y_train = shuffled_Train['batch']
    '''

    X_train =Test.iloc[:, :-188]
    y_train = Test.iloc[:,-188:]
    # print(min_value)
    # X_train, X_test, y_train, y_test = train_test_split(Test.iloc[:, :-188], Test.iloc[:,-188:],
    #                                                     test_size=0.2, random_state=42)
    # print(X_train)
    X_train = np.array(X_train.values)
    # X_test = np.array(X_test.values)
    y_train = np.array(y_train.values)
    # y_test = np.array(y_test.values)

    # indices = random.randint(0, X_train.shape[0], size =inputsize)

    # train_x = data_inputx[indices,:]
    # train_y = data_inputy[indices]
    # test_x = data_inputx[indicesy,:]
    # test_y = data_inputy[indicesy]
    # print(X_train.shape)

    ###onehot
    # one_hot_encoder = OneHotEncoder(sparse=False)

    # y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    # y_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))
    ###区分实验后续使用

    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
    #                                                     test_size=0.2, random_state=42)

    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    # y1_value = X_test.min()

    # print(yy_value, y1_value, X_train.max(),X_test.max())
    # exit()
    print(X_train.shape,'X_train.shape')
    print(y_train.shape,'y_train.shape')
    # print(X_test.shape,"X_test.shape")
    # print(y_test.shape,'y_test.shape')
    print('read file finish')

    return X_train,  y_train



def dataread_rvcse_other(file):
    adata = sc.read_h5ad(filename=file)

    # 查看数据
    # print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # X = (X-X.mean())/X.std()
    X = (X - X.min()) / (X.max() - X.min())
    # print(adata.to_df())

    # print(adata.obs)
    df = adata.obs
    unique_values = df["batch"].unique()

    # 替换值
    for i, value in enumerate(unique_values):
        df["batch"].replace(value, i, inplace=True)

    # 查看结果
    # print(df)
    dd = df["batch"]
    dd = dd.astype("int")
    # max_value = dd.max()
    # min_value = df["perturbation"].min()

    # print(max_value)

    datainput = pd.concat([X, dd], axis=1)
    # print(datainput)
    # print(datainput.iloc[:,:-1])

    ###数据取CSE RV MOCK
    # top3_targets = [0, 1, 2]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[datainput['batch'].isin(top3_targets)]

    ##数据去除"Basal", "Brush+PNEC", "Cycling basal"
    # top3_targets = ["Basal", "Brush+PNEC", "Cycling basal"]  # 前三个类别
    # top3_targets = ["Basal"]
    # top3_targets = ["Brush+PNEC"]
    # top3_targets = ["Doublet"]
    top3_targets = ["Ciliated"]
    print(top3_targets)
    # top3_targets = ["Myoepithelial"]
    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[~df['cell_type1021'].isin(top3_targets)]
    Train = datainput[df['cell_type1021'].isin(top3_targets)]
    Test = datainput[df['cell_type1021'].isin(top3_targets)]
    # print(Train)
    ##打乱顺序
    # shuffled_Train = Train.sample(frac=1, random_state=42)
    # print(shuffled_Train)

    # top1_targets = [3]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Test = datainput[datainput['batch'].isin(top1_targets)]

    # print(Test)
    ###把4类数据删掉
    '''
    X_test = Test.iloc[:,:-1]
    y_test = Test['batch']
    X_train = shuffled_Train.iloc[:,:-1]
    y_train = shuffled_Train['batch']
    '''
    X_train =Test.iloc[:, :-1]
    y_train = Test.iloc[:,-1:]

    # print(min_value)
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-1], Train['batch'],
                                                        test_size=0.1, random_state=42)
    # print(X_train)
    X_train = np.array(X_train.values)
    # X_test = np.array(X_test.values)
    y_train = np.array(y_train.values)
    # y_test = np.array(y_test.values)

    # indices = random.randint(0, X_train.shape[0], size =inputsize)

    # train_x = data_inputx[indices,:]
    # train_y = data_inputy[indices]
    # test_x = data_inputx[indicesy,:]
    # test_y = data_inputy[indicesy]
    # print(X_train.shape)
    one_hot_encoder = OneHotEncoder(sparse=False)

    y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    # y_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))

    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    # y1_value = X_test.min()

    print(yy_value)
    # exit()
    print(X_train.shape)
    print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    print('read file finish')

    return X_train,y_train




def dataread_rvcse_other_loop(file,target):
    adata = sc.read_h5ad(filename=file)

    # 查看数据
    # print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # X = (X-X.mean())/X.std()
    X = (X - X.min()) / (X.max() - X.min())
    # print(adata.to_df())

    # print(adata.obs)
    df = adata.obs
    unique_values = df["batch"].unique()

    # 替换值
    for i, value in enumerate(unique_values):
        df["batch"].replace(value, i, inplace=True)

    # 查看结果
    # print(df)
    dd = df["batch"]
    dd = dd.astype("int")
    # max_value = dd.max()
    # min_value = df["perturbation"].min()

    # print(max_value)

    datainput = pd.concat([X, dd], axis=1)
    # print(datainput)
    # print(datainput.iloc[:,:-1])

    ###数据取CSE RV MOCK
    # top3_targets = [0, 1, 2]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[datainput['batch'].isin(top3_targets)]

    ##数据去除"Basal", "Brush+PNEC", "Cycling basal"
    # top3_targets = ["Basal", "Brush+PNEC", "Cycling basal"]  # 前三个类别
    # top3_targets = ["Basal"]
    # top3_targets = ["Doublet"]
    # top3_targets = ["Ciliated"]
    top3_targets = [target]
    # top3_targets = target.split(str=" ")
    print(top3_targets)
    # top3_targets = ["Myoepithelial"]
    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[~df['cell_type1021'].isin(top3_targets)]
    Train = datainput[df['cell_type1021'].isin(top3_targets)]
    Test = datainput[df['cell_type1021'].isin(top3_targets)]
    # print(Train)
    ##打乱顺序
    # shuffled_Train = Train.sample(frac=1, random_state=42)
    # print(shuffled_Train)

    # top1_targets = [3]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Test = datainput[datainput['batch'].isin(top1_targets)]

    # print(Test)
    ###把4类数据删掉
    '''
    X_test = Test.iloc[:,:-1]
    y_test = Test['batch']
    X_train = shuffled_Train.iloc[:,:-1]
    y_train = shuffled_Train['batch']
    '''
    X_train =Test.iloc[:, :-1]
    y_train = Test.iloc[:,-1:]

    # print(min_value)
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-1], Train['batch'],
                                                        test_size=0.1, random_state=42)
    # print(X_train)
    X_train = np.array(X_train.values)
    # X_test = np.array(X_test.values)
    y_train = np.array(y_train.values)
    # y_test = np.array(y_test.values)

    # indices = random.randint(0, X_train.shape[0], size =inputsize)

    # train_x = data_inputx[indices,:]
    # train_y = data_inputy[indices]
    # test_x = data_inputx[indicesy,:]
    # test_y = data_inputy[indicesy]
    # print(X_train.shape)
    one_hot_encoder = OneHotEncoder(sparse=False)

    y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    # y_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))

    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    # y1_value = X_test.min()

    print(yy_value)
    # exit()
    print(X_train.shape)
    print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    print('read file finish')

    return X_train,y_train

def dataread_rvcse_in(file):
    adata = sc.read_h5ad(filename=file)

    # 查看数据
    # print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # X = (X-X.mean())/X.std()
    X = (X - X.min()) / (X.max() - X.min())
    # print(adata.to_df())

    # print(adata.obs)
    df = adata.obs
    unique_values = df["batch"].unique()

    # 替换值
    for i, value in enumerate(unique_values):
        df["batch"].replace(value, i, inplace=True)

    # 查看结果
    # print(df)
    dd = df["batch"]
    dd = dd.astype("int")
    # max_value = dd.max()
    # min_value = df["perturbation"].min()

    # print(max_value)

    datainput = pd.concat([X, dd], axis=1)
    # print(datainput)
    # print(datainput.iloc[:,:-1])

    ###数据取CSE RV MOCK
    # top3_targets = [0, 1, 2]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[datainput['batch'].isin(top3_targets)]

    ##数据去除"Basal", "Brush+PNEC", "Cycling basal"
    top3_targets = ["Basal", "Brush+PNEC", "Cycling basal"]  # 前三个类别
    top3_targets = ["Basal"]
    # top3_targets = ["Doublet"]
    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    Train = datainput[~df['cell_type1021'].isin(top3_targets)]

    Test = datainput[df['cell_type1021'].isin(top3_targets)]
    # print(Train)
    ##打乱顺序
    # shuffled_Train = Train.sample(frac=1, random_state=42)
    # print(shuffled_Train)

    # top1_targets = [3]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Test = datainput[datainput['batch'].isin(top1_targets)]

    # print(Test)
    ###把4类数据删掉
    '''
    X_test = Test.iloc[:,:-1]
    y_test = Test['batch']
    X_train = shuffled_Train.iloc[:,:-1]
    y_train = shuffled_Train['batch']
    '''
    X_train =Test.iloc[:, :-1]
    y_train = Test.iloc[:,-1:]

    # print(min_value)
    # X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-1], Train['batch'],
    #                                                     test_size=0.1, random_state=42)
    # print(X_train)
    X_train = np.array(X_train.values)
    # X_test = np.array(X_test.values)
    y_train = np.array(y_train.values)
    # y_test = np.array(y_test.values)

    # indices = random.randint(0, X_train.shape[0], size =inputsize)

    # train_x = data_inputx[indices,:]
    # train_y = data_inputy[indices]
    # test_x = data_inputx[indicesy,:]
    # test_y = data_inputy[indicesy]
    # print(X_train.shape)
    one_hot_encoder = OneHotEncoder(sparse=False)

    y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    # y_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))

    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    # y1_value = X_test.min()

    print(yy_value)
    # exit()
    print(X_train.shape)
    print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    print('read file finish')

    return X_train,y_train


def dataread_inter_other(file):
    adata = sc.read_h5ad(filename=file)

    # 查看数据
    # print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # X = (X-X.mean())/X.std()
    X = (X - X.min()) / (X.max() - X.min())
    # print(adata.to_df())

    # print(adata.obs)
    df = adata.obs
    unique_values = df["perturbation"].unique()

    # 替换值
    for i, value in enumerate(unique_values):
        df["perturbation"].replace(value, i, inplace=True)

    # 查看结果
    # print(df)
    dd = df["perturbation"]
    dd = dd.astype("int")
    # max_value = dd.max()
    # min_value = df["perturbation"].min()

    # print(max_value)

    datainput = pd.concat([X, dd], axis=1)
    # print(datainput)
    # print(datainput.iloc[:,:-1])

    ###数据取CSE RV MOCK
    # top3_targets = [0, 1, 2]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[datainput['batch'].isin(top3_targets)]

    ##数据去除"Basal", "Brush+PNEC", "Cycling basal"
    # top3_targets = ["Basal", "Brush+PNEC", "Cycling basal"]  # 前三个类别
    # top3_targets = ["Basal"]
    top3_targets = ["Plasma"]
    # top3_targets = ["Doublet"]
    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[~df['cell_type0528'].isin(top3_targets)]
    Train = datainput[df['cell_type0528'].isin(top3_targets)]
    print(top3_targets)
    Test = datainput[df['cell_type0528'].isin(top3_targets)]
    # print(Train)
    ##打乱顺序
    # shuffled_Train = Train.sample(frac=1, random_state=42)
    # print(shuffled_Train)

    # top1_targets = [3]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Test = datainput[datainput['batch'].isin(top1_targets)]

    # print(Test)
    ###把4类数据删掉
    '''
    X_test = Test.iloc[:,:-1]
    y_test = Test['batch']
    X_train = shuffled_Train.iloc[:,:-1]
    y_train = shuffled_Train['batch']
    '''
    X_train =Test.iloc[:, :-1]
    y_train = Test.iloc[:,-1:]

    # print(min_value)
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-1], Train['perturbation'],
                                                        test_size=0.1, random_state=42)
    # print(X_train)
    X_train = np.array(X_train.values)
    # X_test = np.array(X_test.values)
    y_train = np.array(y_train.values)
    # y_test = np.array(y_test.values)

    # indices = random.randint(0, X_train.shape[0], size =inputsize)

    # train_x = data_inputx[indices,:]
    # train_y = data_inputy[indices]
    # test_x = data_inputx[indicesy,:]
    # test_y = data_inputy[indicesy]
    # print(X_train.shape)
    one_hot_encoder = OneHotEncoder(sparse=False)

    y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    # y_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))

    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    # y1_value = X_test.min()

    print(yy_value)
    # exit()
    print(X_train.shape)
    print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    print('read file finish')

    return X_train,y_train



def dataread_inter_other_loop(file,traget):
    adata = sc.read_h5ad(filename=file)

    # 查看数据
    # print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # X = (X-X.mean())/X.std()
    X = (X - X.min()) / (X.max() - X.min())
    # print(adata.to_df())

    # print(adata.obs)
    df = adata.obs
    unique_values = df["perturbation"].unique()

    # 替换值
    for i, value in enumerate(unique_values):
        df["perturbation"].replace(value, i, inplace=True)

    # 查看结果
    # print(df)
    dd = df["perturbation"]
    dd = dd.astype("int")
    # max_value = dd.max()
    # min_value = df["perturbation"].min()

    # print(max_value)

    datainput = pd.concat([X, dd], axis=1)
    # print(datainput)
    # print(datainput.iloc[:,:-1])

    ###数据取CSE RV MOCK
    # top3_targets = [0, 1, 2]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[datainput['batch'].isin(top3_targets)]

    ##数据去除"Basal", "Brush+PNEC", "Cycling basal"
    # top3_targets = ["Basal", "Brush+PNEC", "Cycling basal"]  # 前三个类别
    # top3_targets = ["Basal"]
    # top3_targets = ["Plasma"]
    top3_targets = [traget]
    # top3_targets = ["Doublet"]
    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[~df['cell_type0528'].isin(top3_targets)]
    Train = datainput[df['cell_type0528'].isin(top3_targets)]
    print(top3_targets)
    Test = datainput[df['cell_type0528'].isin(top3_targets)]
    # print(Train)
    ##打乱顺序
    # shuffled_Train = Train.sample(frac=1, random_state=42)
    # print(shuffled_Train)

    # top1_targets = [3]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Test = datainput[datainput['batch'].isin(top1_targets)]

    # print(Test)
    ###把4类数据删掉
    '''
    X_test = Test.iloc[:,:-1]
    y_test = Test['batch']
    X_train = shuffled_Train.iloc[:,:-1]
    y_train = shuffled_Train['batch']
    '''
    X_train =Test.iloc[:, :-1]
    y_train = Test.iloc[:,-1:]

    # print(min_value)
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-1], Train['perturbation'],
                                                        test_size=0.1, random_state=42)
    # print(X_train)
    X_train = np.array(X_train.values)
    # X_test = np.array(X_test.values)
    y_train = np.array(y_train.values)
    # y_test = np.array(y_test.values)

    # indices = random.randint(0, X_train.shape[0], size =inputsize)

    # train_x = data_inputx[indices,:]
    # train_y = data_inputy[indices]
    # test_x = data_inputx[indicesy,:]
    # test_y = data_inputy[indicesy]
    # print(X_train.shape)
    one_hot_encoder = OneHotEncoder(sparse=False)

    y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    # y_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))

    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    # y1_value = X_test.min()

    print(yy_value)
    # exit()
    print(X_train.shape)
    print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    print('read file finish')

    return X_train,y_train



# file = 'sciplex_othermodel.h5ad'
# file = "SrivatsanTrapnell2020_sciplex2.h5ad"
# x, y = dataread_sciplex_other(file)
# X_train =x[:-1,:]
# y_train =y[:-1,:]
# X_pre = x[1:,:]
# y_pre = y[1:,:]
# print(X_train.shape,y_train.shape,X_pre.shape, y_pre.shape)
