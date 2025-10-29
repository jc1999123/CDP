from concurrent.futures import process
import scanpy as sc

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
import random
def dataread(file):
    adata = sc.read_h5ad(filename=file)

    # 查看数据
    # print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # X = (X-X.mean())/X.std()
    X = (X- X.min())/(X.max()-X.min())
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
    top3_targets = ["B","CD8 T","Monocyte"]
    # top3_targets = ["Myoepithelial"]
    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    Train = datainput[~df['cell_type0528'].isin(top3_targets)]

    # print(min_value)
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:,:-1], Train['perturbation'], test_size=0.1, random_state=42)
    # print(X_train)
    X_train =np.array(X_train.values)
    X_test =np.array(X_test.values)
    y_train =np.array(y_train.values)
    y_test =np.array(y_test.values)
    
    # indices = random.randint(0, X_train.shape[0], size =inputsize)

    # train_x = data_inputx[indices,:]
    # train_y = data_inputy[indices]
    # test_x = data_inputx[indicesy,:]
    # test_y = data_inputy[indicesy]
# print(X_train.shape)
    one_hot_encoder = OneHotEncoder(sparse=False)
    
    y_train  = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test  = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))


    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    y1_value = X_test.min()

    print(yy_value,y1_value)
    # exit()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print('read file finish')

    return X_train, X_test, y_train, y_test 

    # 读取 h5ad 文件


def dataread_small(file,inputsize =10000):
    adata = sc.read_h5ad(filename=file)

    # 查看数据
    # print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # X = (X-X.mean())/X.std()
    X = (X- X.min())/(X.max()-X.min())
    # print(adata.to_df())
    # print(X.shape)

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
    max_value = dd.max()
    # min_value = df["perturbation"].min()


    # print(max_value)

    datainput = pd.concat([X, dd], axis=1)
    # print(datainput.shape())
    # print(datainput.iloc[:,:-1])

    # print(min_value)

    ###变小
    indices = np.random.randint(0, high=datainput.shape[0], size = inputsize)
    datainput =datainput.iloc[indices,:]
    ###变小结束

    X_train, X_test, y_train, y_test = train_test_split(datainput.iloc[:,:-1], datainput['perturbation'], test_size=0.1, random_state=42)
    # print(X_train)
    X_train =np.array(X_train.values)
    X_test =np.array(X_test.values)
    y_train =np.array(y_train.values)
    y_test =np.array(y_test.values)
    
    # indices = random.randint(0, X_train.shape[0], size =inputsize)

    # train_x = data_inputx[indices,:]
    # train_y = data_inputy[indices]
    # test_x = data_inputx[indicesy,:]
    # test_y = data_inputy[indicesy]
# print(X_train.shape)
    one_hot_encoder = OneHotEncoder(sparse=False)
    
    y_train  = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test  = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))


    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    y1_value = X_test.min()

    print(yy_value,y1_value,X_train.max(),X_test.max())
    # exit()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print('read file finish')

    return X_train, X_test, y_train, y_test


def dataread_sciplex(file,inputsize=20000):
    adata = sc.read_h5ad(filename=file)

    # 查看数据
    # print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # zscore = preprocessing.StandardScaler()
    # X =zscore.fit_transform(X)
    # X =pd.DataFrame(X)
    # X = (X-X.mean())/X.std()
    # X = (X - X.min()) / (X.max() - X.min())
    # print(adata.to_df())

    print(adata.obs)
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

    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                        # test_size=0.2, random_state=42)

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

    return X_train, X_test, y_train, y_test

def dataread_rvcse(file,inputsize =20000):
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
    df1 = adata.obs

    # print(X)
    # print(df)

    unique_values = df["batch"].unique()

    # 替换值
    for i, value in enumerate(unique_values):
        df["batch"].replace(value, i, inplace=True)

    # 查看结果
    # print(df)
    dd = df["batch"]
    dd = dd.astype("int")
    # print(unique_values)
    unique_values = df1["cell_type1021"].unique()

    # 替换值
    for i, value in enumerate(unique_values):
        df1["cell_type1021"].replace(value, i, inplace=True)

    # 查看结果
    # print(df)
    dd1 = df1["cell_type1021"]
    dd1 = dd1.astype("int")
    # print(unique_values)
    # exit()
    # max_value = dd.max()
    # min_value = df["perturbation"].min()

    # print(max_value)

    datainput = pd.concat([X, dd,dd1], axis=1)
    print(datainput,'datainput')
    # print(datainput.iloc[:,:-1])



    one_hot_encoder_df =pd.get_dummies(datainput['batch'],prefix="batch")
    datainput.drop("batch",axis=1,inplace=True)
    datainput =pd.concat([datainput,one_hot_encoder_df],axis=1)
    print(datainput.shape,'data_inputshape')


    ###数据取CSE RV MOCK
    top3_targets = [2,5]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[datainput['batch'].isin(top3_targets)]

    ##数据去除"Basal", "Brush+PNEC", "Cycling basal"
    # top3_targets = ["Hillock ", "Ionocyte", "Secretory", "Myoepithelial"]  # 前三个类别
    # df = adata.obs
    # print(df,'df')
    # top3_targets = ["Basal","Brush+PNEC"]  # 前liang个类别

    print(top3_targets)

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[~df['cell_type1021'].isin(top3_targets)]
    Train = datainput[df['cell_type1021'].isin(top3_targets)]
    print(Train.shape,'train.shape')



    one_hot_encoder_df_celltype =pd.get_dummies(Train['cell_type1021'],prefix="cell_type1021")
    Train.drop("cell_type1021",axis=1,inplace=True)
    Train =pd.concat([Train,one_hot_encoder_df_celltype],axis=1)
    print(Train.shape,'Trainshape')
    # exit()
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
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-6], Train.iloc[:, -6:],
                                                        test_size=0.1, random_state=42)
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
    # one_hot_encoder = OneHotEncoder(sparse=False)

    # y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    # y_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))

    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    y1_value = X_test.min()

    print(yy_value, y1_value)
    # exit()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print('read file finish')

    return X_train, X_test, y_train, y_test


def dataread_sciplex2(file,inputsize=20000):
    adata = sc.read_h5ad(filename=file)

    # 查看数据
    # print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # X = (X-X.mean())/X.std()
    # X = (X - X.min()) / (X.max() - X.min())
    # print(adata.to_df())

    print(adata.obs)
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
    print(datainput.shape)
    ###onehot编码
    # one_hot_encoder = OneHotEncoder(sparse=False)
    # datainput.loc['condition'] = one_hot_encoder.fit_transform(datainput.loc['condition'].reshape(-1, 1))

    one_hot_encoder_df =pd.get_dummies(datainput['perturbation'],prefix="perturbation")
    datainput.drop("perturbation",axis=1,inplace=True)
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
    # top3_targets = ['A549']  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[~df['cell_type'].isin(top3_targets)]

    Train =datainput
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




    # print(min_value)
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-5], Train.iloc[:,-5:],
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

    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                        # test_size=0.2, random_state=42)

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

    return X_train, X_test, y_train, y_test


def dataread_sciplex3_smalllable1(file,inputsize=20000):
    adata = sc.read_h5ad(filename=file)

    # 查看数据
    # print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # X = (X-X.mean())/X.std()
    # X = (X - X.min()) / (X.max() - X.min())
    # print(adata.to_df())

    print(adata.obs)
    df = adata.obs
    ###选取10种作为训练
    X = adata[adata.obs['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737 '])]
    df = df[df['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737 '])]

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
    ###不需要datasample
    # datainput =datainput.sample(n= inputsize, random_state= 42)
    ###变小结束

    ###数据取CSE RV MOCK
    # top3_targets = [0, 1, 2]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[datainput['batch'].isin(top3_targets)]

    ##数据去除"Basal", "Brush+PNEC", "Cycling basal"
    top3_targets = ['A549']  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    Train = datainput[~df['cell_type'].isin(top3_targets)]

    # Train =datainput
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




    # print(min_value)
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-7], Train.iloc[:,-7:],
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

    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                        # test_size=0.2, random_state=42)

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

    return X_train, X_test, y_train, y_test

def dataread_sciplex_smalllable(file,inputsize=20000):
    adata1 = sc.read_h5ad(filename=file)
    print(adata1.shape)
    # adata = adata1[adata1.obs['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737'])]
    adata = adata1[adata1.obs['condition'].isin(['control', 'JQ1','Zileuton'])]
    adata.obs['condition'] = adata.obs['condition'].astype(str)
    # 查看数据
    print(adata.X.shape)
    # print(adata.X)
    
    X = adata.to_df()
    # pd.DataFrame(X ).to_csv('mydata.csv')
    # exit()
    X =pd.DataFrame(X)
    X = (X - X.min()) / (X.max() - X.min())
    # X = X.reset_index()
    # dd1 =adata.obs["index"]
    # df_with_index_as_column = adata.obs.reset_index()
    # print(df_with_index_as_column)
    # exit()
    # zscore = preprocessing.StandardScaler()
    # X =zscore.fit_transform(X)
    # X =pd.DataFrame(X)
    # X = (X-X.mean())/X.std()
    # X = (X-X.mean())/X.std()
    ###标准化
    #### X = (X - X.min()) / (X.max() - X.min())
    # X =pd.concat([df_with_index_as_column["index"],X],axis=0)
    print(X)
    X = X.fillna(0.0)
    print(X)
    # exit()

    # adata = adata[~np.isnan(adata.X).any(axis=1)]
    # X = X.dropna(axis=1)
    # # print(adata.to_df())
    # # X = adata.to_df()
    # print(adata.obs)
    df = adata.obs

    # X = adata[adata.obs['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737 '])]
    # df = df[df['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737 '])]


    unique_values = df["condition"].unique()

    # 替换值
    for i, value in enumerate(unique_values):
        df["condition"].replace(value, i, inplace=True)

    # 查看结果
    # print(df)
    dd = df["condition"]
    dd = dd.astype("int")
    # dd = dd.reset_index()
    # max_value = dd.max()
    # min_value = df["perturbation"].min()

    # print(max_value)
    # datainput = pd.merge(X, dd, on='index', how='inner')
    # print(datainput)
    # datainput = datainput.iloc[:,1:].values
    # exit()


    datainput = pd.concat([X, dd], axis=1)

    print(datainput.shape)
    print(datainput,"datainput")
    datainput =  datainput.dropna(axis=1)
    print(datainput.shape,'datainputshape')
    ###onehot编码
    # one_hot_encoder = OneHotEncoder(sparse=False)
    # datainput.loc['condition'] = one_hot_encoder.fit_transform(datainput.loc['condition'].reshape(-1, 1))

    one_hot_encoder_df =pd.get_dummies(datainput['condition'],prefix="condition")
    datainput.drop("condition",axis=1,inplace=True)
    datainput =pd.concat([datainput,one_hot_encoder_df],axis=1)
    print(datainput.shape,'data_inputshape')
    # print(datainput)
    # exit()
    # print(datainput.iloc[:,:-1])
    ###变小
    # exit()
    # indices = np.random.randint(0, high=datainput.shape[0], size = inputsize,random=42)
    # datainput =datainput.iloc[indices,:]
    # datainput =datainput.sample(n= inputsize, random_state= 42)
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
    data_my = Train.iloc[:, :-3]
    print(data_my.shape,"data_my")
    # pd.DataFrame(data_my).to_csv('allmydata_smalldrug.csv')
    # exit()

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
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-3], Train.iloc[:,-3:],
                                                        test_size=0.2, shuffle =False)
    # X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-3], Train.iloc[:,-3:],
    #                                                     test_size=0.2, random_state=42)                                                        
    # print(X_train)
    X_train = np.array(X_train.iloc[:,:].values)
    X_test = np.array(X_test.iloc[:,:].values)
    y_train = np.array(y_train.iloc[:,:].values)
    y_test = np.array(y_test.iloc[:,:].values)

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
                                                        # test_size=0.2, random_state=42)

    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    y1_value = X_test.min()

    print(yy_value, X_train.max(),np.mean(X_train),np.var(X_train),np.median(X_train))
    # exit()
    print(X_train.shape,'_train.shape')
    print(y_train.shape,'y_train.shape')
    print(X_test.shape,"X_test.shape")
    print(y_test.shape,'y_test.shape')
    print('read file finish')

    return X_train, X_test, y_train, y_test


def dataread_sciplex_smalllable_a549(file,inputsize=20000):
    adata1 = sc.read_h5ad(filename=file)
    print(adata1.shape)
    # adata = adata1[adata1.obs['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737'])]
    adata = adata1[adata1.obs['condition'].isin(['control', 'JQ1','Zileuton'])]
    adata.obs['condition'] = adata.obs['condition'].astype(str)
    # 查看数据
    print(adata.X.shape)
    # print(adata.X)
    
    X = adata.to_df()
    # pd.DataFrame(X ).to_csv('mydata.csv')
    # exit()
    # X =pd.DataFrame(X)
    # X = (X - X.min()) / (X.max() - X.min())
    # X = X.reset_index()
    # dd1 =adata.obs["index"]
    # df_with_index_as_column = adata.obs.reset_index()
    # print(df_with_index_as_column)
    # exit()
    # zscore = preprocessing.StandardScaler()
    # X =zscore.fit_transform(X)
    # X =pd.DataFrame(X)
    # X = (X-X.mean())/X.std()
    # X = (X-X.mean())/X.std()
    ###标准化
    #### X = (X - X.min()) / (X.max() - X.min())
    # X =pd.concat([df_with_index_as_column["index"],X],axis=0)
    # print(X)
    # exit()

    # adata = adata[~np.isnan(adata.X).any(axis=1)]
    # X = X.dropna(axis=1)
    # # print(adata.to_df())
    # # X = adata.to_df()
    # print(adata.obs)
    df = adata.obs

    # X = adata[adata.obs['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737 '])]
    # df = df[df['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737 '])]


    unique_values = df["condition"].unique()

    # 替换值
    for i, value in enumerate(unique_values):
        df["condition"].replace(value, i, inplace=True)

    # 查看结果
    # print(df)
    dd = df["condition"]
    dd = dd.astype("int")
    # dd = dd.reset_index()
    # max_value = dd.max()
    # min_value = df["perturbation"].min()

    # print(max_value)
    # datainput = pd.merge(X, dd, on='index', how='inner')
    # print(datainput)
    # datainput = datainput.iloc[:,1:].values
    # exit()


    datainput = pd.concat([X, dd], axis=1)

    print(datainput.shape)
    print(datainput,"datainput")
    datainput =  datainput.dropna(axis=1)
    print(datainput.shape,'datainputshape')
    ###onehot编码
    # one_hot_encoder = OneHotEncoder(sparse=False)
    # datainput.loc['condition'] = one_hot_encoder.fit_transform(datainput.loc['condition'].reshape(-1, 1))

    one_hot_encoder_df =pd.get_dummies(datainput['condition'],prefix="condition")
    datainput.drop("condition",axis=1,inplace=True)
    datainput =pd.concat([datainput,one_hot_encoder_df],axis=1)
    print(datainput.shape,'data_inputshape')
    # print(datainput)
    # exit()
    # print(datainput.iloc[:,:-1])
    ###变小
    # exit()
    # indices = np.random.randint(0, high=datainput.shape[0], size = inputsize,random=42)
    # datainput =datainput.iloc[indices,:]
    # datainput =datainput.sample(n= inputsize, random_state= 42)
    ###变小结束

    ###数据取CSE RV MOCK
    # top3_targets = [0, 1, 2]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[datainput['batch'].isin(top3_targets)]

    ##数据去除"Basal", "Brush+PNEC", "Cycling basal"
    top3_targets = ['A549']  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    Train = datainput[df['cell_type'].isin(top3_targets)]
    # print(df['cell_type'].isin(top3_targets).shape,'A549')
    print(Train.shape,'Train.shape')
    data_my = Train.iloc[:, :-3]
    print(data_my.shape,"data_my")
    # pd.DataFrame(data_my).to_csv('allmydata_a549_smalldrug.csv')
    # exit()

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
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-3], Train.iloc[:,-3:],
                                                        test_size=0.2, shuffle = False)
    # X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-7], Train.iloc[:,-7:],
    #                                                     test_size=0.2, random_state=42)                                                    
    # print(X_train)
    X_train = np.array(X_train.iloc[:,:].values)
    X_test = np.array(X_test.iloc[:,:].values)
    y_train = np.array(y_train.iloc[:,:].values)
    y_test = np.array(y_test.iloc[:,:].values)

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
                                                        # test_size=0.2, random_state=42)

    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    y1_value = X_test.min()

    print(yy_value, X_train.max(),np.mean(X_train),np.var(X_train),np.median(X_train))
    # exit()
    print(X_train.shape,'_train.shape')
    print(y_train.shape,'y_train.shape')
    print(X_test.shape,"X_test.shape")
    print(y_test.shape,'y_test.shape')
    print('read file finish')

    return X_train, X_test, y_train, y_test



def dataread_sciplex_smalllable_test(file,inputsize=20000):
    adata1 = sc.read_h5ad(filename=file)
    print(adata1.shape)
    # adata = adata1[adata1.obs['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737'])]
    adata = adata1[adata1.obs['condition'].isin(['control', 'JQ1','Zileuton'])]
    adata.obs['condition'] = adata.obs['condition'].astype(str)
    # 查看数据
    print(adata.X.shape)
    # print(adata.X)
    
    X = adata.to_df()
    # pd.DataFrame(X ).to_csv('mydata_smalldrug.csv')
    # exit()
    # X =pd.DataFrame(X)
    # X = (X - X.min()) / (X.max() - X.min())
    # X = X.reset_index()
    # dd1 =adata.obs["index"]
    # df_with_index_as_column = adata.obs.reset_index()
    # print(df_with_index_as_column)
    # exit()
    # zscore = preprocessing.StandardScaler()
    # X =zscore.fit_transform(X)
    # X =pd.DataFrame(X)
    # X = (X-X.mean())/X.std()
    # X = (X-X.mean())/X.std()

    # X = (X - X.min()) / (X.max() - X.min())
    # X =pd.concat([df_with_index_as_column["index"],X],axis=0)
    # print(X)
    # exit()

    # adata = adata[~np.isnan(adata.X).any(axis=1)]
    # X = X.dropna(axis=1)
    # # print(adata.to_df())
    # # X = adata.to_df()
    # print(adata.obs)
    df = adata.obs

    # X = adata[adata.obs['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737 '])]
    # df = df[df['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737 '])]


    unique_values = df["condition"].unique()

    # 替换值
    for i, value in enumerate(unique_values):
        df["condition"].replace(value, i, inplace=True)

    # 查看结果
    # print(df)
    dd = df["condition"]
    dd = dd.astype("int")
    # dd = dd.reset_index()
    # max_value = dd.max()
    # min_value = df["perturbation"].min()

    # print(max_value)
    # datainput = pd.merge(X, dd, on='index', how='inner')
    # print(datainput)
    # datainput = datainput.iloc[:,1:].values
    # exit()


    datainput = pd.concat([X, dd], axis=1)

    print(datainput.shape)
    print(datainput,"datainput")
    datainput =  datainput.dropna(axis=1)
    print(datainput.shape,'datainputshape')
    ###onehot编码
    # one_hot_encoder = OneHotEncoder(sparse=False)
    # datainput.loc['condition'] = one_hot_encoder.fit_transform(datainput.loc['condition'].reshape(-1, 1))

    one_hot_encoder_df =pd.get_dummies(datainput['condition'],prefix="condition")
    datainput.drop("condition",axis=1,inplace=True)
    datainput =pd.concat([datainput,one_hot_encoder_df],axis=1)
    print(datainput.shape,'data_inputshape')
    # print(datainput)
    # exit()
    # print(datainput.iloc[:,:-1])
    ###变小
    # exit()
    # indices = np.random.randint(0, high=datainput.shape[0], size = inputsize,random=42)
    # datainput =datainput.iloc[indices,:]
    # datainput =datainput.sample(n= inputsize, random_state= 42)
    ###变小结束

    ###数据取CSE RV MOCK
    # top3_targets = [0, 1, 2]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[datainput['batch'].isin(top3_targets)]

    ##数据去除"Basal", "Brush+PNEC", "Cycling basal"
    top3_targets = ['A549']  # 前三个类别
    Train = datainput[~df['cell_type'].isin(top3_targets)]
    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput
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
    expp = np.array(Train.iloc[:, :-3])
    labell =np.array(Train.iloc[:,-3:])
    return expp ,labell
    '''
    print(Train.iloc[:,:-7],)
    # print(min_value)
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-7], Train.iloc[:,-7:],
                                                        test_size=0.2, random_state=42)
    # print(X_train)
    X_train = np.array(X_train.iloc[:,:].values)
    X_test = np.array(X_test.iloc[:,:].values)
    y_train = np.array(y_train.iloc[:,:].values)
    y_test = np.array(y_test.iloc[:,:].values)

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
                                                        # test_size=0.2, random_state=42)

    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    y1_value = X_test.min()

    print(yy_value, X_train.max(),np.mean(X_train),np.var(X_train),np.median(X_train))
    # exit()
    print(X_train.shape,'_train.shape')
    print(y_train.shape,'y_train.shape')
    print(X_test.shape,"X_test.shape")
    print(y_test.shape,'y_test.shape')
    print('read file finish')

    return X_train, X_test, y_train, y_test
    '''


def dataread_sciplex_smalllable_test_a549(file,inputsize=20000):
    adata1 = sc.read_h5ad(filename=file)
    print(adata1.shape)
    adata = adata1[adata1.obs['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737'])]
    # adata = adata1[adata1.obs['condition'].isin(['control', 'JQ1','Zileuton'])]
    adata.obs['condition'] = adata.obs['condition'].astype(str)
    # 查看数据
    print(adata.X.shape)
    # print(adata.X)
    
    X = adata.to_df()
    # pd.DataFrame(X ).to_csv('mydata.csv')
    # exit()
    # X =pd.DataFrame(X)
    # X = (X - X.min()) / (X.max() - X.min())
    # X = X.reset_index()
    # dd1 =adata.obs["index"]
    # df_with_index_as_column = adata.obs.reset_index()
    # print(df_with_index_as_column)
    # exit()
    # zscore = preprocessing.StandardScaler()
    # X =zscore.fit_transform(X)
    # X =pd.DataFrame(X)
    # X = (X-X.mean())/X.std()
    # X = (X-X.mean())/X.std()

    # X = (X - X.min()) / (X.max() - X.min())
    # X =pd.concat([df_with_index_as_column["index"],X],axis=0)
    # print(X)
    # exit()

    # adata = adata[~np.isnan(adata.X).any(axis=1)]
    # X = X.dropna(axis=1)
    # # print(adata.to_df())
    # # X = adata.to_df()
    # print(adata.obs)
    df = adata.obs

    # X = adata[adata.obs['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737 '])]
    # df = df[df['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737 '])]


    unique_values = df["condition"].unique()

    # 替换值
    for i, value in enumerate(unique_values):
        df["condition"].replace(value, i, inplace=True)

    # 查看结果
    # print(df)
    dd = df["condition"]
    dd = dd.astype("int")
    # dd = dd.reset_index()
    # max_value = dd.max()
    # min_value = df["perturbation"].min()

    # print(max_value)
    # datainput = pd.merge(X, dd, on='index', how='inner')
    # print(datainput)
    # datainput = datainput.iloc[:,1:].values
    # exit()


    datainput = pd.concat([X, dd], axis=1)

    print(datainput.shape)
    print(datainput,"datainput")
    datainput =  datainput.dropna(axis=1)
    print(datainput.shape,'datainputshape')
    ###onehot编码
    # one_hot_encoder = OneHotEncoder(sparse=False)
    # datainput.loc['condition'] = one_hot_encoder.fit_transform(datainput.loc['condition'].reshape(-1, 1))

    one_hot_encoder_df =pd.get_dummies(datainput['condition'],prefix="condition")
    datainput.drop("condition",axis=1,inplace=True)
    datainput =pd.concat([datainput,one_hot_encoder_df],axis=1)
    print(datainput.shape,'data_inputshape')
    # print(datainput)
    # exit()
    # print(datainput.iloc[:,:-1])
    ###变小
    # exit()
    # indices = np.random.randint(0, high=datainput.shape[0], size = inputsize,random=42)
    # datainput =datainput.iloc[indices,:]
    # datainput =datainput.sample(n= inputsize, random_state= 42)
    ###变小结束

    ###数据取CSE RV MOCK
    # top3_targets = [0, 1, 2]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[datainput['batch'].isin(top3_targets)]

    ##数据去除"Basal", "Brush+PNEC", "Cycling basal"
    top3_targets = ['A549']  # 前三个类别
    Train = datainput[df['cell_type'].isin(top3_targets)]
    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput
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
    expp = np.array(Train.iloc[:, :-3])
    labell =np.array(Train.iloc[:,-3:])
    return expp ,labell

def dataread_sciplex_smalllable_other(file,inputsize=20000):
    adata1 = sc.read_h5ad(filename=file)
    print(adata1.shape)
    adata = adata1[adata1.obs['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737'])]
    # adata = adata1[adata1.obs['condition'].isin(['control', 'JQ1','Zileuton'])]
    adata.obs['condition'] = adata.obs['condition'].astype(str)
    # 查看数据
    print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # zscore = preprocessing.StandardScaler()
    # X =zscore.fit_transform(X)
    # X =pd.DataFrame(X)
    # X = (X-X.mean())/X.std()
    # X = (X - X.min()) / (X.max() - X.min())
    X = (X - X.min()) / (X.max() - X.min())
    # print(adata.to_df())

    print(adata.obs)
    df = adata.obs

    # X = adata[adata.obs['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737 '])]
    # df = df[df['condition'].isin(['control', 'JQ1','Zileuton','2-Methoxyestradiol','A-366','AC480','ABT-737 '])]


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
    print(datainput,"datainput")
    datainput =  datainput.dropna(axis=1)
    print(datainput.shape,'datainputshape')
    ###onehot编码
    # one_hot_encoder = OneHotEncoder(sparse=False)
    # datainput.loc['condition'] = one_hot_encoder.fit_transform(datainput.loc['condition'].reshape(-1, 1))

    one_hot_encoder_df =pd.get_dummies(datainput['condition'],prefix="condition")
    datainput.drop("condition",axis=1,inplace=True)
    datainput =pd.concat([datainput,one_hot_encoder_df],axis=1)
    print(datainput.shape,'data_inputshape')
    # print(datainput)
    # exit()
    # print(datainput.iloc[:,:-1])
    ###变小
    # exit()
    # indices = np.random.randint(0, high=datainput.shape[0], size = inputsize,random=42)
    # datainput =datainput.iloc[indices,:]
    # datainput =datainput.sample(n= inputsize, random_state= 42)
    ###变小结束

    ###数据取CSE RV MOCK
    # top3_targets = [0, 1, 2]  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    # Train = datainput[datainput['batch'].isin(top3_targets)]

    ##数据去除"Basal", "Brush+PNEC", "Cycling basal"
    top3_targets = ['A549']  # 前三个类别

    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    Train = datainput[df['cell_type'].isin(top3_targets)]
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
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-7], Train.iloc[:,-7:],
                                                        test_size=0.2, random_state=42)
    # print(X_train)

    X_train = np.array(Train.iloc[:, :-7].values)
    y_train = np.array(Train.iloc[:, -7:].values)
    # X_train = np.array(X_train.values)
    # X_test = np.array(X_test.values)
    # y_train = np.array(y_train.values)
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
                                                        # test_size=0.2, random_state=42)

    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    # y1_value = X_test.min()

    print(yy_value,  X_train.max(),np.mean(X_train),np.var(X_train),np.median(X_train))
    # exit()
    print(X_train.shape,'X_train.shape')
    print(y_train.shape,'y_train.shape')
    # print(X_test.shape,"X_test.shape")
    # print(y_test.shape,'y_test.shape')
    print('read file finish')

    return X_train, y_train

def dataread_inter_CD4(file):
    adata = sc.read_h5ad(filename=file)

    # 查看数据
    # print(adata.X.shape)
    # print(adata.X)
    X = adata.to_df()
    # X = (X-X.mean())/X.std()
    X = (X- X.min())/(X.max()-X.min())
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
    top3_targets = ["B"]
    # top3_targets = ["Myoepithelial"]
    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    Train = datainput[~df['cell_type0528'].isin(top3_targets)]

    # print(min_value)
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:,:-1], Train['perturbation'], test_size=0.1, random_state=42)
    # print(X_train)
    X_train =np.array(X_train.values)
    X_test =np.array(X_test.values)
    y_train =np.array(y_train.values)
    y_test =np.array(y_test.values)
    
    # indices = random.randint(0, X_train.shape[0], size =inputsize)

    # train_x = data_inputx[indices,:]
    # train_y = data_inputy[indices]
    # test_x = data_inputx[indicesy,:]
    # test_y = data_inputy[indicesy]
# print(X_train.shape)
    one_hot_encoder = OneHotEncoder(sparse=False)
    
    y_train  = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test  = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))


    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    y1_value = X_test.min()

    print(yy_value,y1_value)
    # exit()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print('read file finish')

    return X_train, X_test, y_train, y_test


def dataread_inter_B(file):
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
    top3_targets = ["CD4 T"]
    # top3_targets = ["Myoepithelial"]
    # 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
    Train = datainput[~df['cell_type0528'].isin(top3_targets)]

    # print(min_value)
    X_train, X_test, y_train, y_test = train_test_split(Train.iloc[:, :-1], Train['perturbation'], test_size=0.1,
                                                        random_state=42)
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
    one_hot_encoder = OneHotEncoder(sparse=False)

    y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))

    # encoder = LabelEncoder()
    # encoded_data = encoder.fit_transform(y_train)
    # print(encoded_data)
    # one_hot_encoded_data =np.keras.utils.to_categorical(y_train, num_classes=max_value)
    # one_hot_encoded_data =np.eye(max_value)[y_train]

    # print(one_hot_encoded_data.shape)
    yy_value = X_train.min()
    y1_value = X_test.min()

    print(yy_value, y1_value)
    # exit()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print('read file finish')

    return X_train, X_test, y_train, y_test
# file = 'Integrated_.h5ad'
# file = 'rvcse_221021.h5ad'
# file = 'sciplex_othermodel.h5ad'
# X_train, X_test, y_train, y_test =dataread_sciplex(file)



# X_train, X_test, y_train, y_test = dataread_sciplex_smalllable('sciplex_othermodel.h5ad')