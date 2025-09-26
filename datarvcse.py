import scanpy as sc

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


file = 'rvcse_221021.h5ad'
# file =  'AdamsonWeissman2016_GSM2406675_10X001.h5ad'
# file ='AdamsonWeissman2016_GSM2406677_10X005.h5ad'
# file='SrivatsanTrapnell2020_sciplex4.h5ad'
# file='SrivatsanTrapnell2020_sciplex2.h5ad'
# file ='ChangYe2021.h5ad'

adata = sc.read_h5ad(filename=file)

# 查看数据
print(adata.X.shape)
# print(adata.X)
X = adata.to_df()
# print(adata.to_df())

print(adata.obs)
# exit()
df = adata.obs
# csv_file_path = 'ChangYe2021.csv'

# 使用 to_csv 方法将 DataFrame 写入 CSV 文件
# df.to_csv(csv_file_path)
# exit()

### 名字设置
####
# petname =batch or perturbation
petname = 'batch'
###
###
# cellname = "cell_type1021" or "cell_type0528"

cellname = "cell_type1021"


###
label_counts = df.groupby('cell_type1021')['batch'].count()

print(label_counts)

# exit()

value_counts = df[petname].value_counts()

# 输出不同的值及其计数
for value, count in value_counts.items():
    print(f"zhi: {value}, count: {count}")


print('finsh perturbation')
unique_values = df[petname].unique()

# unique_values2 = df["cell_type0528"].unique()

value_counts = df[cellname].value_counts()

# 输出不同的值及其计数
for value, count in value_counts.items():
    print(f"zhi: {value}, count: {count}")
# 替换值
for i, value in enumerate(unique_values):
    df[petname].replace(value, i, inplace=True)

# label_counts = df.groupby('cell_type1021')['batch'].value_counts()
# print(label_counts)


# 查看结果
print(df)
dd = df[petname]
dd = dd.astype("int")
max_value = dd.max()
# min_value = df["perturbation"].min()


print(max_value)

datainput = pd.concat([X, dd], axis=1)
print(datainput)
# print(datainput.iloc[:,:-1])
top3_targets = [0, 1, 2]  # 前三个类别

# 使用 isin 方法检查每个元素是否在前三个类别中，并使用布尔索引筛选对应的行
filtered_df = datainput[datainput['batch'].isin(top3_targets)]

print(filtered_df,1)


# print(min_value)
exit()


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

print(yy_value,y1_value)
# exit()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print('read file finish')