import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

data = pd.read_csv("data/2020-01-10.csv")

print(data.head())

#----------DROPPING COLS-------------------#
a = data.isnull().mean() * 100
cols_to_drop = a[a > 30].index.tolist()
df = data.drop(columns=cols_to_drop)

print(f"\n Dropped {len(cols_to_drop)} columns: {cols_to_drop}\n")
print(f"{data.shape} -> {df.shape}")

#-----------EDA---------------#
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
str_cols = df.select_dtypes(include=['object']).columns.tolist()
print(len(num_cols))
print("\n BREAK \n")
print(len(str_cols))
print("\n REMAINING NULLS.. \n")
print(df.isnull().sum())
print(df.isnull().mean())

#----USEFUL COLS-----------#

X = df.iloc[: , 5:]
Y = df.iloc[: , 4:5]

# print("\n X HEAD: \n")
# print(X.head())
# print("\n Y HEAD: \n")
# print(Y.head())

imputer = SimpleImputer(strategy="most_frequent")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)


# scores = mutual_info_classif(X_imputed,Y, random_state=42)
# mi_df = pd.DataFrame({
#     "Feature": X_imputed.columns,
#     "MI_SCORES": scores
# }).sort_values(by="MI_SCORES",ascending=False)

# print(mi_df)

new_df = X_imputed.filter(like="normalized")
useless_cols = [
    # Identifier (causes overfitting / data leakage
    
    # Power & Retract cycles (near-zero variance in 24/7 datacenter drives)
     'smart_4_normalized',
     'smart_12_normalized',
     'smart_192_normalized',
     'smart_193_normalized',
    
    # Sparsely populated or vendor-specific / uninformative metrics
     'smart_240_normalized',
     'smart_241_normalized',
     'smart_242_normalized',
     
    
]
new_df = new_df.drop(columns=useless_cols)
print(new_df.head())
print(new_df.shape)
print(new_df.isnull().sum())
print(Y.isnull().sum())

final_df = new_df.copy()
final_df['failure'] = Y

final_df.to_csv("train_data_bz.csv",index=False)






