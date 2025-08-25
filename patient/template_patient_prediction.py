import rdt
from patient_data import TabularPatientBase
from tabular_utils import read_csv_to_df
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
# For Individual Patient Outcome Prediction Task (Tabular Patient: Index)
df = read_csv_to_df("demo_data/patient/tabular/patient_tabular.csv")
# Note that read_csv_to-df will automatically convert column names to lowercase
# Custom Metadata Conversion
print(df.head(),"\n",df.shape)
'''
##去掉没有年龄的
mask_valid_age = df["age"].notna() & (df["age"] > 0)
df = df.loc[mask_valid_age].reset_index(drop=True)
##去掉subject_id
if "subject_id" in df.columns:
    df = df.drop(columns=["subject_id"])
print(df.head(),"\n",df.shape)
##gender转换成boolean
df["gender"] = (
    df["gender"].astype("string").str.strip().str.upper()
    .map({"M": True, "F": False})
)
'''

#数据类型转换
patient_data_custom = TabularPatientBase(df,
    metadata={
        'transformers': {
        'gender': rdt.transformers.UniformEncoder(),
        'mortality': rdt.transformers.BinaryEncoder(),
        'ethnicity': rdt.transformers.OneHotEncoder(),
    },
    })
print(patient_data_custom.df.head())
##简单的死亡分类预测
X = patient_data_custom.df
y = df.pop("mortality")
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
model = XGBClassifier(random_state=0)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print(acc)
'''
# Auto Metadata Conversion
patient_data_auto = TabularPatientBase(df=df)
print(patient_data_auto.df.head())
# Restore the Original Metadata
df_reversed = patient_data_custom.reverse_transform()
print(df_reversed.head())
'''




