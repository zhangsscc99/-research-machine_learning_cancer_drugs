import pandas as pd
df=pd.read_csv("main_folder(ks4)/drug_with_fields.csv")
df2=df.iloc[0,1:20]
df3=df.iloc[1,1:20]
print(df2.shape,df3.shape)