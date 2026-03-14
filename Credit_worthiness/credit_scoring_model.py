import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


df=pd.read_excel(r"C:\Users\acer\Downloads\archive (6)\CreditWorthiness.xlsx")
print(df.dtypes)


x=df.drop('creditScore',axis=1)
y=df['creditScore']
print(y.value_counts())

x.boxplot()
plt.show()

df_str=x.drop(x.select_dtypes(include=['int64']),axis=1)
df_ints=x.select_dtypes(include=['int64'])

le=LabelEncoder()
for cols in df_str:
  df_str[cols]=le.fit_transform(df_str[cols])
df_str

le=LabelEncoder()
y=le.fit_transform(y)

x=pd.concat([df_str,df_ints],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)


y_pred=model.predict(x_test)
y_pred_train=model.predict(x_train)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


user_inputs={}
df=pd.read_excel(r"C:\Users\acer\Downloads\archive (6)\CreditWorthiness.xlsx")
for cols in df.columns:
  if cols =='creditScore':
    print(' ')
  else:
    if df[cols].dtype == 'object':
     options=df[cols].dropna().unique()
     print(f"\nEnter Value for {cols}:")
     for i, opt in enumerate(options, start=1):
       print(f"{i} = {opt}")
     choice = int(input("Select option number: "))
     user_inputs[cols] = options[choice-1]
    
    else:  # numeric column
         max_val=df[cols].max()
         min_val=df[cols].min()
         value = float(input(f"Enter value for {cols}: between {min_val}  and {max_val}"))
         user_inputs[cols] = value

print("\nUser Input Data:")
print(user_inputs)



input_df = pd.DataFrame([user_inputs])

categorical_cols = df_str.columns
encoded_input_data = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df[col])
    encoded_input_data[col] = le.transform(input_df[col])

numerical_cols = df_ints.columns
for col in numerical_cols:
    encoded_input_data[col] = input_df[col]

X_user = pd.DataFrame(encoded_input_data)[x.columns]

X_user_array = X_user.values

prediction = model.predict(X_user_array)
if prediction[0] == 0:
  prediction ='bad'
else:
  prediction='Good'

print("\nPredicted Output:", prediction)
