#%%
import pandas as pd
import numpy as np

# %%
df_raw = pd.read_csv('model_data.csv')
df_raw.head(10)

# %%
print(list(df_raw))

# %%
cols_drop =['PERMIT_', 'STNOFROM','STNOTO','STNAME','SUFFIX', 'PLACEMENT','DIGDATE','EXPIRATIONDATE','DIGGINGDONE','PRIMARYCONTACTFIRST','PRIMARYCONTACTLAST','EMERGENCY','FullRdName', 
 'REQUESTDATE_Year','Census_Tract','tractce10','populationtotals_totpop_cy','community']

update_df= df_raw.drop(cols_drop,axis=1)
update_df.info()

# %%
update_df['REQUESTDATE_Month'] = pd.Categorical(update_df['REQUESTDATE_Month'])
update_df['Emergency2'] = pd.Categorical(update_df['Emergency2'])
update_df['UrbSubRural'] = pd.Categorical(update_df['UrbSubRural'])

# %%
from sklearn.preprocessing import LabelEncoder
labelEnconder = LabelEncoder()
update_df['DIRECTION_ENC']= labelEnconder.fit_transform(update_df['DIRECTION'])
update_df=update_df.drop(['DIRECTION'],axis=1)
print(list(update_df))
update_df.head()

# %%
y= update_df['DamageRisk']
x_cols=['Emergency2','UrbSubRural','DistToWaterSource','NeigY','NeigY']
x = update_df[x_cols]

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=101)

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=50)
reg.fit(x_train,y_train)
y_test_pred = reg.predict(x_test)

# %%
from sklearn.metrics import r2_score, mean_squared_error 
rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))
rmse

# %%
r2 = r2_score(y_test,y_test_pred)
r2

# %%
feature_importance = reg.feature_importances_
print("Importance of Features:")
for i, dt in enumerate(x.columns.to_list()):
    print("{}. {} ({})".format(i+1, dt, feature_importance[i]))

# %%
from sklearn import linear_model
lreg = linear_model.LinearRegression()
lreg.fit(x_train,y_train)


# %%
y_pred = lreg.predict(x_test)

# %%
rmseLreg = np.sqrt(mean_squared_error(y_test,y_pred))
rmseLreg
# %%
r2lreg = r2_score(y_test,y_pred)
r2lreg
# %%
