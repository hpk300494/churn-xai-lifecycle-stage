"""Shared data preparation for the pooled-gap scripts: a copy of the notebook's data prep, lifecycle stages and
logistic-regression pipeline (same split, seed 42, engineered features, tenure-quartile stages)."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
RANDOM_STATE=42
import os, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA_PATH = pathlib.Path(os.environ.get('TELCO_XLSX', ROOT / 'Telco_customer_churn 2.xlsx'))
OUT = pathlib.Path(os.environ.get('POOLED_GAP_OUT', HERE / 'out')); OUT.mkdir(parents=True, exist_ok=True)
if not DATA_PATH.exists():
    raise SystemExit(f'Data file not found: {DATA_PATH}\nSee data/README.md, or set TELCO_XLSX=/path/to/file.xlsx')
df_raw=pd.read_excel(DATA_PATH)
DROP=['CustomerID','Count','Country','State','City','Zip Code','Lat Long','Latitude','Longitude','Churn Label','Churn Score','CLTV','Churn Reason']
df=df_raw.drop(columns=[c for c in DROP if c in df_raw.columns]).copy().rename(columns={'Tenure Months':'Tenure','Churn Value':'Churn'})
df['Total Charges']=pd.to_numeric(df['Total Charges'],errors='coerce')
X_full=df.drop(columns=['Churn']); y_full=df['Churn']
X_train_raw,X_test_raw,y_train,y_test=train_test_split(X_full,y_full,test_size=0.20,stratify=y_full,random_state=RANDOM_STATE)
ADDON=['Online Security','Online Backup','Device Protection','Tech Support','Streaming TV','Streaming Movies']
def collapse(X):
    X=X.copy()
    for c in ADDON: X[c]=X[c].replace('No internet service','No')
    X['Multiple Lines']=X['Multiple Lines'].replace('No phone service','No'); return X
def eng(X,fill,base):
    X=X.copy(); X['Total Charges']=X['Total Charges'].fillna(fill); X=collapse(X)
    X['Num Addon Services']=(X[ADDON]=='Yes').sum(axis=1)
    X['Avg Monthly Spend']=X['Total Charges']/(X['Tenure']+1)
    X['Charges Per Service']=X['Monthly Charges']/(X['Num Addon Services']+1)
    tb=X['Internet Service'].map(base).fillna(X['Monthly Charges'])
    X['Expansion Revenue Ratio']=((X['Monthly Charges']-tb)/X['Monthly Charges']).clip(lower=0)
    return X
tc=collapse(X_train_raw); zm=(tc[ADDON]=='No').all(axis=1)
base=tc.loc[zm].groupby('Internet Service')['Monthly Charges'].median().to_dict()
X_train=eng(X_train_raw,0.0,base); X_test=eng(X_test_raw,0.0,base)
q=X_train['Tenure'].quantile([.25,.5,.75]).values
edges=[-np.inf,q[0],q[1],q[2],np.inf]; ST=['Onboarding','Growth','Renewal','Maturity']
X_train['Stage']=pd.cut(X_train['Tenure'],bins=edges,labels=ST); X_test['Stage']=pd.cut(X_test['Tenure'],bins=edges,labels=ST)
CAT=['Gender','Senior Citizen','Partner','Dependents','Phone Service','Multiple Lines','Internet Service','Online Security','Online Backup','Device Protection','Tech Support','Streaming TV','Streaming Movies','Contract','Paperless Billing','Payment Method']
NUM=['Tenure','Monthly Charges','Total Charges','Num Addon Services','Avg Monthly Spend','Charges Per Service','Expansion Revenue Ratio']
FEAT=CAT+NUM
def prep(): return ColumnTransformer([('num',StandardScaler(),NUM),('cat',OneHotEncoder(handle_unknown='ignore',drop='if_binary'),CAT)])
def lr_pipe(): return Pipeline([('prep',prep()),('clf',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=RANDOM_STATE))])
def rf_pipe(): return Pipeline([('prep',prep()),('clf',RandomForestClassifier(class_weight='balanced',random_state=RANDOM_STATE,n_jobs=1))])
GRIDS={'LogisticRegression':(lr_pipe,{'clf__C':[0.1,1.0,10.0]}),'RandomForest':(rf_pipe,{'clf__n_estimators':[200],'clf__max_depth':[6,None]})}
def vdata(v):
    if v=='Pooled': return X_train[FEAT],y_train,X_test[FEAT],y_test
    a=X_train['Stage']==v; b=X_test['Stage']==v
    return X_train.loc[a,FEAT],y_train.loc[a],X_test.loc[b,FEAT],y_test.loc[b]
VARS=['Pooled']+ST
