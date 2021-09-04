import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

"will need changing if files uploaded to GCP or AWS bucket"

file = r'/Users/anand/Documents/Sleep/WSC - variable cross-check_sparse.xlsx'
data = r'/Users/anand/Documents/Sleep/wsc-dataset-0.2.0.csv'

def get_data():
    """
    returns df/excel indicating which columns to keep and data_df/csv which is main data frame
    """
    df = pd.read_excel(file)
    data_df = pd.read_csv(data)
    return df, data_df


def clean_data(df, data_df, agethresh, duplicates = True):

    "Remove specific columns according to excel sheett"

    deleted = df[df['Proposed Removal'] == 'R']
    deleted_cols = deleted.iloc[:, 0]
    data_df = data_df.drop(deleted_cols.to_list(), axis=1)
    data_df.set_index('wsc_id', inplace=True)

    "Drop duplicates if duplicates = False"

    if duplicates == False:
        data_df.drop_duplicates('wsc_id', inplace=True)

    "Set nans to zero for certain variables"

    data_df.nasal_cong_none.replace({np.nan:0,'Y':1}, inplace=True)
    data_df.num_pregnancies.replace({np.nan:0}, inplace=True)
    data_df.packs_week.replace({np.nan:0}, inplace=True)
    data_df.pack_years.replace({np.nan:0}, inplace=True)

    "Filter the datafram according to agethresh"

    data_df.drop(index=data_df[(data_df.age > agethresh)].index, inplace=True)

    return data_df


def preproc_gen(data_df):

    objlist = []

    for n in data_df.dtypes[data_df.dtypes == 'object'].index:
        objlist.append(n)


    "Binariser -  should work if nans are present or not"

    for i, v in enumerate(objlist):

        ##columns with 2 variables eg. [N,Y] or [M,F]

        if len(data_df[v].unique()) == 2:
            ##print(data_df[v].unique(),v)
            data_df[objlist[i]].replace\
            ({data_df[objlist[i]].unique()[0]:0,data_df[objlist[i]].unique()[1]:1}, inplace=True)

        #### ALL columns with 3 variables - which appear like [N,Y,nan]
        if len(data_df[v].unique()) == 3:
            ##print(data_df[v].unique(),v)
            data_df[objlist[i]].replace\
            ({'N':0,'Y':1}, inplace=True)

    ohe = OneHotEncoder(handle_unknown='ignore',sparse=False)


    ## Only variables which need OHE
    X1 = data_df[['thyroid_problem']]
    X2 = data_df[['hormone_therapy']]

    ##fit transform, extract column names, make dataframe with column names, drop nan row

    X1t = ohe.fit_transform(X1)
    colnames = list(ohe.get_feature_names())
    X1df = pd.DataFrame(X1t, columns = colnames)
    X1df.drop(columns = 'x0_nan', inplace=True)
    X1df.index = data_df.index

    X2t = ohe.fit_transform(X2)
    colnames = list(ohe.get_feature_names())
    X2df = pd.DataFrame(X2t, columns = colnames)
    X2df.drop(columns = 'x0_nan', inplace=True)
    X2df.index = data_df.index

    frames = [data_df, X1df, X2df]
    data_df = pd.concat(frames, axis = 1)

    ##drop original row names

    data_df.drop(columns = ['thyroid_problem','hormone_therapy'], inplace=True)

    "Set targets"

    targs = ['sleep_latency', 'tst', 'tst_rem', 'tst_nrem', 'tso', 'totsleep',
        'ess', 'p_eval_sleep', 'a_eval_slept', 'a_eval_hour', 'a_eval_sleep',
        'ps_eds', 'se', 'waso', 'sleepiness', 'workday', 'weekend', 'ps_diff',
        'ps_diff', 'ps_backsleep', 'ps_wakerepeat', 'ps_wakeup', 'ps_eds']

    imputenum = SimpleImputer(strategy='median')
    data_df[targs] = imputenum.fit_transform(data_df[targs])

    return data_df, targs


def preproc_x(data_df, targs, balthresh):

    "drop all targets from X"

    X = data_df.drop(columns=targs)
    cols = X.columns

    "set numerical features"

    numeric_features = [
        'hdl', 'ldl', 'total_cholesterol', 'triglycerides', 'weightkg',
        'hipgirthm', 'neckgirthm', 'waistgirthm', 'waisthip', 'sitsysm',
        'sitdiam', 'packs_week', 'pack_years', 'naps', 'snore_freq',
        'num_pregnancies', 'age', 'heightcm', 'caffeine', 'alcohol_wk',
        'eval_general', 'eval_life', 'eval_health', 'snore_vol', 'choke_freq',
        'apnea_freq', 'awake_freq'
    ]

    "set categorical features as features which aren't numerical or target"

    categoric = data_df.drop(columns=targs)
    categoric.drop(columns=numeric_features, inplace=True)
    categorical_features = categoric.columns

    "categorical transformer"

    categorical_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='median'))])

    "numerical transformer"

    numeric_transformer = Pipeline(
        steps=[('imputer',
                SimpleImputer(strategy='mean')), ('scaler', RobustScaler())])

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),\
        ('cat', categorical_transformer, categorical_features)])

    prepipe = Pipeline(steps=[('preprocessor', preprocessor)])

    X = prepipe.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    X.drop(columns=['wsc_vst'], inplace=True)

    "remove imbalanced features according to balthresh"

    imbalanced_classes = []
    for col in X.columns:
        _ = X.columns.get_loc(col)
        if X.iloc[:,
                  _].value_counts(normalize=True).head(1).values > balthresh:
            imbalanced_classes.append((col, X.iloc[:, _].value_counts(
                normalize=True).head(1).values.astype(float)))

    imbalanced_list = []
    for classes in imbalanced_classes:
        imbalanced_list.append(classes[0])

    X.drop(imbalanced_list, axis=1, inplace=True)

    return X

def preproc_y(data_df, target, a, b):

    "target options are 'se', 'waso', 'tst"
    "a and b are classification thresholds based on normal parameters decided for that target"

    if target == 'waso':
      y = data_df.waso
      y = y.reset_index(drop=True)
      y[(y <= a)] = 0
      y[(y > a)] = 1

    elif target == 'se' or target == 'tst':
      y = data_df[target]
      y = y.reset_index(drop=True)

      y[(y < a) | (y > b)] = 1
      y[(y >= a) & (y <= b)] = 0

    return y


# if __name__ == '__main__':
#     df = get_data()
