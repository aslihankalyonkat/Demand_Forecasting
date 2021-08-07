import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

################################################    OBJECTİVE    ################################################
# A chain of stores wants a 3-month demand forecast for its 10 different stores and 50 different products.
#################################################################################################################

################################################  ABOUT DATASET  ################################################
# This dataset is presented to test different time series techniques.
# A store chain's 5-year data includes information on 10 different stores and 50 different products.
#################################################################################################################

################################################    VARIABLES    ################################################
# date  – Date of sales data (No holiday effects or store closures.)
# Store – Store ID (Unique number for each store)
# Item  – Product ID (Unique number for each product)
# Sales – Number of items sold (The number of items sold from a particular store on a given date)
#################################################################################################################

##### READ DATASET #####
train = pd.read_csv('data/train.csv', parse_dates=['date'])
test = pd.read_csv('data/test.csv', parse_dates=['date'])
df = pd.concat([train, test], sort=False)

#####################################################
# EDA
#####################################################
def check_dataframe(dataframe, row=5):
    print('####################  Shape   ####################')
    print(dataframe.shape)
    print('####################  Types   ####################')
    print(dataframe.dtypes)
    print('####################   Head   ####################')
    print(dataframe.head(row))
    print('####################   Tail   ####################')
    print(dataframe.tail(row))
    print('####################    NA    ####################')
    print(dataframe.isnull().sum())
    print('#################### Describe ####################')
    print(dataframe.describe().T)

check_dataframe(df)

##### Min and max date #####
df["date"].min(), df["date"].max()
# (Timestamp('2013-01-01 00:00:00'), Timestamp('2018-03-31 00:00:00'))

##### Sales distribution #####
df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])
# count    913000.000000
# mean         52.250287
# std          28.801144
# min           0.000000
# 10%          20.000000
# 30%          33.000000
# 50%          47.000000
# 70%          64.000000
# 80%          76.000000
# 90%          93.000000
# 95%         107.000000
# 99%         135.000000
# max         231.000000

##### Number of stores #####
df[["store"]].nunique() # 10

##### Number of items #####
df[["item"]].nunique() # 50

##### Are there an equal number of unique items in each store? #####
df.groupby(["store"])["item"].nunique()
# Each store has 50 unique item

##### Is the number of sales equal in each store? #####
df.groupby(["store", "item"]).agg({"sales": ["sum"]})
# The sales numbers of the items in the stores are different.
# e.g store item        sum
#       1     1      36468.0
#       10    1      45168.0

##### Sales statistics in store-item breakdown #####
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

#####################################################
# FEATURE ENGINEERING
#####################################################

##### Date Features #####
def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    df['quarter'] = df.date.dt.quarter
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype(int)
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype(int)
    df['week_passed'] = np.floor((df.date - pd.to_datetime('2012-12-31')).dt.days / 7)
    df['week_passed'] = df['week_passed'].astype(int)
    df['quarter_passed'] = (df['year'] - 2013) * 4 + df['quarter']
    return df


df = create_date_features(df)

##### Random Noise #####
def random_noise(dataframe):
    return np.random.normal(scale=0.01, size=(len(dataframe)))

##### Lag/Shifted Features #####
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

##### Rolling Mean Features #####
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [91, 182, 365, 546, 728])

##### Exponentially Weighted Mean Features #####
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)

##### One-Hot Encoding #####
df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

##### Converting sales to log(1+sales) #####
df['sales'] = np.log1p(df["sales"].values)


#####################################################
# Model
#####################################################

##### Custom Cost Function #####
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

##### Time-Based Validation Sets #####

# Train set until the beginning of 2017 (until the end of 2016).
train = df.loc[(df["date"] < "2017-01-01"), :]

# First 3 months of 2017 as validation set.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

# controling shape
Y_train.shape, X_train.shape, Y_val.shape, X_val.shape
# ((730500,), (730500, 150), (45000,), (45000, 150))

########################
# LightGBM Model
########################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'boosting_type': 'gbdt',
              'objective': 'regression_l1',
              'max_depth': 7,
              'num_leaves': 28,
              'learning_rate': 0.05,
              'feature_fraction': 0.9,
              'bagging_fraction': 0.8,
              'verbose': 0,
              'num_boost_round': 10000,
              'bagging_freq': 5,
              'early_stopping_rounds': 500,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

# [780]	training's l1: 0.123987	training's SMAPE: 12.7335	valid_1's l1: 0.132893	valid_1's SMAPE: 13.6568

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))
# 13.656775324435896

##### Feature Importance #####
def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.savefig("feature_importances.png")
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30, plot=True)


########################
# Final Model
########################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'metric': {'mae'},
              'boosting_type': 'gbdt',
              'objective': 'regression_l1',
              'max_depth': 7,
              'num_leaves': 28,
              'learning_rate': 0.05,
              'feature_fraction': 0.9,
              'bagging_fraction': 0.8,
              'verbose': 0,
              'num_boost_round': model.best_iteration,
              'bagging_freq': 5,
              'nthread': -1}

##### LightGBM dataset #####
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
test_preds = model.predict(X_test, num_iteration=model.best_iteration)

##### Create submission #####
submission_df = test.loc[:, ['id', 'sales']]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)
submission_df.to_csv('submission_demand.csv', index=False)