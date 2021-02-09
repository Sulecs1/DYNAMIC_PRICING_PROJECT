######################################################################
#                  DYNAMIC PRICING PROJECT                           #
######################################################################
#<<<Şule AKÇAY >>>

#kütüphaneleri ekledim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from scipy import stats
from itertools import combinations


#eklentiler eklendi
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#veri setini çektim
data = pd.read_csv(r"C:\Users\Suleakcay\PycharmProjects\pythonProject6\datasets\pricing.csv", sep=";")
df = data.copy()
#veri setinde aykırı değer olduğu için median değeri tercih edildi
df.describe([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df.groupby("category_id").agg({"price":{"median"}})

####################
# Data Preparing   #
####################

def outlier_thresholds(dataframe, variable):
    quartileone = dataframe[variable].quantile(0.10)
    quartilethree = dataframe[variable].quantile(0.90)
    interquantile_range = quartilethree - quartileone
    up_limit = quartilethree + 1.5 * interquantile_range
    low_limit = quartileone - 1.5 * interquantile_range
    return low_limit, up_limit


def has_outliers(temp_df, numeric_columns):
    for columns in numeric_columns:
     low_limit, up_limit = outlier_thresholds(temp_df, columns)
     if temp_df[(temp_df[columns] > up_limit) | (temp_df[columns] < low_limit)].any(axis=None):
        outliersnumber = temp_df.loc[(temp_df[numeric_columns] > up_limit), numeric_columns] = up_limit
        print(columns, ":", outliersnumber, "outliers")

def remove_outliers(temp_df, numeric_columns):
    for col in numeric_columns:
        low_limit, up_limit = outlier_thresholds(temp_df, col)
        dataframe_without_outliers = temp_df[~((temp_df[col] < low_limit) | (temp_df[col] > up_limit))]
    return dataframe_without_outliers

df = remove_outliers(df, ["price"])


def abtestfunc(dataframe, first_group, second_group):
    group_A_df = dataframe.loc[dataframe["category_id"] == first_group, "price"]
    group_B_df = dataframe.loc[dataframe["category_id"] == second_group, "price"]

    norm_A = stats.shapiro(group_A_df)[1] >= 0.05
    norm_B = stats.shapiro(group_B_df)[1] >= 0.05
    if norm_A & norm_B:
        var_group = stats.levene(group_A_df, group_B_df)[1] >= 0.05
        if var_group:
            ab_test1 = stats.ttest_ind(group_A_df, group_B_df, equal_var=True)[1] >= 0.05
            if ab_test1:
                print(f'{first_group}&{second_group} itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark yoktur!')
            else:
                print(
                    f'{first_group}&{second_group} itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark vardır!')
        else:
            ab_test2 = stats.ttest_ind(group_A_df, group_B_df, equal_var=False)[1] >= 0.05
            if ab_test2:
                print(
                    f'{first_group}&{second_group} itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark yoktur!')
            else:
                print(
                    f'{first_group}&{second_group} itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark vardır!')

    else:
        ab_test3 = stats.mannwhitneyu(group_A_df, group_B_df)[1] >= 0.05
        if ab_test3:
            print(
                f'{first_group}&{second_group} itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark yoktur!')
        else:
            print(
                f'{first_group}&{second_group} itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark vardır!')

df.groupby("category_id").agg({'price':['mean', 'median', 'count', 'std']})

grouplist = [i for i in combinations(df["category_id"].unique(), 2)]

for group in grouplist:
    abtestfunc(df, group[0], group[1])


#output
#489756&361254 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark vardır!
#489756&874521 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark vardır!
#489756&326584 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark vardır!
#489756&675201 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark vardır!
#489756&201436 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark vardır!
#361254&874521 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark yoktur!
#361254&326584 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark vardır!
#361254&675201 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark yoktur!
#361254&201436 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark yoktur!
#874521&326584 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark vardır!
#874521&675201 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark yoktur!
#874521&201436 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark yoktur!
#326584&675201 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark vardır!
#326584&201436 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark vardır!
#675201&201436 itemlerinin fiyat ortalamaları arasında istatistiksel açıdan fark yoktur!


#Confidence Interval
sms.DescrStatsW(df["price"]).tconfint_mean()
#(38.334045331672925, 39.216612629527816)
#category_id{201436,326584,361254,489756,675201,874521 } olan itemlerin
#489756,326584,675201 itemlerinin fiyatlarının
#güven aralığında olması gerekir.


#sümilasyon
#minimum kazanç için
minfreq = len(df[df['price'] >= 35.693170])
income_min = minfreq * 35.6931
income_min

#ortalama kazanç
meanfreq = len(df[df['price'] >= 37.443592])
income_mean = minfreq * 37.4435
income_mean

#maksimum kazanç
maxfreq = len(df[df['price'] >= 41.294823])
income_max = maxfreq * 41.2948
income_max
print('Minimum kazanç:{}, Ortalama Kazanç: {}, Maksimum Kazanç :{}'.format(income_min, income_mean, income_max))

#İtemlera güven aralığına göre fiyatlandırma yapılması değişken değil sabit fiyata göre olması bu iki şartı sağlamasıfaydalı olur.



