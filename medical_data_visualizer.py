import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('medical_examination.csv')
df_ovr_wt = pd.read_csv('medical_examination_with_overwt.csv')
df_nor_cg = pd.read_csv('medical_examination_with_nor_cg.csv')
df_cln = pd.read_csv('medical_examination_cleaned.csv')

#df_info = df.info()
#print(df_info)

#function which works for adding a column labeled as overweight with calculation patients bmi!
def overweight():
    height_in_m = df['height'] / 100
    bmi = df['weight'] / (height_in_m ** 2)
    df['overweight'] = (bmi > 25).astype(int)

    df.to_csv('medical_examination_with_overwt.csv', index=False)
    return df.head(10)

#function to work normalization on cholesterol and gluc column based on their values!
def normal_cg():
    df_ovr_wt['cholesterol'] = (df_ovr_wt['cholesterol'] > 1).astype(int)
    df_ovr_wt['gluc'] = (df_ovr_wt['gluc'] > 1).astype(int)

    df_ovr_wt.to_csv('medical_examination_with_nor_cg.csv', index=False)

    return df_ovr_wt[['cholesterol', 'gluc']].head(10)

#function to work on cleaning process of dataframe within specific conditions!
def clean():
    condition_1 = df_nor_cg['ap_lo'] <= df_nor_cg['ap_hi']

    min_height = df_nor_cg['height'].quantile(0.025)
    max_height = df_nor_cg['height'].quantile(0.975)

    condition_2 = (df_nor_cg['height'] >= min_height) & (df_nor_cg['height'] <= max_height)

    min_weight = df_nor_cg['weight'].quantile(0.025)
    max_weight = df_nor_cg['weight'].quantile(0.975)

    condition_3 = (df_nor_cg['weight'] >= min_weight) & (df_nor_cg['weight'] <= max_weight)
    
    combined_condition = condition_1 & condition_2 & condition_3

    cleaned_df = df_nor_cg[combined_condition]

    cleaned_df.to_csv('medical_examination_cleaned.csv')

    return cleaned_df

#function to work on melt the data frame columns without cardio, and displaying it's plot graph based on cardio and it's values
def melt_plot():
    
    melt_df = pd.melt(df_cln, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
                        var_name='feature', value_name='value')
    grp_df = melt_df.groupby(['cardio', 'feature', 'value']).size().reset_index(name='count')

    sns.catplot(data= grp_df, 
                    x= 'feature', y= 'count', hue= 'value', col= 'cardio', kind= 'bar', height= 5, aspect= 1)

    plt.show()

def corr_matrix_plot():
    corr_mat = df_cln.corr()

    upper_triangle_mask = np.triu(np.ones_like(corr_mat, dtype=bool))

    plt.figure(figsize=(10,8))

    sns.heatmap(corr_mat, mask= upper_triangle_mask, annot= True, fmt= ".2f", cmap= 'coolwarm', square= True, linewidths= .5, cbar_kws= {"shrink": .5})

    plt.show()