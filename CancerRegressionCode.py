import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd


dataframe = pd.read_csv('/Users/kendalllucas/Documents/Regression Project/data/U.S._Chronic_Disease_Indicators (1).csv')

alcohol_df = dataframe[dataframe["Topic"].str.contains("Alcohol", na = False)]
alcohol_df = alcohol_df[alcohol_df["Question"].str.contains("Binge drinking", na = False)]
alcohol_df = alcohol_df[alcohol_df["DataValueType"].str.contains("Crude Prevalence", na = False)]
#alcohol_df = alcohol_df[alcohol_df["YearStart"]==2019]

cancer_df = dataframe[dataframe["Question"].str.contains("cancer", case= False, na= False)]
cancer_df= cancer_df[cancer_df["DataValueType"].str.contains("Crude Prevalence", na= False)]
cancer_df = cancer_df[cancer_df["YearStart"]==2020]
alcohol_df = alcohol_df[alcohol_df["YearStart"]==2020]

#print(set(alcohol_df["YearStart"]).intersection(set(cancer_df["YearStart"])))


merged = pd.merge(
    alcohol_df,
    cancer_df,
    on = ["LocationDesc", "YearStart"],
    how= "inner", 
    suffixes=("_binge", "_cancer")
)



merged = merged[["LocationDesc", "DataValue_binge", "DataValue_cancer"]]
merged = merged.dropna()

merged["DataValue_binge"] = pd.to_numeric(merged["DataValue_binge"], errors="coerce")
merged["DataValue_cancer"] = pd.to_numeric(merged["DataValue_cancer"], errors="coerce")
merged = merged.groupby("LocationDesc").mean().reset_index() 


plt.scatter(merged["DataValue_binge"], merged["DataValue_cancer"],  alpha = 0.5)
m, b = np.polyfit(merged["DataValue_binge"], merged["DataValue_cancer"], 1)

plt.plot(
    merged["DataValue_binge"],
    m*merged["DataValue_binge"] + b,
    color = "black"
)


plt.xlabel("Binge Drinking Prevalence")
plt.ylabel("Cancer Prevalence")
plt.title("Binge Drinking vs Cancer by State/Year")
plt.show()

print(merged.head())