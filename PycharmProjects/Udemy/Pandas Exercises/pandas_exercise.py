import pandas as pd
import numpy as np

data = pd.read_csv("/home/srijan/PycharmProjects/mlaicrc/PycharmProjects/Udemy/Pandas Exercises/Salaries.csv")

print(data.head())

print(data.info())

print(data['BasePay'].mean())

print(data['OvertimePay'].max())

print(data[data['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle'])

print(data[data['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits'])

print(data[data['TotalPayBenefits'] == data['TotalPayBenefits'].max()])

print(data[data['TotalPayBenefits'] == data['TotalPayBenefits'].min()])

print(data.groupby('Year').mean()['BasePay'])

print(data['JobTitle'].nunique())

print(data['JobTitle'].value_counts().head(5))

print(sum(data[data['Year'] == 2013]['JobTitle'].value_counts() == 1)) #important

def isChief(title) :
    if 'chief' in title.lower() :
        return True
    else :
        return False
print(sum(data['JobTitle'].apply(lambda x: isChief(x))))

data['title_len'] = data['JobTitle'].apply(len)
print(data[['title_len', 'TotalPayBenefits']].corr())