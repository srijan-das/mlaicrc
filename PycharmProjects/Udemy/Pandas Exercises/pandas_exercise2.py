import pandas as pd
import numpy as np

ecom = pd.read_csv("/home/srijan/PycharmProjects/mlaicrc/PycharmProjects/Udemy/Pandas Exercises/Ecommerce Purchases.csv")
print(ecom.head(5))

print(ecom.info())

print(ecom['Purchase Price'].mean())
print(ecom['Purchase Price'].max())
print(ecom['Purchase Price'].min())

print(ecom[ecom['Language'] == 'en'].count())

print(ecom[ecom['Job'] == 'Lawyer'].count())

print(ecom['AM or PM'].value_counts()) # IMP

print(ecom['Job'].value_counts().head(5))

print(ecom[ecom['Lot'] == '90 WT']['Purchase Price'])

print(ecom[ecom['Credit Card'] == 4926535242672853]['Email'])

print(ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95)].count())

print(sum(ecom['CC Exp Date'].apply(lambda x: x[3:]) == '25'))

print(ecom['Email'].apply(lambda x: x.split('@')[1]).value_counts().head(5))