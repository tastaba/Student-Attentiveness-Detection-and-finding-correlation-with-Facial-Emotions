import numpy as np
##Combining with the AWS emotion awsEmotion.csv
import pandas as pd
a = pd.read_csv("classifier.csv")
b = pd.read_csv("awsEmotion.csv")
print(a)
print(b)
merged = pd.merge(a, b, on='ImageId')
print(merged)
print(merged.columns)
print(len(merged.columns))

##write to csv
merged.to_csv('combined.csv', index=False)
