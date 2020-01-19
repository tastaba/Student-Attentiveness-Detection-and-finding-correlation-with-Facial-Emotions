import os
path = 'C:/Users/Tasnia Tabassum/Desktop/test_attentive'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, 'test_attentive_' + str(i)+'.jpg'))
    i = i+1