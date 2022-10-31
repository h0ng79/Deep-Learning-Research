import numpy as np
import random
#from keras.models import model_from_json
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import pandas
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler,minmax_scale,normalize

dataframe = pandas.read_csv('Data_I_corr_1.csv')
dataset = dataframe.values


data =  dataset[:,0:3]
scaler = MinMaxScaler()
data1 = scaler.fit_transform(data)


nb_set = 69
nb_test=int(nb_set*0.8)

x_train = data1[:nb_test,0:2]
x_test  = data1[nb_test:,0:2]
x = data1[:,0:2]
print('x_train',x_train)

y_train = data1[:nb_test,2:3]
y_test  = data1[nb_test:,2:3]
y = data1[:,2:3]


#reshpae input data
x_train = x_train.reshape(x_train.shape[0], 1,x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0],1, x_test.shape[1])
x = x.reshape(x.shape[0],1, x.shape[1])


model = model_from_json(open('model.json').read())
model.load_weights('target_weight.h5')
pred_test = model.predict(x)


plt.plot(y,'ro')
plt.plot(pred_test,'b+')
plt.title('Prediction')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
plt.legend(['Actual', 'Prediction'], loc='lower right')
plt.show()

