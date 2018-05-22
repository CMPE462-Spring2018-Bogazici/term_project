import numpy as np
from skimage import io as io
import torch
import random

loc = ""
N , D_in , H , D_out = 36*1000 , 128*128 , 250 , 36

# Read Data and Save
# Randomly choose for train and test
x_train = np.zeros( (36*900 , D_in ), dtype = np.float)
y_train = np.zeros( 36*900 , dtype = np.long)
x_test = np.zeros( (36*100 , D_in) , dtype = np.float)
y_test = np.zeros( 36*100 , dtype = np.long)
k = 0
l = 0
for i in range(1 , 37):
    r = random.sample(range(1,1001) ,100)
    for  j in range(1 ,1001):
        img = io.imread(loc + "Sample" + str(i).zfill(3) + "/img" + str(i).zfill(3) + "-" + str(j).zfill(5) + ".png" , as_gray = True)
        img = img.flatten()
        img = list(map(lambda x: 1.0 if x>127 else 0.0 , img ))
        if j in r:
            x_test[k , : ] = np.array(img)
            y_test[k] = i-1
            k = k + 1
        else:
            y_train[l] = i-1
            x_train[l , : ] = np.array(img) 
            l = l + 1

np.save('xtrain' , x_train)
np.save('xtest' , x_test)
np.save('ytrain' , y_train)
np.save('ytest' , y_test)



# Load train data and train the NN
x_train = np.load('xtrain.npy')
y_train = np.load('ytrain.npy')
index = np.random.permutation(36*900)
x_train = torch.from_numpy(x_train[index , :])
y_train = torch.from_numpy(y_train[index])

model = torch.nn.Sequential(
    torch.nn.Linear(D_in , H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.LogSoftmax(),
)

model = model.double()
loss_fn = torch.nn.NLLLoss(size_average=False)
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for i in range(900):
    #y_pred = model(x)
    index =random.sample(range(0,36*900) , 400)
    loss = loss_fn(model(x_train[index , : ]) , y_train[index].type(torch.LongTensor))
    print( i , loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
torch.save(model , 'model.pt')




# Load model and test
x_test = np.load('xtest.npy')
y_test = np.load('ytest.npy')
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)
model = torch.load('model.pt')
with torch.no_grad():
    y_pred = model(x_test)
    k = 0
    for i in range(0,len(y_test)):
        if torch.argmax(y_pred[i , :] ).type(torch.int32) - y_test[i] != 0:
            k = k+1
    print(k , len(y_test))
    
