import numpy as np
from rbm import RBM 

r = RBM(num_visible = 1682, num_hidden = 20)
ubase = open('ua.base', 'r+')
newTrain = open('train.txt','w+')
arrayTrain = np.ndarray(shape=(943,1682), dtype=int)
for line in ubase:
	content=line.strip().split("\t")
	a, b, c, d = [int(i) for i in content]
	if(c <3):
		c1 = 0
	else:
		c1 = 1
	newTrain.write('%d\t%d\t%d\n' % (a, b, c1))
	arrayTrain[a-1,b-1]=c1

r.train(arrayTrain, max_epochs = 500)
print(r.weights)

utest = open('ua.test', 'r+')
newTest = open('test.txt','w+')
arrayTest = np.ndarray(shape=(943,1682), dtype=int)
for line in utest:
    content=line.strip().split("\t")
    a, b, c, d = [int(i) for i in content]
    if(c <3):
        c1 = 0
    else:
        c1 = 1
    newTest.write('%d\t%d\t%d\n' % (a, b, c1))
    arrayTest[a-1,b-1]=c1

for user in range(1, 943):
    print(r.run_visible(arrayTest[user-1]))


	
