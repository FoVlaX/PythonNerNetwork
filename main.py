import numpy as np
from PIL import Image
from function import DD
from NeronNetwork import NN
import math
from NeronNetwork import SNN
r = []
g = []
b = []

print('Create image....')
hcnt = 30
for i in range(hcnt):
    r.append(0)
    g.append(0)
    b.append(0)
    img = Image.open('images/image ('+str(1+i)+').jpg')
    I = np.asarray(img,dtype='uint8')
    r[i],g[i],b[i] = DD.ConToData(I)
    print(i+1)

for i in range(30):
    r.append(0)
    g.append(0)
    b.append(0)
    img = Image.open('images1/image ('+str(1+i)+').jpg')
    I = np.asarray(img,dtype='uint8')
    r[i+hcnt],g[i+hcnt],b[i+hcnt] = DD.ConToData(I)
    print(i+1)



nuclear = np.array([2])


y = np.array([
    [1], #cat
    [0] #notcat
])
ny = np.array([
    [0], #cat
    [1] #notcat
])

networkR = SNN(r[0],128,128,nuclear,y,1000,1)
networkG = SNN(g[0],128,128,nuclear,y,1000,1)
networkB = SNN(b[0],128,128,nuclear,y,1000,1)




print('Teaching..')


for i in range(5):
    for j in range(30):
        networkR.Teach( r[j], y, 128,128)
        networkG.Teach( g[j], y, 128,128)
        networkB.Teach( b[j], y, 128,128)
        networkR.Teach( r[j+hcnt], ny, 128,128)
        networkG.Teach( g[j+hcnt], ny, 128,128)
        networkB.Teach( b[j+hcnt], ny, 128, 128)
        print('   ',j)
    print(i)

print('retecahre...\n\n\n')
networkR.save()
networkG.save()
networkB.save()

'''
networkR.load()
networkG.load()
networkB.load()
'''


r1,r2,r3,r4,r5 = networkR.feedforward(r[0],128,128)
g1,g2,g3,g4,g5 = networkG.feedforward(g[0],128,128)
b1,b2,b3,b4,b4 = networkB.feedforward(b[0],128,128)





print(r1[1][2])


