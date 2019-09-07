import numpy as np
import math
class NN:
    def __init__(self, x, y,nCount,deepCount):
        self.input = x
        self.DC = deepCount+2
        self.y = y
        self.W = []
        self.b = []
        for i in range(deepCount):
            if i ==0:
                self.W.append(np.random.rand(nCount,self.input.shape[0]))
            else:
                self.W.append(np.random.rand(nCount, nCount))
            self.W[i]/=10
        self.W.append(np.random.rand(self.y.shape[0],nCount))
        self.al = 0.5
        for i in range(deepCount):
            self.b.append(np.random.rand(nCount,1))
            self.b[i]/=10
        self.b.append(np.random.rand(self.y.shape[0],1))
        self.d_W = []
        self.d_b = []
        for i in range(deepCount+1):
            self.d_W.append(0)
            self.d_b.append(0)
    def SD(self,x):
        return 1/(1 + np.exp(-x))
    def d_SD(self,x):
        return self.SD(x)*(1 - self.SD(x))
    def feedforward(self,x1):
        f = []
        h = []
        h.append(x1)
        f.append(x1)
        for i in range(self.DC):
            if i!=0:
                h.append(np.dot(self.W[i-1],f[i-1])+self.b[i-1])
                f.append(self.SD(h[i]))

        return h, f

    def Teach(self,x,y):
        h,f = self.feedforward(x)
        l = 1
        Y = 2*(y - f[self.DC-1])*self.d_SD(h[self.DC-1])
        while (self.DC - l > 0):
            self.d_b[self.DC-1-l] = Y
            self.d_W[self.DC-1-l] = np.dot(Y,f[self.DC-1-l].T)
            if self.DC - l > 1:
                Y = np.dot(Y.T,self.W[self.DC-1-l]).T*self.d_SD(h[self.DC-l-1])
            l+=1
        for i in range(self.DC-1):
            self.b[i]+=0.005*self.d_b[i]
            self.W[i]+=0.005*self.d_W[i]
        return np.dot(Y.T,self.W[0])
    def save(self):
        N = 0
        for i in self.W:
            np.save('W'+str(N),i)
            N+=1
        N = 0
        for i in self.b:
            np.save('b'+str(N),i)
            N+=1
    def load(self):
        for N in range(self.DC-1):
            self.W[N] = np.load('W'+str(N)+'.npy')
            self.b[N] = np.load('b'+str(N)+'.npy')
class SNN:
    def __init__(self,x,n,m, nuclear, y, nCount, deepCount):
        self.W = []
        self.Wx = []
        n1 = n
        m1 = m
        h =0
        for i in nuclear:
            self.W.append(np.random.rand(i*i,1))
            self.W[h]/=10
            h+=1
            n1-=(i-1)
            m1-=(i-1)
        x1 = np.zeros((m1*n1,1))
        self.N = NN(x1,y,nCount,deepCount)
    def GetPos(self,i,j,N,M):
        return i*M + j
    def SKA(self,x,nuc,n,m): # х - входной вектор, n - колво строк в матрице от которой получен х, m - столбцов, nuc - ядро свертки (тож. вектор)
        s = int(math.sqrt(nuc.shape[0]))
        sm = s - 1
        n1 = n - sm
        m1 = m - sm
        cnt = int(n1*m1)
        y = np.zeros((cnt,1))
        dy = np.zeros((cnt,s*s))
        if s % 2 == 0:
            centr = (s//2-1,s//2-1)
        else:
            centr = (s//2,s//2)
        for i in range(centr[0],n - (sm - centr[0])):
            for j in range(centr[1],m - (sm - centr[1])):
                sum = 0
                for v in range(-centr[0],s-centr[0]):
                    for u in range(-centr[1],s-centr[0]):
                        sum+=x[self.GetPos(i+v,j+u,n,m)][0]*nuc[self.GetPos(v+centr[0],u+centr[1],s,s)][0]
                        dy[self.GetPos(i-centr[0],j-centr[1],n1,m1)][self.GetPos(v+centr[0],u+centr[1],s,s)] = x[self.GetPos(i+v,j+u,n,m)][0]
                y[self.GetPos(i-centr[0],j-centr[1],n1,m1)][0] = sum
        return y, dy
    def feedforward(self,x,n,m):
        dylist = []
        for i in self.W:
            x, dy = self.SKA(x,i,n,m)
            dylist.append(dy)
            n = n - int(math.sqrt(i.shape[0])) + 1
            m = m - int(math.sqrt(i.shape[0])) + 1
        return self.N.feedforward(x), dylist, x, n, m
    def Rot(self,x):
        y = np.zeros(x.shape)
        n = x.shape[0]
        i = n-1
        v = 0
        while i>=0:
            y[i][0] = x[v][0]
            i-=1
            v+=1
        return y
    def UNSKA(self,x,nuc,n,m):
        s = int(math.sqrt(nuc.shape[0]))
        sm = s-1
        n1 = n + sm
        m1 = m + sm
        cnt = int(n1 * m1)
        nuc = self.Rot(nuc)
        y = np.zeros((cnt, 1))
        if s % 2 == 0:
            centr = (s // 2 - 1, s // 2 - 1)
        else:
            centr = (s // 2, s // 2)
        for i in range(0, n1):
            for j in range(0, m1):
                sum = 0
                for v in range(-centr[0], s - centr[0]):
                    for u in range(-centr[1], s - centr[0]):
                        if i + v -sm >= 0 and  i+v-sm <n and j+u-sm >=0 and j+u-sm< m:
                            sum += x[self.GetPos(i + v-sm, j + u-sm, n, m)][0] * nuc[self.GetPos(v + centr[0], u + centr[1], s, s)][0]
                y[self.GetPos(i , j , n1, m1)][0] = sum
        return y, n1, m1
    def Teach(self,x,y,n,m):
        yt, dylist, x1,n,m = self.feedforward(x,n,m)
        dg = self.N.Teach(x1,y)
        O = len(dylist) - 1
        dw = []
        for i in range(len(dylist)):
            dw.append(0)
        while O >=0:
            dw[O] = np.dot(dg,dylist[O]).T
            if O > 0:
                dg,n ,m = self.UNSKA(dg.T,self.W[O],n,m)
                dg = dg.T
            O-=1
        for i in range(len(self.W)):
            self.W[i] +=0.005*dw[i]
    def save(self):
        self.N.save()
        t = 0
        for i in self.W:
            np.save('SW'+str(t),i)
            t+=1
    def load(self):
        self.N.load()
        t = 0
        for i in range(len(self.W)):
            self.W[i] = np.load('SW'+str(t)+'.npy')
            t+=1