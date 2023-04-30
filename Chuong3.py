import cv2
import numpy as np

L = 256

# Làm âm ảnh
def Negative(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = L-1-r
            imgout[x,y] = s
    return imgout

# Làm sáng ảnh
def Logarit(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    c = (L-1)/np.log(L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r==0:
                r = 1
            s = c*np.log(1+r)
            imgout[x,y] = np.uint8(s)
    return imgout

# Làm tối ảnh
def Power(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    gamma = 5.0
    c = np.power(L-1, 1-gamma)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = c*np.power(r, gamma)
            imgout[x, y]= np.uint8(s)
    return imgout

# Đảo độ tương phản
def PiecewiseLinear(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    rmin, rmax, vi_tri_rmin, vi_tri_rmax = cv2.minMaxLoc(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L-1
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            # Đoạn I
            if r < r1:
                s = (s1/r1)*r
            # Đoạn II
            elif r < r2:
                s = ((s2-s1)/(r2-r1))*(r-r1) + s1
            # Đoạn III
            else:
                s = ((L-1-s2)/(L-1-r2))*(r-r2) + s2
            imgout[x, y] = np.uint8(s)
    return imgout   

# Tăng độ sáng
def IRange(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    A = 150
    B = 230
    I = 200
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if A <= r <= B:
                s = I
            else:
                s = r
            imgout[x, y] = s
    return imgout

# Bit
def BitPlane(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    
    # Bước 1: Phân tích thành 8 ảnh mặt phẳng bit
    plane1 = np.zeros((M, N), np.uint8)
    plane2 = np.zeros((M, N), np.uint8)
    plane3 = np.zeros((M, N), np.uint8)
    plane4 = np.zeros((M, N), np.uint8)
    plane5 = np.zeros((M, N), np.uint8)
    plane6 = np.zeros((M, N), np.uint8)
    plane7 = np.zeros((M, N), np.uint8)
    plane8 = np.zeros((M, N), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            # Mặt phẳng bit 1
            mask = 0x01
            s = r & mask
            if s > 0:
                s = 255
            else:
                s = 0
            plane1[x, y] = s
            # Mặt phẳng bit 2
            mask = 0x02
            s = r & mask
            if s > 0:
                s = 255
            else:
                s = 0
            plane2[x, y] = s
            # Mặt phẳng bit 3
            mask = 0x04
            s = r & mask
            if s > 0:
                s = 255
            else:
                s = 0
            plane3[x, y] = s
            # Mặt phẳng bit 4
            mask = 0x08
            s = r & mask
            if s > 0:
                s = 255
            else:
                s = 0
            plane4[x, y] = s
            # Mặt phẳng bit 5
            mask = 0x10
            s = r & mask
            if s > 0:
                s = 255
            else:
                s = 0
            plane5[x, y] = s
            # Mặt phẳng bit 6
            mask = 0x20
            s = r & mask
            if s > 0:
                s = 255
            else:
                s = 0
            plane6[x, y] = s
            # Mặt phẳng bit 7
            mask = 0x40
            s = r & mask
            if s > 0:
                s = 255
            else:
                s = 0
            plane7[x, y] = s
            # Mặt phẳng bit 8
            mask = 0x80
            s = r & mask
            if s > 0:
                s = 255
            else:
                s = 0
            plane8[x, y] = s

    # Bước 2: Tạo lặp lại ảnh dùng mặt phẳng 8 7 6 5
    for x in range(0, M):
        for y in range(0, N):
            if plane8[x, y] == 255:
                a = 0x80
            else:
                a = 0x00
            if plane7[x, y] == 255:
                b = 0x40
            else:
                b = 0x00
            if plane6[x, y] == 255:
                c = 0x20
            else:
                c = 0x00
            if plane5[x, y] == 255:
                d = 0x10
            else:
                d = 0x00
            s = a | b | c | d
            imgout[x, y] = s
    return imgout

def Histogram(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, L), np.uint8)
    h=np.zeros(L,np.int32)
    for x in range (0,M):
        for y in range (0,N):
            r=imgin[x,y]
            h[r]=h[r]+1

    p=h/(M*N)
    scale=2000
    for r in range (0,L):
        cv2.line(imgout,(r,M-1),(r,M-1- int(scale*p[r])),(0,0,0))
    return imgout

def HistEqual(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    h=np.zeros(L,np.int32)
    for x in range (0,M):
        for y in range (0,N):
            r=imgin[x,y]
            h[r]=h[r]+1

    p=h/(M*N)
    s=np.zeros(L,np.float32)
    for k in range (0,L):
        for j in range (0,k+1):
            s[k]=s[k]+p[j]
            
    for x in range (0,M):
        for y in range (0,N):
            r=imgin[x,y]
            imgout[x,y] = np.uint8((L-1)*s[r])
    return imgout

def HistEqualcolor(imgin):
    M,N,C = imgin.shape
    imgout = np.zeros((M,N,C), np.uint8)

    R=imgin[:,:,2]
    G=imgin[:,:,1]
    B=imgin[:,:,0]

    R=cv2.equalizeHist(R)
    G=cv2.equalizeHist(G)
    B=cv2.equalizeHist(B)

    imgout[:,:,2]=R
    imgout[:,:,1]=G
    imgin[:,:,0]=B
    return imgout

def LocalHist(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m=3
    n=3
    a=m//2
    b=n//2
    w=np.zeros((m,n),np.uint8)
    for x in range (a,M-a):
        for y in range(b,N-b):
            for s in range(-a,a+1):
                for t in range(-b,b+1):
                    w[s+a,t+b]=imgin[x+s,y+t]
            w=cv2.equalizeHist(w)
            imgout[x,y]=w[a,b]
    return imgout

def HistStat(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 3
    n = 3
    a = m // 2
    b = n // 2
    w = np.zeros((m,n), np.uint8)

    mG, sigmaG = cv2.meanStdDev(imgin)
    k0 = 0.0 
    k1 = 0.1
    k2 = 0.0
    k3 = 0.1
    c = 22.8

    for x in range(a, M-a):
        for y in range(b, N-b):
            for s in range(-a,a+1):
                for t in range(-b,b+1):
                    w[s+a,t+b] = imgin[x+s,y+t]
            msxy, sigmasxy = cv2.meanStdDev(w)
            r = imgin[x,y]
            if (k0*mG <= msxy <= k1*mG) and (k2*mG <= sigmasxy <= k3*sigmaG):
                imgout[x,y] = np.uint8(c*r)
            else:
                imgout[x,y] = r

    return imgout

def MyBoxFilter(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, L), np.uint8)
    m, n = 21, 21
    w = np.ones((m,n), np.float64)
    w = w/(m*n)

    a = m//2
    b = n//2
    for x in range(0, M):
        for y in range(0, N):
            r = 0.0
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    r = r + w[s+a, t+b]*imgin[(x+s)%M, (y+t)%N]
            imgout[x,y] = np.uint8(r)
    return imgout

def BoxFilter(imgin):
    m, n = 21, 21
    w = np.ones((m,n), np.float64)
    w = w/(m*n)
    imgout = cv2.filter2D(imgin, cv2.CV_8UC1, w)
    return imgout

def TinhBoLocGauss(m, n, sigma):
    w = np.zeros((m,n), np.float64)
    a = m // 2
    b = n // 2
    for s in range(-a, a+1):
        for t in range(-b, b+1):
            w[s+a, t+b] = np.exp(-(s**2+t**2)/(2*sigma**2))
    sum = np.sum(w)
    w=w/sum
    return w

def GaussFilter(imgin):
    m, n = 3, 3
    sigma = 7.0
    w = TinhBoLocGauss(m, n, sigma)
    imgout = cv2.filter2D(imgin, cv2.CV_8UC1, w)
    return imgout

def Smooth(imgin):
    M, N = imgin.shape
    temp = cv2.blur(imgin, (15,15))
    retval, imgout = cv2.threshold(temp, 64, L-1, cv2.THRESH_BINARY)
    return imgout
