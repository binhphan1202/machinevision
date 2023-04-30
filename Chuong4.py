import cv2
import numpy as np

L = 256

def Spectrum(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)

    # Bước 1 và 2:
    # Tạo ảnh fp có kích thước PxQ
    # Và thêm số 0 vào phần mở rộng
    fp = np.zeros((P,Q), np.float32)
    fp[:M, :N] = imgin
    fp = fp/(L-1)

    # Bước 3: Nhân với (-1)^(x+y)
    # để dời biến đổi Fourier vào tâm
    for x in range(0, M):
        for y in range(0, N):
            if(x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]

    # Bước 4: Biến đổi DFT
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Bước 5: Tính phổ
    R = F[:,:,0]
    I = F[:,:,1]
    S = np.sqrt(R**2 + I**2)

    S = np.clip(S, 0, L-1)
    S = S.astype(np.uint8)
    return S

def FrequencyFilter(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)

    # Bước 1 và 2:
    # Tạo ảnh fp có kích thước PxQ
    # Và thêm số 0 vào phần mở rộng
    fp = np.zeros((P,Q), np.float32)
    fp[:M, :N] = imgin

    # Bước 3: Nhân với (-1)^(x+y)
    # để dời biến đổi Fourier vào tâm
    for x in range(0, M):
        for y in range(0, N):
            if(x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]

    # Bước 4: Biến đổi DFT
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Bước 5: Tạo bộ lọc H thực
    H = np.zeros((P,Q), np.float32)
    D0 = 60
    n = 2
    for u in range(0, P):
        for v in range(0, Q):
            Duv = np.sqrt((u-P//2)**2 + (v-Q//2)**2)
            if Duv > 0:
                H[u,v] = 1.0/(1.0 + np.power(D0/Duv, 2*n))
    
    # Bước 6: G = F*H
    G = F.copy()
    for u in range(0, P):
        for v in range(0, Q):
            G[u,v,0] = F[u,v,0]*H[u,v]
            G[u,v,1] = F[u,v,1]*H[u,v]

    # Bước 7: Biến đổi IDFT
    gp = cv2.idft(G, flags=cv2.DFT_SCALE)
    # Lấy phần thực
    gp = gp[:,:,0]
    for x in range(0, P):
        for y in range(0, Q):
            if(x+y) % 2 == 1:
                gp[x,y] = -gp[x,y]
    # Bỏ phần mở rộng
    g = gp[:M, :N]

    g = np.clip(g, 0, L-1)
    g = g.astype(np.uint8)
    return g

def DrawNotchRejectFilter():
    P = 250
    Q = 180
    u1, v1 = 44, 58
    u2, v2 = 40, 119
    u3, v3 = 86, 59
    u4, v4 = 82, 119

    D0 = 10
    n = 2
    H = np.ones((P, Q), np.float32)

    for u in range(0, P):
        for v in range(0, Q):
            h = 1.0
            # Bộ lọc u1, v1
            Duv = np.sqrt((u-u1)**2 + (v-v1)**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u1))**2 + (v-(Q-v1))**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0

            # Bộ lọc u2, v2
            Duv = np.sqrt((u-u2)**2 + (v-v2)**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u2))**2 + (v-(Q-v2))**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0

            # Bộ lọc u3, v3
            Duv = np.sqrt((u-u3)**2 + (v-v3)**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u3))**2 + (v-(Q-v3))**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0

            # Bộ lọc u4, v4
            Duv = np.sqrt((u-u4)**2 + (v-v4)**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u4))**2 + (v-(Q-v4))**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0

            H[u,v] = h
    H = H*(L-1)
    H = H.astype(np.uint8)
    return H

def CreateNotchRejectFilter():
    P = 250
    Q = 180
    u1, v1 = 44, 58
    u2, v2 = 40, 119
    u3, v3 = 86, 59
    u4, v4 = 82, 119

    D0 = 10
    n = 2
    H = np.ones((P, Q), np.float32)

    for u in range(0, P):
        for v in range(0, Q):
            h = 1.0
            # Bộ lọc u1, v1
            Duv = np.sqrt((u-u1)**2 + (v-v1)**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u1))**2 + (v-(Q-v1))**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0

            # Bộ lọc u2, v2
            Duv = np.sqrt((u-u2)**2 + (v-v2)**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u2))**2 + (v-(Q-v2))**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0

            # Bộ lọc u3, v3
            Duv = np.sqrt((u-u3)**2 + (v-v3)**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u3))**2 + (v-(Q-v3))**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0

            # Bộ lọc u4, v4
            Duv = np.sqrt((u-u4)**2 + (v-v4)**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u4))**2 + (v-(Q-v4))**2)
            if Duv > 0:
                h  = h*(1.0/(1.0 +  np.power(D0/Duv, 2*n)))
            else:
                h = h*0.0

            H[u,v] = h
    return H

def NotchRejectFilter(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)

    # Bước 1 và 2:
    # Tạo ảnh fp có kích thước PxQ
    # Và thêm số 0 vào phần mở rộng
    fp = np.zeros((P,Q), np.float32)
    fp[:M, :N] = imgin

    # Bước 3: Nhân với (-1)^(x+y)
    # để dời biến đổi Fourier vào tâm
    for x in range(0, M):
        for y in range(0, N):
            if(x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]

    # Bước 4: Biến đổi DFT
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Bước 5: Tạo bộ lọc H thực
    H = CreateNotchRejectFilter()
    
    # Bước 6: G = F*H
    G = F.copy()
    for u in range(0, P):
        for v in range(0, Q):
            G[u,v,0] = F[u,v,0]*H[u,v]
            G[u,v,1] = F[u,v,1]*H[u,v]

    # Bước 7: Biến đổi IDFT
    gp = cv2.idft(G, flags=cv2.DFT_SCALE)
    # Lấy phần thực
    gp = gp[:,:,0]
    for x in range(0, P):
        for y in range(0, Q):
            if(x+y) % 2 == 1:
                gp[x,y] = -gp[x,y]
    # Bỏ phần mở rộng
    g = gp[:M, :N]

    g = np.clip(g, 0, L-1)
    g = g.astype(np.uint8)
    return g
