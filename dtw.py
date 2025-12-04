# DTW between one-dimensional signals
import numpy as np

def dtw(sp,score,uv_val=-1):
    # sp: list or 1-d array
    # score: list or 1-d array
    # if sp[i] == uv_val, this frame should not be shrunk/extended
    N = len(sp)
    M = len(score)
    #print("N=",N," M=",M)
    g = np.zeros((N,M))
    b = np.zeros((N,M),dtype=np.int16)
    # b: backpointer
    #   0: (i-1,j-1)  1: (i-2,j-1)  2: (i,j-1)
    for i in range(N):
        for j in range(M):
            v = abs(sp[i]-score[j])
            if j == 0:
                if i == 0:
                    g[i,j] = v
                else:
                    g[i,j] = 1e10
                continue
            if j == 1:
                if i == 0:
                    g[i,j] = v+g[i,j-1]
                    b[i,j] = 2
                if i == 1:
                    g0 = v+g[i-1,j-1]
                    g2 = v+g[i,j-1]
                    if g0 <= g2:
                        g[i,j] = g0
                        b[i,j] = 0
                    else:
                        g[i,j] = g2
                        b[i,j] = 2
                else:
                    g[i,j] = 1e10
                continue
            if i == 0: # j >= 2 hereafter
                g[i,j] = v+g[i,j-1]
                b[i,j] = 2
                continue
            if i == 1:
                g0 = v+g[i-1,j-1]
                g2 = v+g[i,j-1]
                if g0 <= g2:
                    g[i,j] = g0
                    b[i,j] = 0
                else:
                    g[i,j] = g2
                    b[i,j] = 2
                g[i,j] = v+g[i-1,j]
                b[i,j] = 1
            else: # i >= 2, j >= 2
                g0 = v+g[i-1,j-1]
                g1 = v+g[i-1,j]
                g2 = v+g[i,j-1]
                if sp[i] == uv_val:
                    if g0 <= g1:
                        g[i,j] = g0
                    else:
                        g[i,j] = g1
                        b[i,j] = 1      
                else:
                    if g0 <= g1:
                        if g0 <= g2:
                            g[i,j] = g0
                            b[i,j] = 0
                        else:
                            g[i,j] = g2
                            b[i,j] = 2
                    elif g1 <= g2:
                        g[i,j] = g1
                        b[i,j] = 1
                    else:
                        g[i,j] = g2
                        b[i,j] = 2
    # backtrace
    opt = []
    i = N-1
    j = M-1
    while i >= 0 and j >= 0:
        #print((i,j),sp[i],score[j])
        opt.append((i,j))
        if b[i,j] == 0:
            i -= 1
            j -= 1
        elif b[i,j] == 1:
            i -= 2
            j -= 1
        else: # 2
            j -= 1
    opt.reverse()
    return opt

if __name__ == "__main__": 
    sp = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 
          2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 0, 
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2,
          2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1,
          0, 0, 0, 0, 0, 0]
    notes = [0., 10.0, 10.0, 10.0, 10.0, 8.4, 2.0, 10.0, 10.0, 10.0, 10.0, 8.4, 2.0, 10.0, 10.0,
           10.0, 10.0, 8.4, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0,
           8.4, 2.0, 10.0, 10.0, 10.0, 10.0, 8.4, 2.0, 10.0, 10.0, 10.0, 10.0, 8.4, 2.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 8.4, 2.0, 10.0, 10.0,
           10.0, 10.0, 8.4, 2.0, 10.0, 10.0, 10.0, 10.0, 8.4, 2.0, 10.0, 10.0, 10.0, 10.0,
           8.4, 2.0, 10.0, 10.0, 10.0, 10.0, 8.4, 2.0, 10.0, 10.0, 10.0, 10.0, 8.4, 2.0,
           10.0, 10.0, 10.0, 10.0, 8.4, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0. ]
    mapper = {0:0, 1:10, 2:2}
    sp = [mapper[x] for x in sp]
    print(dtw(sp,notes,uv_val=2))


