import numpy as np

def numpy_rle(arr):
    if len(arr) == 0:
        return [], []
    arr = np.asarray(arr)  
    # 隣り合う要素が異なる場所（変化点）を探す
    y = arr[1:] != arr[:-1]
    # 変化点のインデックスを取得し、両端を考慮して調整
    i = np.append(np.where(y), len(arr) - 1)
    
    # 各ランの長さを計算（現在の変化点 - 一つ前の変化点）
    z = np.diff(np.append(-1, i))
    
    # (値, 個数) の形式で返すなら zip する
    # arr[i] は各ランの最後の値（つまりそのランの値）
    return list(zip(arr[i], z))

def stretch_idx(x,target_length,start_idx):
    rl = numpy_rle(x)
    stretchable = 0
    unstretchable = 0
    for vuv,l in rl:
        if vuv == 2:
            unstretchable += l
        else:
            stretchable += l
    if len(x) == target_length:
        # no need to stretch/shrink
        return np.array([i for i in range(start_idx,start_idx+target_length)])
    factor = (target_length-unstretchable)/stretchable
    stretched_rl = np.zeros(target_length,dtype=np.int16)
    idx = start_idx
    pos = 0
    for vuv,l in rl:
        if vuv == 2:
            for i in range(l):
                if pos >= len(stretched_rl):
                    break
                stretched_rl[pos] = idx
                idx += 1
                pos += 1
        else:
            sl = int(l*factor+0.5)
            if sl == 0:
                sl = 1
            fl = pos+sl
            if fl > target_length:
                fl = target_length
            t = 0
            while pos < fl:
                stretched_rl[pos] = idx+int(t/sl*l)
                t += 1
                pos += 1
            idx = idx+l
    return stretched_rl
                




#data = [1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1]
#result = numpy_rle(np.array(data))
#print(result)