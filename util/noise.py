import numpy as np
import random
import matplotlib.pyplot as plt
def filter_shift(T_array,shift):
    batch_size, len_x = T_array.shape
    Ans = T_array.copy()
    for i in range(batch_size):
        ans = np.zeros(len_x+shift)
        ans[shift:shift+len_x] = T_array[i,:]
        Ans[i,:] = ans[:len_x]
    return Ans
def add_noise(T,type,para):
    T_array = T.copy()
    if type == 'gauss':
        batch_size, len_x = T_array.shape
        assert len(para) ==2
        sigma = para[1]
        mu = para[0]
        for i in range(batch_size):
            T_array[i] += np.random.normal(mu, sigma,len_x)

    elif type == 'salt':
        SNR = para[0]
        h ,w ,c = T_array.shape
        mask = np.random.choice((0, 1, 2), size=(h, w,1), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
        mask = np.repeat(mask, c, axis=2)  # 按channel 复制到 与img具有相同的shape
        T_array[mask == 1] = 255  # 盐噪声
        T_array[mask == 2] = 0  # 椒噪声

    elif type == 'white':
        SNR = para[0]
        batch_size, len_x = T_array.shape
        Ps = np.sum(np.power(T_array, 2)) / len_x
        Pn = Ps / (np.power(10, SNR / 10))
        noise = np.random.randn(len_x) * np.sqrt(Pn)
        T_array += noise

    elif type == 'poisson':
        batch_size, len_x = T_array.shape
        lamb= para[0]
        for i in range(batch_size):
            T_array[i] += np.random.poisson(lamb,len_x)

    else:
        raise AssertionError
    return T_array

if __name__ == '__main__':
    T_array = np.random.normal(0,1,[1,90])
    plt.plot(T_array.T)
    # plt.show()
    T_noise = filter_shift(T_array,10)
    plt.plot(T_noise.T)
    plt.show()
