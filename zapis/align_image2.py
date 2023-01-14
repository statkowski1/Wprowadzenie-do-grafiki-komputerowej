def align_image(self, tail_elimination: bool = True) -> 'BaseImage':
    R = self.data[:, :, 0].astype('float32')
    G = self.data[:, :, 1].astype('float32')
    B = self.data[:, :, 2].astype('float32')
    I = (R + G + B) / 3
    wynikP: np.ndarray = np.zeros(I.shape)
    if tail_elimination == False:
        m = I.min()
        M = I.max()
    elif tail_elimination == True:
        cumulated:np.ndarray = np.zeros(256)
        m = 0
        M = 0
        for i in range(0, I.shape[0]):
            for j in range(0, I.shape[1]):
                cumulated[I[i][j].astype('int64')] += 1
        for i in range(1, 256):
            cumulated[i] = cumulated[i] + cumulated[i - 1]
        x = 0.05 * cumulated[255]
        X = 0.95 * cumulated[255]
        for i in range(0, cumulated.shape[0]):
            if(cumulated[i]<=x):
                m = i
            elif(cumulated[i]>=X):
                M = i
    for i in range(0, I.shape[0]):
        for j in range(0, I.shape[1]):
            wynikP[i][j] = (I[i][j] - m) * 255 / (M - m)
    wynik: np.ndarray = np.zeros(256)
    for i in range(0, wynikP.shape[0]):
        for j in range(0, wynikP.shape[1]):
            wynik[wynikP[i][j].astype('int64')] += 1
    plt.plot(range(0, 256), wynik)
    plt.show()
    #metoda zwracajaca poprawiony obraz metoda wyrownywania histogramow
    pass
