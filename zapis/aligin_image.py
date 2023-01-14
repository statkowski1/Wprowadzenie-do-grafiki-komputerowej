def align_image(self, tail_elimination: bool = True) -> 'BaseImage':
    R = self.data[:, :, 0].astype('float32')
    G = self.data[:, :, 1].astype('float32')
    B = self.data[:, :, 2].astype('float32')
    I = (R + G + B) / 3
    m = I.min()
    M = I.max()
    wynikP: np.ndarray = np.zeros(256)
    for i in range(0, I.shape[0]):
        for j in range(0, I.shape[1]):
            wynikP[I[i][j].astype('int64')] += 1
    if tail_elimination == False:
        wynik:np.ndarray = np.zeros(256)
        for i in range(0, wynikP.shape[0]):
            wynik[i] = (wynikP[i] - m) * 255 / (M - m)
    elif tail_elimination == True:
        cumulated:np.ndarray = np.zeros(256)
        for i in range(0, I.shape[0]):
            for j in range(0, I.shape[1]):
                cumulated[I[i][j].astype('int64')] += 1
        for i in range(1, 256):
            cumulated[i] = cumulated[i] + cumulated[i - 1]
        x = 0.05 * cumulated[255]
        X = 0.95 * cumulated[255]
        wynik = wynikP
        for i in range(0, cumulated.shape[0]):
            if(cumulated[i]<=x):
                wynik[i] = m
            elif(cumulated[i]>=X):
                wynik[i] = M
    plt.plot(range(0, 256), wynik)
    plt.show()
    #metoda zwracajaca poprawiony obraz metoda wyrownywania histogramow
    pass
