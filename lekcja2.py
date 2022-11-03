import matplotlib.colors
from matplotlib.image import imread
from matplotlib.pyplot import imshow
from matplotlib.image import imsave
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import math
from matplotlib.colors import hsv_to_rgb



class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4  # obraz 2d


class BaseImage:
    data: np.ndarray  # tensor przechowujacy piksele obrazu
    color_model: ColorModel  # atrybut przechowujacy biezacy model barw obrazu

    def __init__(self, path: str) -> None:
        self.data = imread(path)
        self.color_model = 0
        # inicjalizator wczytujacy obraz do atrybutu data na podstawie sciezki
        pass

    def save_img(self, path: str) -> None:
        imsave(path, self.data)
        # metoda zapisujaca obraz znajdujacy sie w atrybucie data do pliku
        pass

    def show_img(self) -> None:
        if(self.color_model==0):
            imshow(self.data)
            plt.show()
        # metoda wyswietlajaca obraz znajdujacy sie w atrybucie data
        pass

    def get_layer(self, layer_id: int) -> 'BaseImage':
        if (layer_id == 0):
            return self.data[:, :, 0]
        elif (layer_id == 1):
            return self.data[:, :, 1]
        elif (layer_id == 2):
            return self.data[:, :, 2]
        else:
            return 'Błąd, argument może być tylko 0, 1 lub 2!'
        # metoda zwracajaca warstwe o wskazanym indeksie
        pass

    def to_hsv(self) -> 'BaseImage':

        R = self.data[:, :, 0].astype('float32')
        G = self.data[:, :, 1].astype('float32')
        B = self.data[:, :, 2].astype('float32')
        M_tmp = np.maximum(R, G)
        M = np.maximum(M_tmp, B)
        m_tmp = np.minimum(R, G)
        m = np.minimum(m_tmp, B)

        V = M / 255
        S:np.ndarray = np.zeros(M.shape)
        H:np.ndarray = np.zeros(G.shape)
        for i in range(0, M.shape[0]):
            for j in range(0, M.shape[1]):
                if(M[i][j] > 0):
                    S[i][j] = 1 - m[i][j] / M[i][j]
                else:
                    S[i][j] = 0
        for i in range(0, R.shape[0]):
            for j in range(0, R.shape[1]):
                tmp_sqrt = np.sqrt(R[i][j] ** 2 + G[i][j] ** 2 + B[i][j] ** 2 - R[i][j] * G[i][j] - R[i][j] * B[i][j] - G[i][j] * B[i][j], dtype=np.float64)
                if (G[i][j] >= B[i][j]):
                    H[i][j] = np.degrees(np.arccos(((R[i][j] - G[i][j] / 2 - B[i][j] / 2) / tmp_sqrt)))
                else:
                    H[i][j] = 360 - np.degrees(np.arccos((R[i][j] - G[i][j] / 2 - B[i][j] / 2) / tmp_sqrt))
        HSV = np.stack((H/360, S, V), axis=2)
        self.color_model = 1
        self.data = HSV
        return HSV
        # metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        # metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        pass

    def to_hsi(self) -> 'BaseImage':
        R = self.data[:, :, 0].astype('float32')
        G = self.data[:, :, 1].astype('float32')
        B = self.data[:, :, 2].astype('float32')

        M_tmp = np.maximum(R, G)
        M = np.maximum(M_tmp, B)
        m_tmp = np.minimum(R, G)
        m = np.minimum(m_tmp, B)

        I = (R + G + B)/3

        S: np.ndarray = np.zeros(M.shape)
        H: np.ndarray = np.zeros(G.shape)
        for i in range(0, M.shape[0]):
            for j in range(0, M.shape[1]):
                if (M[i][j] > 0):
                    S[i][j] = 1 - m[i][j] / M[i][j]
                else:
                    S[i][j] = 0
        for i in range(0, R.shape[0]):
            for j in range(0, R.shape[1]):
                tmp_sqrt = np.sqrt(
                    R[i][j] ** 2 + G[i][j] ** 2 + B[i][j] ** 2 - R[i][j] * G[i][j] - R[i][j] * B[i][j] - G[i][j] * B[i][
                        j], dtype=np.float64)
                if (G[i][j] >= B[i][j]):
                    H[i][j] = np.degrees(np.arccos(((R[i][j] - G[i][j] / 2 - B[i][j] / 2) / tmp_sqrt)))
                else:
                    H[i][j] = 360 - np.degrees(np.arccos((R[i][j] - G[i][j] / 2 - B[i][j] / 2) / tmp_sqrt))
                    if (H[i][j] > 1 and H[i][j] <= 0):
                        print('H[' + str(i) + '][' + str(j) + '] = ' + str(H[i][j]))

        HSI = np.stack((H, S, I), axis=2).astype("uint8")
        self.data = HSI
        self.color_model = 2
        return HSI

        # metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        # metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        pass

    def to_hsl(self) -> 'BaseImage':
        R = self.data[:, :, 0].astype('float32')
        G = self.data[:, :, 1].astype('float32')
        B = self.data[:, :, 2].astype('float32')

        M_tmp = np.maximum(R, G)
        M = np.maximum(M_tmp, B)
        m_tmp = np.minimum(R, G)
        m = np.minimum(m_tmp, B)
        d = (M - m) / 255
        L = (0.5 * (M + m)) / 255

        S: np.ndarray = np.zeros(M.shape)
        H: np.ndarray = np.zeros(M.shape)

        for i in range(0, L.shape[0]):
            for j in range(0, L.shape[1]):
                if(L[i][j]>0):
                    S[i][j] = d[i][j] / (1 - np.abs(2 * L[i][j] - 1))
                else:
                    S[i][j] = 0
        for i in range(0, R.shape[0]):
            for j in range(0, R.shape[1]):
                tmp_sqrt = np.sqrt(R[i][j] ** 2 + G[i][j] ** 2 + B[i][j] ** 2 - R[i][j] * G[i][j] - R[i][j] * B[i][j] - G[i][j] * B[i][j], dtype=np.float64)
                if (G[i][j] >= B[i][j]):
                    H[i][j] = np.degrees(np.arccos(((R[i][j] - G[i][j] / 2 - B[i][j] / 2) / tmp_sqrt)))
                else:
                    H[i][j] = 360 - np.degrees(np.arccos((R[i][j] - G[i][j] / 2 - B[i][j] / 2) / tmp_sqrt))

        HSL = np.stack((H, S, L), axis=2).astype("uint8")
        self.date = HSL
        self.color_model = 3
        return HSL
        # metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        # metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        pass

    def to_rgb(self) -> 'BaseImage':
        #HSV
        if(self.color_model==1):
            H = self.data[:, :, 0].astype('float32')
            S = self.data[:, :, 1].astype('float32')
            V = self.data[:, :, 2].astype('float32')
            R: np.ndarray = np.zeros(H.shape)
            G: np.ndarray = np.zeros(H.shape)
            B: np.ndarray = np.zeros(H.shape)
            M = 255 * V
            m = M * (1 - S)
            z = (M - m) * (1 - np.abs(((H/60) % 2) - 1))
            for i in range(0, H.shape[0]):
                for j in range(0, H.shape[1]):
                    if(H[i][j]<60 and H[i][j]>=0):
                        R[i][j] = M[i][j]
                        G[i][j] = z[i][j] + m[i][j]
                        B[i][j] = m[i][j]
                    elif(H[i][j]<120 and H[i][j]>=60):
                        R[i][j] = z[i][j] + m[i][j]
                        G[i][j] = M[i][j]
                        B[i][j] = m[i][j]
                    elif(H[i][j]<180 and H[i][j]>=120):
                        R[i][j] = m[i][j]
                        G[i][j] = M[i][j]
                        B[i][j] = z[i][j] + m[i][j]
                    elif(H[i][j]<240 and H[i][j]>=180):
                        R[i][j] = m[i][j]
                        G[i][j] = M[i][j]
                        B[i][j] = z[i][j] + m[i][j]
                    elif(H[i][j]<300 and H[i][j]>=240):
                        R[i][j] = z[i][j] + m[i][j]
                        G[i][j] = m[i][j]
                        B[i][j] = M[i][j]
                    elif(H[i][j]<360 and H[i][j]>=300):
                        R[i][j] = M[i][j]
                        G[i][j] = m[i][j]
                        B[i][j] = z[i][j] + m[i][j]
        #HSI
        elif(self.color_model==2):
            H = self.data[:, :, 0].astype('float32')
            S = self.data[:, :, 1].astype('float32')
            I = self.data[:, :, 2].astype('float32')
            R: np.ndarray = np.zeros(H.shape)
            G: np.ndarray = np.zeros(H.shape)
            B: np.ndarray = np.zeros(H.shape)
            for i in range(0, H.shape[0]):
                for j in range(0, H.shape[1]):
                    if(H[i][j]==0):
                        R[i][j] = I[i][j] + 2 * I[i][j] * S[i][j]
                        G[i][j] = I[i][j] - I[i][j] * S[i][j]
                        B[i][j] = I[i][j] - I[i][j] * S[i][j]
                    elif(H[i][j]<120 and H[i][j]>0):
                        R[i][j] = I[i][j] + I[i][j] * S[i][j] * np.cos(H[i][j]) / np.cos(60 - H[i][j])
                        G[i][j] = I[i][j] + I[i][j] * S[i][j] * (1 - np.cos(H[i][j]) / np.cos(60 - H[i][j]))
                        B[i][j] = I[i][j] - I[i][j] * S[i][j]
                    elif(H[i][j]==120):
                        R[i][j] = I[i][j] - I[i][j] * S[i][j]
                        G[i][j] = I[i][j] + 2 * I[i][j] * S[i][j]
                        B[i][j] = I[i][j] - I[i][j] * S[i][j]
                    elif(H[i][j]<240 and H[i][j]>120):
                        R[i][j] = I[i][j] - I[i][j] * S[i][j]
                        G[i][j] = I[i][j] + I[i][j] * S[i][j] * np.cos(H[i][j] - 120) / np.cos(180 - H[i][j])
                        B[i][j] = I[i][j] + I[i][j] * S[i][j] * (1 - np.cos(H[i][j] - 120) / np.cos(180 - H[i][j]))
                    elif(H[i][j]==240):
                        R[i][j] = I[i][j] - I[i][j] * S[i][j]
                        G[i][j] = I[i][j] - I[i][j] * S[i][j]
                        B[i][j] = I[i][j] + 2 * I[i][j] * S[i][j]
                    elif(H[i][j]<360 and H[i][j]>240):
                        R[i][j] = I[i][j] + I[i][j] * S[i][j] * (1 - np.cos(H[i][j] - 240) / np.cos(300 - H[i][j]))
                        G[i][j] = I[i][j] - I[i][j] * S[i][j]
                        B[i][j] = I[i][j] + I[i][j] * S[i][j] * np.cos(H[i][j] - 240) / np.cos(300 - H[i][j])
        #HSL
        elif(self.color_model==3):
            H = self.data[:, :, 0].astype('float32')
            S = self.data[:, :, 1].astype('float32')
            L = self.data[:, :, 2].astype('float32')
            R: np.ndarray = np.zeros(H.shape)
            G: np.ndarray = np.zeros(H.shape)
            B: np.ndarray = np.zeros(H.shape)
            d = S * (1 - np.abs(2 * L - 1))
            m = 255 * (L - 0.5 * d)
            x = d * (1 - np.abs(((H / 60) % 2) - 1))
            for i in range(0, H.shape[0]):
                for j in range(0, H.shape[1]):
                    if(H[i][j]<60 and H[i][j]>=0):
                        R[i][j] = 255 * d[i][j] + m[i][j]
                        G[i][j] = 255 * x[i][j] + m[i][j]
                        B[i][j] = m[i][j]
                    if(H[i][j]<120 and H[i][j]>=60):
                        R[i][j] = 255 * x[i][j] + m[i][j]
                        G[i][j] = 255 * d[i][j] + m[i][j]
                        B[i][j] = m[i][j]
                    if(H[i][j]<180 and H[i][j]>=120):
                        R[i][j] = m[i][j]
                        G[i][j] = 255 * d[i][j] + m[i][j]
                        B[i][j] = 255 * x[i][j] + m[i][j]
                    if(H[i][j]<240 and H[i][j]>=180):
                        R[i][j] = m[i][j]
                        G[i][j] = 255 * x[i][j] + m[i][j]
                        B[i][j] = 255 * d[i][j] + m[i][j]
                    if(H[i][j]<300 and H[i][j]>=240):
                        R[i][j] = 255 * x[i][j] + m[i][j]
                        G[i][j] = m[i][j]
                        B[i][j] = 255 * d[i][j] + m[i][j]
                    if(H[i][j]<360 and H[i][j]>=300):
                        R[i][j] = 255 * d[i][j] + m[i][j]
                        G[i][j] = m[i][j]
                        B[i][j] = 255 * x[i][j] + m[i][j]
        RGB = np.stack((R, G, B), axis=2).astype("uint8")
        self.color_model = 0
        self.data = RGB
        return RGB
        # metoda dokonujaca konwersji obrazu w atrybucie data do modelu rgb
        # metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        pass

x = BaseImage('lena.jpg')
x.to_hsl()
print(x.color_model)
x.to_rgb()
print(x.color_model)
x.show_img()
        
