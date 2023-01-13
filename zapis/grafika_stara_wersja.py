import matplotlib.colors
from matplotlib.image import imread
from matplotlib.pyplot import imshow
from matplotlib.image import imsave
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import math
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional


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
                        j])
                if (G[i][j] >= B[i][j]):
                    H[i][j] = np.degrees(np.arccos(((R[i][j] - G[i][j] / 2 - B[i][j] / 2) / tmp_sqrt)))
                else:
                    H[i][j] = 360 - np.degrees(np.arccos((R[i][j] - G[i][j] / 2 - B[i][j] / 2) / tmp_sqrt))

        HSI = np.stack((H, S, I), axis=2)
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
                    S[i][j] = d[i][j] / (1 - abs(2 * L[i][j] - 1))
                else:
                    S[i][j] = 0
        for i in range(0, R.shape[0]):
            for j in range(0, R.shape[1]):
                tmp_sqrt = np.sqrt(R[i][j] ** 2 + G[i][j] ** 2 + B[i][j] ** 2 - R[i][j] * G[i][j] - R[i][j] * B[i][j] - G[i][j] * B[i][j], dtype=np.float64)
                if (G[i][j] >= B[i][j]):
                    H[i][j] = np.degrees(np.arccos(((R[i][j] - G[i][j] / 2 - B[i][j] / 2) / tmp_sqrt)))
                else:
                    H[i][j] = 360 - np.degrees(np.arccos((R[i][j] - G[i][j] / 2 - B[i][j] / 2) / tmp_sqrt))
        HSL = np.stack((H, S, L), axis=2)
        self.data = HSL
        self.color_model = 3
        return HSL
        # metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        # metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        pass

    def wiedzma(self) -> 'BaseImage':
        R = self.data[:, :, 0].astype('float32')
        GB: np.ndarray = np.zeros(R.shape)
        self.data = R.astype('uint8')
        imshow(self.data, cmap="hot")
        plt.show()
        return np.stack((R, GB, GB), axis=2).astype('uint8')

    def picture_plus1(self) -> 'BaseImage':
        X = self.data
        X = X + 1
        imsave('lenaplus1.jpg', X)

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
                        G[i][j] = z[i][j] + m[i][j]
                        B[i][j] = M[i][j]
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

class GrayScaleTransform(BaseImage):
    def __init__(self, path: str) -> None:
        self.data = imread(path)
        pass

    def to_gray(self) -> BaseImage:
        R = self.data[:, :, 0].astype('float32')
        G = self.data[:, :, 1].astype('float32')
        B = self.data[:, :, 2].astype('float32')
        I = (R + G + B) / 3
        GRAY = np.stack((I, I, I), axis=2).astype('uint8')
        self.color_model = 4
        self.data = GRAY
        return GRAY
        #metoda zwracajaca obraz w skali szarosci jako obiekt klasy BaseImage
        pass
    def to_gray2(self) -> BaseImage:
        R = self.data[:, :, 0].astype('float32')
        G = self.data[:, :, 1].astype('float32')
        B = self.data[:, :, 2].astype('float32')
        I = (R + G + B) / 3
        GRAY = np.stack((R*0.299, G*0.587, B*0.114), axis=2).astype('uint8')
        self.color_model = 4
        self.data = GRAY
        return GRAY
    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> BaseImage:
        R = self.data[:, :, 0].astype('float32')
        G = self.data[:, :, 1].astype('float32')
        B = self.data[:, :, 2].astype('float32')
        I = (R + G + B) / 3
        if w is not None:
            L0 = I + 2 * w
            L1 = I + w
            L2 = I
        else:
            L0 = I * alpha_beta[0]
            L1 = I
            L2 = I * alpha_beta[1]
        for i in range(0, L0.shape[0]):
            for j in range(0, L0.shape[1]):
                if(L0[i][j]>255):
                    L0[i][j] = 255
                elif(L0[i][j]<0):
                    L0[i][j] = 0
                if (L1[i][j] > 255):
                    L1[i][j] = 255
                elif (L1[i][j] < 0):
                    L1[i][j] = 0
                if(L2[i][j]>255):
                    L2[i][j] = 255
                elif(L2[i][j]<0):
                    L2[i][j] = 0
        SEPIA = np.stack((L0, L1, L2), axis=2).astype('uint8')
        self.data = SEPIA
        return  SEPIA
        #metoda zwracajaca obraz w sepii jako obiekt klasy BaseImage
        #sepia tworzona metoda 1 w przypadku przekazania argumentu alpha_beta
        #lub metoda 2 w przypadku przekazania argumentu w
        pass

class Image(GrayScaleTransform):
    def __init__(self, path: str):
        super().__init__(path)
    pass

class Histogram:
    #klasa reprezentujaca histogram danego obrazu
    values: np.ndarray  # atrybut przechowujacy wartosci histogramu danego obrazu

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        pass

    def plot(self) -> None:
        R = self.values[:, :, 0].ravel()
        G = self.values[:, :, 1].ravel()
        B = self.values[:, :, 2].ravel()
        f = plt.figure(figsize=(10, 3))
        ax1 = f.add_subplot(131)
        ax2 = f.add_subplot(132)
        ax3 = f.add_subplot(133)
        ax1.set_ylim([-200, 3800])
        ax2.set_ylim([-200, 2600])
        ax3.set_ylim([-200, 4500])
        f.tight_layout(pad=0.5)
        ax1.hist(B, bins=256, range=[0, 256], color='red', histtype='step')
        ax2.hist(G, bins=256, range=[0, 256], color='green', histtype='step')
        ax3.hist(R, bins=256, range=[0, 256], color='blue', histtype='step')
        plt.show()
        #metoda wyswietlajaca histogram na podstawie atrybutu values
        pass

class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1
class ImageComparison(BaseImage):
    def __init__(self, path:str):
        super().__init__(path)
    #Klasa reprezentujaca obraz, jego histogram oraz metody porównania
    def histogram(self) -> Histogram:
        X = self.data
        if(self.color_model != 4):
            R = X[:, :, 0].ravel()
            G = X[:, :, 1].ravel()
            B = X[:, :, 2].ravel()
            print('G(shape) = '+str(len(G)))
            W1:np.ndarray = np.zeros(256)
            W2: np.ndarray = np.zeros(256)
            W3: np.ndarray = np.zeros(256)
            for i in range(0, R.shape[0]):
                W1[R[i].astype('uint8')] += 1
            for i in range(0, G.shape[0]):
                W2[G[i].astype('uint8')] += 1
            for i in range(0, B.shape[0]):
                W3[B[i].astype('uint8')] += 1
            W = np.stack((W1, W2, W3), axis=1)
            f = plt.figure(figsize=(10, 3))
            ax1 = f.add_subplot(131)
            ax2 = f.add_subplot(132)
            ax3 = f.add_subplot(133)
            f.tight_layout()
            ax1.plot(W[:, 2], color='red')
            ax2.plot(W[:, 1], color='green')
            ax3.plot(W[:, 0], color='blue')
            plt.show()
            return W
        else:
            GRAY = X.ravel()
            W:np.ndarray = np.zeros(256)
            for i in range(0, GRAY.shape):
                W[GRAY[i].astype('uint8')] += 1
            return W
        #metoda zwracajaca obiekt zawierajacy histogram biezacego obrazu (1- lub wielowarstwowy)
        pass

    def compare_to(self, other: Image, method: ImageDiffMethod) -> float:
        X1 = self.data
        X1A = X1[:, :, 0]
        X1B = X1[:, :, 1]
        X1C = X1[:, :, 2]
        I1 = ((X1A + X1B + X1C) / 3).ravel()
        X2 = other.data
        X2A = X2[:, :, 0]
        X2B = X2[:, :, 1]
        X2C = X2[:, :, 2]
        I2 = ((X2A + X2B + X2C) / 3).ravel()
        X1:np.ndarray = np.zeros(256)
        X2: np.ndarray = np.zeros(256)
        for i in range(0, I1.shape[0]):
            X1[I1[i].astype('int64')] += 1
        for i in range(0, I2.shape[0]):
            X2[I2[i].astype('int64')] += 1
        #MSE
        if(method == 0):
            MSE:float = 0
            for i in range(0, X1.shape[0]):
                MSE += (X1[i] - X2[i]) * (X1[i] - X2[i])
            MSE = MSE / 256
            return MSE
        #RMSE
        elif(method == 1):
            RMSE:float = 0
            for i in range(0, X1.shape[0]):
                RMSE += (X1[i] - X2[i]) * (X1[i] - X2[i])
            RMSE = np.sqrt(RMSE / 256)
            return RMSE
        #metoda zwracajaca mse lub rmse dla dwoch obrazow
        pass

class Image(GrayScaleTransform, ImageComparison):
    def __init__(self, path: str):
        super().__init__(path)
    pass

class Histogram(Histogram):
    #kontunuacja implementacji klasy
    def to_cumulated(self) -> 'Histogram':
        wynik:np.ndarray = np.zeros((3, 256))
        for i in range(0, self.values.shape[0]):
            for j in range(0, self.values.shape[1]):
                for k in range(0, self.values.shape[2]):
                    wynik[k][self.values[i][j][k].astype('int64')] += 1
        wynik = wynik.astype('int64')
        for i in range(0, wynik.shape[0]):
            for j in range(1, wynik.shape[1]):
                wynik[i][j] += wynik[i][j-1]
        return wynik
        #metoda zwracajaca histogram skumulowany na podstawie stanu wewnetrznego obiektu
        pass
class ImageAligning(BaseImage):
    #klasa odpowiadająca za wyrównywanie hostogramu
    data: np.ndarray
    def __init__(self, path:str) -> None:
        self.data = imread(path)
        #inicjalizator ...
        pass
    def align_image(self, tail_elimination: bool = True) -> 'BaseImage':
        if tail_elimination == False:
            m:np.ndarray = np.zeros(3)
            m[0] = self.data[:, :, 0].ravel().min()
            m[1] = self.data[:, :, 1].ravel().min()
            m[2] = self.data[:, :, 2].ravel().min()
            M:np.ndarray = np.zeros(3)
            M[0] = self.data[:, :, 0].ravel().max()
            M[1] = self.data[:, :, 1].ravel().max()
            M[2] = self.data[:, :, 2].ravel().max()
        if tail_elimination == True:
            wynik2: np.ndarray = np.zeros((3, 256))
            for i in range(0, self.data.shape[0]):
                for j in range(0, self.data.shape[1]):
                    for k in range(0, self.data.shape[2]):
                        wynik2[k][self.data[i][j][k].astype('int64')] += 1
            wynik2 = wynik2.astype('int64')
            for i in range(0, wynik2.shape[0]):
                for j in range(1, wynik2.shape[1]):
                    wynik2[i][j] += wynik2[i][j - 1]
            tmp: np.ndarray = np.zeros(3)
            for i in range(0, wynik2.shape[0]):
                for j in range(0, wynik2.shape[1]):
                    tmp[i] += wynik2[i][j]
            tmp2: np.ndarray = np.zeros((2, 3))
            for i in range(tmp.shape[0]):
                tmp2[0][i] = tmp[i] * 0.05
                tmp2[1][i] = tmp[i] * 0.95
            m: np.ndarray = np.zeros(3)
            M: np.ndarray = np.zeros(3)
            tmp3 = 0
            for i in range(0, wynik2.shape[0]):
                tmp3 = 0
                for j in range(0, wynik2.shape[1]):
                    tmp3 += wynik2[i][j]
                    if tmp3 >= tmp2[0][i]:
                        m[i] = j
                    if tmp3 >= tmp2[1][i]:
                        M[i] = j
        wynik:np.ndarray = np.zeros(self.data.shape)
        for i in range(0, self.data.shape[0]):
            for j in range(0, self.data.shape[1]):
                for k in range(0, self.data.shape[2]):
                    wynik[i][j][k] = (self.data[i][j][k] - m[k]) * (255 / (M[k] - m[k]))
        W: np.ndarray = np.zeros((256, 3))
        R = wynik[:, :, 0].ravel()
        G = wynik[:, :, 1].ravel()
        B = wynik[:, :, 2].ravel()
        for i in range(0, R.shape[0]):
            W[R[i].astype('int64')][0] += 1
        for i in range(0, G.shape[0]):
            W[G[i].astype('int64')][1] += 1
        for i in range(0, B.shape[0]):
            W[B[i].astype('int64')][2] += 1
        f = plt.figure(figsize=(10, 3))
        ax1 = f.add_subplot(131)
        ax2 = f.add_subplot(132)
        ax3 = f.add_subplot(133)
        f.tight_layout()
        ax1.plot(W[:, 0])
        ax2.plot(W[:, 1])
        ax3.plot(W[:, 2])
        plt.show()
        #metoda zwracajaca poprawiony obraz metoda wyrownywania histogramow
        pass
class Image(GrayScaleTransform, ImageComparison, ImageAligning):
    def __init__(self, path: str):
        super().__init__(path)
    #interfejs glownej klasy biblioteki c.d.
    pass

class ImageFiltration:
    def conv_2d(self, image: BaseImage, kernel: np.ndarray, prefix: float = 1) -> BaseImage:
        image = image.data
        image1 = image[:,:,0]
        image2 = image[:,:,1]
        image3 = image[:,:,2]
        image = np.stack((image1, image2, image3), axis=0)
        print('image')
        print(image)
        print('kernel')
        print(kernel.shape)
        wynik: np.ndarray = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        print('wynik')
        print(wynik.shape)
        for i1 in range(0, image.shape[2]):
            for i2 in range(0, image.shape[0]):
                for i3 in range(0, image.shape[1]):
                    for i4 in range(0, kernel.shape[0]):
                        for i5 in range(0, kernel.shape[1]):
                            if(((i2-((kernel.shape[0]-1)/2)+i4)>=0 and ((i3-((kernel.shape[1]-1)/2)+i5)>=0 or (i2-((kernel.shape[0]-1)/2)+i4)<=(image.shape[1]-4)) and (i3-((kernel.shape[1]-1)/2)+i5)<=(image.shape[2]-4))):
                                print('1 = '+str(i2-((kernel.shape[0]-1)/2)+i4))
                                print('2 = '+str(i3-((kernel.shape[1]-1)/2)+i5))
                                print('3 = '+str(image.shape[1]-4))
                                print('4 = '+str(image.shape[2]-4))
                                wynik[i1][i2][i3] += image[i1][int(i2-((kernel.shape[0]-1)/2)+i4)][int(i3-((kernel.shape[1]-1)/2)+i5)] * kernel[i4][i5]
        print('prefix'+str(prefix))
        wynik = wynik * prefix
        wynik1 = wynik[0,:,:]
        wynik2 = wynik[1,:,:]
        wynik3 = wynik[2,:,:]
        print(wynik1.shape)
        wynik = np.stack((wynik1, wynik2, wynik3), axis=2)
        print(wynik1)
        print(wynik.shape)
        imshow(wynik)
        plt.show()
        return wynik
        #kernel: filtr w postaci tablicy numpy
        #prefix: przedrostek filtra, o ile istnieje; Optional - forma poprawna obiektowo, lub domyslna wartosc = 1 - optymalne arytmetycznie
        #metoda zwroci obraz po procesie filtrowania

        pass
class Image(GrayScaleTransform, ImageComparison, ImageAligning, ImageFiltration):

    #interfejs glowny biblioteki c.d.

    pass

x = BaseImage('lena.jpg')
y = Histogram(imread('lena.jpg'))
# z = ImageAligning('lena.jpg')
# z.align_image(True)
g = ImageFiltration()
v = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
print(v.shape)
g.conv_2d(x, v, 1/9)
