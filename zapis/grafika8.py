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
import cv2
import sys

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
        I = R * 0.299 + G * 0.587 + B * 0.114
        GRAY = np.stack((I, I, I), axis=2).astype('uint8')
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
    pass

class Histogram(BaseImage):
    #klasa reprezentujaca histogram danego obrazu
    values: np.ndarray  # atrybut przechowujacy wartosci histogramu danego obrazu

    def plot(self) -> None:
        R = self.data[:, :, 0].ravel()
        G = self.data[:, :, 1].ravel()
        B = self.data[:, :, 2].ravel()
        f = plt.figure(figsize=(10, 3))
        ax1 = f.add_subplot(131)
        ax2 = f.add_subplot(132)
        ax3 = f.add_subplot(133)
        # ax1.set_ylim([-200, 3800])
        # ax2.set_ylim([-200, 2600])
        # ax3.set_ylim([-200, 4500])
        f.tight_layout(pad=0.5)
        ax1.hist(B, bins=256, range=[0, 256], color='red', histtype='step')
        ax2.hist(G, bins=256, range=[0, 256], color='green', histtype='step')
        ax3.hist(R, bins=256, range=[0, 256], color='blue', histtype='step')
        plt.show()
        #metoda wyswietlajaca histogram na podstawie atrybutu values
        pass

class Image(GrayScaleTransform, Histogram):
    pass

class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1
class ImageComparison(BaseImage):
    #Klasa reprezentujaca obraz, jego histogram oraz metody porównania
    def histogram(self) -> Histogram:
        X = self.data
        if(self.color_model != 4):
            R = X[:, :, 0].ravel()
            G = X[:, :, 1].ravel()
            B = X[:, :, 2].ravel()
            print('G(shape) = '+str(len(G)))
            W1: np.ndarray = np.zeros(256)
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
            GRAY = X[:,:,0].ravel()
            print(X[:,:,0].shape)
            W:np.ndarray = np.zeros(256)
            for i in range(0, GRAY.shape[0]):
                W[GRAY[i].astype('uint8')] += 1
            plt.plot(W)
            plt.show()
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

class Image(GrayScaleTransform, Histogram, ImageComparison):
    pass

class Histogram(Histogram):
    #kontunuacja implementacji klasy
    def to_cumulated(self) -> 'Histogram':
        wynik:np.ndarray = np.zeros((256, 3))
        for i in range(0, self.data.shape[0]):
            for j in range(0, self.data.shape[1]):
                for k in range(0, self.data.shape[2]):
                    wynik[self.data[i][j][k].astype('int64')][k] += 1
        for k in range(0, 3):
            for i in range(1, 256):
                wynik[i][k] = wynik[i][k] + wynik[i-1][k]
        f = plt.figure(figsize=(10, 3))
        ax1 = f.add_subplot(131)
        ax2 = f.add_subplot(132)
        ax3 = f.add_subplot(133)
        f.tight_layout()
        ax1.plot(range(0, 256), wynik[:,0])
        ax2.plot(range(0, 256), wynik[:,1])
        ax3.plot(range(0, 256), wynik[:,2])
        plt.show()
        return wynik
        #metoda zwracajaca histogram skumulowany na podstawie stanu wewnetrznego obiektu
        pass

class ImageAligning(BaseImage):
    #klasa odpowiadająca za wyrównywanie hostogramu
    data: np.ndarray
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
        #imshow(np.stack((wynikP, wynikP, wynikP), axis=2).astype('uint8'))
        plt.plot(range(0, 256), wynik)
        plt.show()
        #metoda zwracajaca poprawiony obraz metoda wyrownywania histogramow
        pass
    def align_image2(self, tail_elimination: bool = True):
        wynikP = np.zeros(self.data.shape)
        if tail_elimination == False:
            m = np.zeros(3)
            M = np.zeros(3)
            for i in range(0, 3):
                m[i] = self.data[:, :, i].min()
                M[i] = self.data[:, :, i].max()
            print(m)
            print(M)
        elif tail_elimination == True:
            cumulated: np.ndarray = np.zeros((256, 3))
            m = np.zeros(3)
            M = np.zeros(3)
            for i in range(0, self.data.shape[0]):
                for j in range(0, self.data.shape[1]):
                    for k in range(0, self.data.shape[2]):
                        cumulated[self.data[i][j][0].astype('int64')][k] += 1
            for i in range(1, 256):
                for k in range(0, cumulated.shape[1]):
                    cumulated[i][k] = cumulated[i][k] + cumulated[i - 1][k]
            print('0 = '+ str(cumulated[255][0])+' 1 = '+ str(cumulated[255][1])+' 2 = '+ str(cumulated[255][2]))
            x = np.zeros(3)
            X = np.zeros(3)
            for i in range(0, 3):
                x[i] = 0.05 * cumulated[255][i]
                X[i] = 0.95 * cumulated[255][i]
            for i in range(0, cumulated.shape[0]):
                for j in range(0, cumulated.shape[1]):
                    if (cumulated[i][j] <= x[j]):
                        m[j] = i
                    elif (cumulated[i][j] >= X[j]):
                        M[j] = i
        print(self.data.shape[2])
        for i in range(0, self.data.shape[0]):
            for j in range(0, self.data.shape[1]):
                for k in range(0, self.data.shape[2]):
                    wynikP[i][j][k] = (self.data[i][j][k] - m[k]) * 255 / (M[k] - m[k])
        wynik: np.ndarray = np.zeros((256, 3))
        for i in range(0, wynikP.shape[0]):
            for j in range(0, wynikP.shape[1]):
                for k in range(0, wynikP.shape[2]):
                    wynik[wynikP[i][j][k].astype('int64')][k] += 1
        print(wynikP)
        # imshow(wynikP.astype('int64'))
        f = plt.figure(figsize=(10, 3))
        ax1 = f.add_subplot(131)
        ax2 = f.add_subplot(132)
        ax3 = f.add_subplot(133)
        f.tight_layout()
        ax1.plot(range(0, 256), wynik[:, 0])
        ax2.plot(range(0, 256), wynik[:, 1])
        ax3.plot(range(0, 256), wynik[:, 2])
        plt.show()
    def clahe(self, path: str, option):
        lake_color = cv2.imread(path, cv2.IMREAD_COLOR)
        if option == 1:
            lake_gray = cv2.cvtColor(lake_color, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(4, 4)
            )
            equalized_lake_gray = clahe.apply(lake_gray)
            plt.subplot(221)
            plt.imshow(lake_gray, cmap='gray')

            plt.subplot(222)
            plt.hist(lake_gray.ravel(), bins=256, range=(0, 256), color='gray')

            plt.subplot(223)
            plt.imshow(equalized_lake_gray, cmap='gray')

            plt.subplot(224)
            plt.hist(equalized_lake_gray.ravel(), bins=256, range=(0, 256), color='gray')
        elif option==2:
            lake_rgb = cv2.cvtColor(lake_color, cv2.COLOR_BGR2RGB)
            lake_lab = cv2.cvtColor(lake_color, cv2.COLOR_BGR2LAB)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lake_lab[..., 0] = clahe.apply(lake_lab[..., 0])
            lake_color_equalized = cv2.cvtColor(lake_lab, cv2.COLOR_LAB2RGB)
            plt.subplot(221)
            plt.imshow(lake_rgb)

            plt.subplot(222)
            plt.hist(lake_rgb[..., 0].ravel(), bins=256, range=(0, 256), color='b')
            plt.hist(lake_rgb[..., 1].ravel(), bins=256, range=(0, 256), color='g')
            plt.hist(lake_rgb[..., 2].ravel(), bins=256, range=(0, 256), color='r')

            plt.subplot(223)
            plt.imshow(lake_color_equalized)

            plt.subplot(224)
            plt.hist(lake_color_equalized[..., 0].ravel(), bins=256, range=(0, 256), color='b')
            plt.hist(lake_color_equalized[..., 1].ravel(), bins=256, range=(0, 256), color='g')
            plt.hist(lake_color_equalized[..., 2].ravel(), bins=256, range=(0, 256), color='r')
        plt.show()

class Image(GrayScaleTransform, Histogram, ImageComparison, ImageAligning):
    #interfejs glownej klasy biblioteki c.d.
    pass

class ImageFiltration():
    def conv_2d(self, image: BaseImage, kernel: np.ndarray, prefix: float = 1) -> BaseImage:
        # np.set_printoptions(threshold=sys.maxsize)
        # np.set_printoptions(precision=20)
        image = image.data
        wynik = np.zeros(image.shape)
        k1 = int((kernel.shape[0]-1)/2)
        k2 = int((kernel.shape[1]-1)/2)
        for k in range(0, wynik.shape[2]):
            for i in range(k1, wynik.shape[0]-k1):
                for j in range(k2, wynik.shape[1]-k2):
                    for r in range(0, kernel.shape[0]):
                        for s in range(0, kernel.shape[1]):
                            # print('image = '+str(image[i+r-k1][j+s-k2][k]))
                            # print('kernel = '+str(kernel[r][s]))
                            wynik[i][j][k] += image[i+r-k1][j+s-k2][k] * kernel[r][s]
                    wynik[i][j][k] = wynik[i][j][k] * prefix
        for k in range(0, wynik.shape[2]):
            for i in range(0, wynik.shape[0]):
                for j in range(0, wynik.shape[1]):
                    if(wynik[i][j][k]>255):
                        wynik[i][j][k] = 255
                    if(wynik[i][j][k]<0):
                        wynik[i][j][k] = 0
                    if(i>0 and i<k1 or j>0 and j<k2 or i<wynik.shape[0] and i>wynik.shape[0]-k1 or j<wynik.shape[1] and j>wynik.shape[1]-k2):
                        wynik[i][j][k] = image[i][j][k]
        #wynik = wynik.astype('float64')
        # np.set_printoptions(threshold=sys.maxsize)
        # print('podstawowy obraz')
        # print(image[2:50, 2:50, 0])
        # print('wynik')
        # print(wynik[2:50, 2:50, 0])
        imshow(wynik)
        plt.show()
        return wynik
        #kernel: filtr w postaci tablicy numpy
        #prefix: przedrostek filtra, o ile istnieje; Optional - forma poprawna obiektowo, lub domyslna wartosc = 1 - optymalne arytmetycznie
        #metoda zwroci obraz po procesie filtrowania
        pass

class Image(GrayScaleTransform, Histogram, ImageComparison, ImageAligning, ImageFiltration):
    #interfejs glowny biblioteki c.d.
    pass

class Thresholding(BaseImage):
    def threshold(self, value: int) -> BaseImage:
        R = self.data[:, :, 0].astype('float32')
        G = self.data[:, :, 1].astype('float32')
        B = self.data[:, :, 2].astype('float32')
        I = (R + G + B) / 3
        wynik = np.zeros(R.shape)
        for i in range(0, I.shape[0]):
            for j in range(0, I.shape[1]):
                if(I[i][j]<value):
                    wynik[i][j] = 0
                else:
                    wynik[i][j] = 255
        GRAY = np.stack((wynik, wynik, wynik), axis=2).astype('uint8')
        #wynik = wynik.astype('int64')
        imshow(GRAY)
        plt.show()
        return wynik
        #metoda dokonujaca operacji segmentacji za pomoca binaryzacji
        pass
    def otsu(self, path: str, option):
        lena_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, thresh_otsu = cv2.threshold(
            lena_gray,
            thresh=0,
            maxval=255,
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if option==1:
            plt.imshow(thresh_otsu, cmap='gray')
            #obraz robi sie szary
        elif option==2:
            th_adaptive = cv2.adaptiveThreshold(
                lena_gray,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                thresholdType=cv2.THRESH_BINARY,
                blockSize=13,
                C=8
            )
            plt.imshow(th_adaptive, cmap='gray')
            #obraz wyglada jak szkic (biale tlo, kontury czarne)
        elif option==3:
            canny_edges = cv2.Canny(
                lena_gray,
                16,  # prog histerezy 1
                40,  # prog histerezy 2
                3  # wielkoscc filtra sobela
            )
            plt.imshow(canny_edges, cmap='gray')
            #linie na obrazie przypominaja pnacze (czarne tlo, kontury biale)
        plt.show()

class Image(GrayScaleTransform, ImageComparison, Histogram, ImageAligning, ImageFiltration, Thresholding):
    #interfejs glowny biblioteki c.d.
    # def __init__(self, path: str) -> None:
    #     super().__init__(path)
    pass

class EdgeDetection(BaseImage):
    def canny(self, path: str, option, th0: int=20, th1: int=50, kernel_size: int=3):
        lines_img = cv2.imread('lines.jpg', cv2.IMREAD_GRAYSCALE)
        _, lines_thresh = cv2.threshold(
            lines_img,
            thresh=0,
            maxval=255,
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        lines_edges = cv2.Canny(lines_thresh, th0, th1, kernel_size)
        lines = cv2.HoughLinesP(
            lines_edges,
            2,
            np.pi / 180,
            30
        )
        checkers_img = cv2.imread('checkers.png')
        checkers_gray = cv2.cvtColor(checkers_img, cv2.COLOR_BGR2GRAY)
        checkers_color = cv2.cvtColor(checkers_img, cv2.COLOR_BGR2RGB)
        circles = cv2.HoughCircles(
            checkers_gray,
            method=cv2.HOUGH_GRADIENT,
            dp=2,
            minDist=60,
            minRadius=20,
            maxRadius=100
        )
        if option==1:
            plt.imshow(lines_img, cmap='gray')
        elif option==2:
            plt.imshow(lines_thresh, cmap='gray')
        elif option==3:
            plt.imshow(lines_edges, cmap='gray')
        elif option==4:
            print(len(lines))
        elif option==5:
            result_lines_img = cv2.cvtColor(lines_img, cv2.COLOR_GRAY2RGB)
            for line in lines:
                x0, y0, x1, y1 = line[0]
                cv2.line(result_lines_img, (x0, y0), (x1, y1), (0, 255, 0), 5)
            plt.imshow(result_lines_img)
        elif option==6:
            print(len(circles[0]))
        elif option==7:
            for (x, y, r) in circles.astype(int)[0]:
                cv2.circle(checkers_color, (x, y), r, (0, 255, 0), 4)

            plt.imshow(checkers_color)

        plt.show()

class Image(GrayScaleTransform, ImageComparison, Histogram, ImageAligning, ImageFiltration, Thresholding, EdgeDetection):
    pass

x = BaseImage('sudoku.jpg')
# y = Histogram(imread('lena.jpg'))
# z = ImageAligning('lena.jpg')
# z.align_image(True)
g = ImageFiltration()
#v = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
#v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
v = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#print(v.shape)
#g.conv_2d(x, v, 1)
#x.show_img()

#q = ImageFiltration()
# tab = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
# np.set_printoptions(threshold=sys.maxsize)
# prefix = 1/9

# tab = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
# prefix = 1/256
# q.conv_2d(x, tab, prefix)
# y.to_cumulated()

# r = Image('lena.jpg')
# r.conv_2d(x, [[1, 1, 1], [1, 1, 1], [1, 1, 1]], 1/9)
#r.threshold(90)

# s = Histogram(imread('lena.jpg'))
# s.plot()

# M = imread('lena.jpg')
# np.set_printoptions(threshold=sys.maxsize)
# print(M[2:50, 2:50, 0])

# n = Image('lena.jpg')
# m = Image('checkers.png')
# k = ImageFiltration('sudoku.jpg')
# #n.threshold(127)
# #n.clahe('lena.jpg', 2)
# #n.canny('checkers.jpg', 7)
# k.conv_2d('sudoku.jpg', [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# p = imread('sudoku.jpg')
# for i in range(0, p.shape[0]):
#     for j in range(0, p.shape[1]):
#         for k in range(0, p.shape[2]):
#             print(p[i][j][k])
# p = p.astype('float64')
# imshow(p)
# plt.show()

# r = Image('sudoku.jpg')
# r.threshold(0.3)

j = Image('lena.jpg')
j.align_image2(False)
