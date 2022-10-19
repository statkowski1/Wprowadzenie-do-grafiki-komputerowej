from matplotlib.image import imread
from matplotlib.pyplot import imshow
from matplotlib.image import imsave
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

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
        #inicjalizator wczytujacy obraz do atrybutu data na podstawie sciezki
        pass

    def save_img(self, path: str) -> None:
        imsave(path, self.data)
        #metoda zapisujaca obraz znajdujacy sie w atrybucie data do pliku
        pass

    def show_img(self) -> None:
        imshow(self.data)
        plt.show()
        #metoda wyswietlajaca obraz znajdujacy sie w atrybucie data
        pass

    def get_layer(self, layer_id: int) -> 'BaseImage':
        if(layer_id==0):
            return self.data[:, :, 0]
        elif(layer_id==1):
            return self.data[:, :, 1]
        elif(layer_id==2):
            return self.data[:, :, 2]
        else:
            return 'Błąd, argument może być tylko 0, 1 lub 2!'
        #metoda zwracajaca warstwe o wskazanym indeksie
        pass
    
    def to_hsv(self) -> 'BaseImage':
        R, G, B = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
        M = max(R, G, B)
        m = min(R, G, B)
        V = M / 255
        if(M>0):
            S = 1 - m / M
        else:
            S = 0
        if(G >= B):
            return 1/math.cos(((R - G/2 -B/2)/(math.sqrt(R*R + G*G + B*B - R*G - R*B -G*B))))
        else:
            return 360 - (1/math.cos((R - G/2 - B/2)/(math.sqrt(R*R + G*G + B*B - R*G - R*B - G*B))))
        #metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        #metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        pass

    def to_hsi(self) -> 'BaseImage':

        #metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        #metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        pass

    def to_hsl(self) -> 'BaseImage':

        #metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        #metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        pass

    def to_rgb(self) -> 'BaseImage':

        #metoda dokonujaca konwersji obrazu w atrybucie data do modelu rgb
        #metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        pass

x = BaseImage('lena.jpg')
#x.show_img()
x.save_img('f.png')
print(x.get_layer(0))
x.to_hsv()
        
