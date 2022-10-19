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
        
        