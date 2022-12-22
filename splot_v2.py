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
