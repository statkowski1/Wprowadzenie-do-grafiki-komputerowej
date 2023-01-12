class ImageFiltration:
    def conv_2d(self, image: BaseImage, kernel: np.ndarray, prefix: float = 1) -> BaseImage:
        image = image.data
        wynik = np.zeros(image.shape)
        k1 = int((kernel.shape[0]-1)/2)
        k2 = int((kernel.shape[1]-1)/2)
        print(k1)
        for k in range(0, wynik.shape[2]):
            for i in range(k1, wynik.shape[0]-k1):
                for j in range(k2, wynik.shape[1]-k2):
                    for r in range(0, kernel.shape[0]):
                        for s in range(0, kernel.shape[1]):
                            wynik[i][j][k] += image[i+r-k1][j+s-k2][k] + kernel[r][s]
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
        wynik = wynik.astype('int64')
        print(wynik[:, 0:2, 0])
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
#g = ImageFiltration()
#v = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
#print(v.shape)
#g.conv_2d(x, v, 1/9)
#x.show_img()

q = ImageFiltration()
# tab = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
# np.set_printoptions(threshold=sys.maxsize)
# prefix = 1/9

tab = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
prefix = 1/256
q.conv_2d(x, tab, prefix)


#y.to_cumulated()
