import cv2 as cv
import numpy as np
from pathlib import Path
BASE = Path(__file__).parent
UTILS = BASE / ".utils"
import filtri

if __name__ == '__main__':
    
    #1. FUNKCIJA: konvolucija
    slika = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32)
    jedro = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    slika = slika.astype(np.float32) / 255
    jedro = jedro.astype(np.float32) / 255
    rezultat = filtri.konvolucija(slika, jedro)
    print(rezultat)
    cv.imshow("Konvolucija", rezultat)
    cv.waitKey(0)


    #2. FUNCKIJA: sobelov filter
    slika1 = cv.imread(str(UTILS/"lenna_GRAY.png")).astype(np.float32)/255    
    sobelov_filter = filtri.sobel_vertikalno(slika1, 0.9, (0,1.0,0))
    cv.imshow("Sobelov filter", sobelov_filter)
    cv.waitKey(0)


    #3. FUNKCIJA: poišči kotičke rotiranih kvadratov
    slika2 = cv.imread(str(UTILS/"rotirani_kvadrati.png")).astype(np.float32)/255    
    oglisce = filtri.poisci_koticke_rotiranih_kvadratov(slika2)
    cv.imshow("Poisci koticek rotiranih kvadratov", oglisce)
    cv.waitKey(0)


    #4. FUNKCIJA: iskanje znaka
    slika3 = cv.imread(str(UTILS/"crke.png")).astype(np.float32)/255    
    znak_A = filtri.poisci_znak_a(slika3)
    cv.imshow("Znak A", znak_A)
    cv.waitKey(0)


    #5. FUNKCIJA: oceni orientacijo
    slika4 = cv.imread(str(UTILS/"horizont_rot_45.png")).astype(np.float32)/255    
    horizont = filtri.oceni_orientacijo_horizonta(slika4)
    cv.imshow("Oceni orientacijo horizonta", horizont)
    print("Kot: ", horizont)

    cv.waitKey(0)
    cv.destroyAllWindows()

