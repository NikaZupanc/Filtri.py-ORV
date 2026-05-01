import cv2 as cv
import numpy as np


def konvolucija(slika: np.ndarray, jedro: np.ndarray) -> np.ndarray:

    barvna_slika = False
    if len(slika.shape) == 3:
        barvna_slika = True

    visina_slike = slika.shape[0]
    sirina_slike = slika.shape[1]

    nova_slika = np.zeros_like(slika)
    visina_filtra = jedro.shape[0]
    sirina_filtra = jedro.shape[1]

    polovica_visine = (visina_filtra - 1) // 2
    polovica_sirine = (sirina_filtra - 1) // 2

    for i in range(visina_slike):
        for j in range(sirina_slike):

            if(barvna_slika):
                podslika = np.zeros((visina_filtra, sirina_filtra, 3), dtype=slika.dtype)
            else:
                podslika = np.zeros_like(jedro)
            for ji in range(visina_filtra):      
                for jj in range(sirina_filtra):  

                    odmik_y = ji - polovica_visine
                    odmik_x = jj - polovica_sirine    

                    yy = i + odmik_y
                    xx = j + odmik_x 

                    if yy < 0:
                        yy = 0 
                    if yy >= visina_slike:
                          yy = visina_slike - 1 

                    if xx < 0:
                        xx = 0 
                    if xx >= sirina_slike:
                        xx = sirina_slike - 1 

                    if (barvna_slika):
                        podslika[ji, jj, :] = slika[yy, xx, :]
                    else:
                        podslika[ji, jj] = slika[yy, xx]

            sestevek_vrednosti = 0;
            for k in range(visina_filtra):      
                for l in range(sirina_filtra):  
                    sestevek_vrednosti += podslika[k,l] * jedro [k,l]

            nova_slika[i,j] = sestevek_vrednosti
    return nova_slika


def sobel_vertikalno(slika: np.ndarray, max_gradient: np.float32, barva: tuple) -> np.ndarray:

    if len(slika.shape) == 2:   
        rezultat = cv.cvtColor(slika, cv.COLOR_GRAY2BGR)
        sivinska = slika
    else:                       
        rezultat = slika.copy()
        sivinska = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    
    sivinska_glajena = cv.GaussianBlur(sivinska, (3, 3), 0)
    
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])    
    
    gradient_y = cv.filter2D(sivinska_glajena, -1, sobel_y)
    gradient = np.abs(gradient_y)

    maska = gradient > max_gradient
    rezultat[maska] = barva

    return rezultat


def poisci_koticke_rotiranih_kvadratov(slika: np.ndarray) -> np.ndarray:

    if len(slika.shape) == 3:
        sivinska = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    else:
        sivinska = slika.copy()

    invertirana = 1.0 - sivinska

    jedro_Z = np.array([ [-10, -10, -10],
                         [-10,   0, -10],
                         [  0,   2,   0]
    ])

    jedro_D = np.array([ [0, -10, -10],
                         [2,   0, -10],
                         [0, -10, -10]
    ])

    jedro_L = np.array([ [-10, -10, 0],
                         [-10,   0, 2],
                         [-10, -10, 0]
    ])

    jedro_S = np.array([ [  0,   2,   0],
                         [-10,   0, -10],
                         [-10, -10, -10]
    ])

    iskanje_Z = cv.filter2D(invertirana, -1, jedro_Z)
    iskanje_D = cv.filter2D(invertirana, -1, jedro_D)
    iskanje_L = cv.filter2D(invertirana, -1, jedro_L)
    iskanje_S = cv.filter2D(invertirana, -1, jedro_S)
    
    rezultat = np.stack([
        iskanje_Z,
        iskanje_D,
        iskanje_L,
        iskanje_S
    ], axis=2)

    return rezultat



def poisci_znak_a(slika: np.ndarray) -> np.ndarray:

    if len(slika.shape) == 3:
        sivinska = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    else:
        sivinska = slika.copy()

    rezultat = np.ones_like(sivinska)
    
    jedro_A = np.array([ [1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1],
                         [1, 0, 1],
                         [1, 0, 1]
    ])

    iskanje = cv.filter2D(sivinska, -1, jedro_A)

    maska = iskanje < 1.2
    rezultat[maska] = 0.0 

    return rezultat


def oceni_orientacijo_horizonta(slika: np.ndarray) -> float:  

    if len(slika.shape) == 3:
        sivinska = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    else:
        sivinska = slika.copy()

    sivinska_gauss = cv.GaussianBlur(sivinska, (7, 7), 0)

    gy = cv.Sobel(sivinska_gauss,-1,0,1) 
    gx = cv.Sobel(sivinska_gauss,-1,1,0) 

    G = np.sqrt(gx**2 + gy**2)  
    A = np.atan2(gy,gx)         

    maska = G > 0.1
    A_uporabni = A[maska]
    G_uporabni = G[maska]

    A_stopinje = np.degrees(A_uporabni)

    A_stopinje[A_stopinje > 90] -= 180
    A_stopinje[A_stopinje < -90] += 180

    hist = np.zeros(181, dtype=np.float32)

    indeksi = np.round(A_stopinje).astype(int) + 90

    for i in range(len(indeksi)):
        hist[indeksi[i]] += G_uporabni[i]

    naj_indeks = np.argmax(hist)
    kot = naj_indeks - 90

    return kot

