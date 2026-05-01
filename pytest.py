import numpy as np
import filtri


def test_konvolucija_vrne_pravilno_obliko():
    slika = np.ones((3, 3), dtype=np.float32)
    jedro = np.ones((3, 3), dtype=np.float32) / 9

    rezultat = filtri.konvolucija(slika, jedro)

    assert rezultat.shape == slika.shape


def test_sobel_vertikalno_vrne_pravilno_obliko():
    slika = np.zeros((5, 5), dtype=np.float32)

    rezultat = filtri.sobel_vertikalno(slika, 0.9, (0, 1.0, 0))

    assert rezultat.shape == (5, 5, 3)


def test_poisci_koticke_rotiranih_kvadratov_vrne_4_kanale():
    slika = np.zeros((5, 5), dtype=np.float32)

    rezultat = filtri.poisci_koticke_rotiranih_kvadratov(slika)

    assert rezultat.shape == (5, 5, 4)



def test_poisci_znak_a_vrne_pravilno_obliko():
    slika = np.zeros((7, 7), dtype=np.float32)

    rezultat = filtri.poisci_znak_a(slika)

    assert rezultat.shape == slika.shape


