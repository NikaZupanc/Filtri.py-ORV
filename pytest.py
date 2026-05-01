import numpy as np
import filtri


def test_konvolucija_vrne_pravilno_obliko():
    slika = np.ones((3, 3), dtype=np.float32)
    jedro = np.ones((3, 3), dtype=np.float32) / 9

    rezultat = filtri.konvolucija(slika, jedro)

    assert rezultat.shape == slika.shape

