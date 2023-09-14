import sys
import numpy as np
import matplotlib
import matplotlib.image as image
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog

def Open_file():
    """Otevře dialog pro výběr souboru a vrátí cestu k vybranému souboru."""
    root = Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(parent=root,initialdir="/",title='Please select a directory')
    return filename

def Find_closest_number_index(sorted_list, target):
    """
    Nalezne index nejbližšího čísla k danému číslu v seřazeném seznamu.

    parametry:
        sorted_list -- seřazený seznam
        target -- hledané číslo
    """
    #Zavedeni intervalu, ve kterem se hleda
    left, right = 0, len(sorted_list) - 1

    #Zavedeni nezname pro nejblizsi index a nejmensi rozdil
    closest_index = None
    min_diff = float('inf')

    #Provedeni puleni intervalu
    while left <= right:
        #Nalezeni stredu intervalu
        mid = left + (right - left) // 2

        #Zavedeni hodnoty stredu a rozdilu mezi hledanym cislem a aktualnim cislem 
        current_number = sorted_list[mid]
        current_diff = abs(target - current_number)

        #Pokud je minimalni rozdil mensi nez aktualni, aktualizujeme rozdil a nejblizsi index
        if current_diff < min_diff:
            min_diff = current_diff
            closest_index = mid

        #Pokud je stejny vratime pozici stredu
        if current_number == target:
            return mid
        
        #Pokud je aktualni cislo mensi nez hledane posuneme levou hranici doprava (jelikoz se jedna o serazeny seznam)
        elif current_number < target:
            left = mid + 1

        #Pokud je aktualni cislo vetsi nez hledane posuneme pravou hranici doleva
        else:
            right = mid - 1

    #Vraceni indexu nejblizsiho cisla k hledanemu cislu
    return closest_index

def Calculate_rank(A, quality, DOWNSAMPLE = 1):
    """
    Vypočítá hodnotu počtu ranků podle požadované kvality.
    
    parametry:
        A -- vstupní obrázek
        quality -- požadovaná kvalita v procentech
        DOWNSAMPLE -- Faktor zmenšení obrázku (výchozí hodnota 1)
    """
    #Rozdeleni podle barev a pokud si to uzivatel vybral, tak i rovnou zmenseni
    R = A[::DOWNSAMPLE, ::DOWNSAMPLE, 0]
    G = A[::DOWNSAMPLE, ::DOWNSAMPLE, 1]
    B = A[::DOWNSAMPLE, ::DOWNSAMPLE, 2]

    #Prevedeni do cernobile pro jednodussi zachazeni 
    X = 0.2989 * R + 0.5870 * G + 0.1140 * B

    #Provedeni singularniho rozkladu
    U, S, VT = np.linalg.svd(X)

    #Spocitani rozptylu zobrazeneho kazdou singularni hodnotou
    total_S = S.sum()
    n = len(S)
    ranks = range(1,  n + 1)
    quality_retained = 100 * np.cumsum(S) / total_S

    #Nalezeni index cisla v 'info_retained', ktere je nejbliz pozadovane kvalite
    rank = Find_closest_number_index(quality_retained, quality)

    #Vraceni ranku odpovidajiciho pozadovane kvalite
    return rank

def Compression_Color(A, r, DOWNSAMPLE = 1):
    """
    Kompresuje barevný obrázek pomocí singulárního rozkladu.
    
    parametry:
        A -- vstupní obrázek
        r -- počet zachovaných singulárních hodnot
        DOWNSAMPLE -- Faktor zmenšení obrázku (výchozí hodnota 1)
    """
    #Rozdeleni obrazku podle RGB
    R = A[:: DOWNSAMPLE, :: DOWNSAMPLE, 0]
    G = A[:: DOWNSAMPLE, :: DOWNSAMPLE, 1]
    B = A[:: DOWNSAMPLE, :: DOWNSAMPLE, 2]
    
    #Docasna zmena na cernobile pro jednodussi zachazeni
    X = 0.2989 * R + 0.5870 * G + 0.1140 * B

    #Pouziti singularniho rozkladu na puvodni matici
    U, S, VT = np.linalg.svd(X)

    #Spocitani rozptylu zobrazeneho kazdou singularni hodnotou
    total_S = S.sum()
    n = len(S)
    ranks = range(1,  n + 1)
    quality_retained = 100 * np.cumsum(S) / total_S
    
    #Pouziti singularniho rozkladu pro jednotlive barvy
    Ur, Sr, VTr = np.linalg.svd(R)
    Ug, Sg, VTg = np.linalg.svd(G)
    Ub, Sb, VTb = np.linalg.svd(B)

    #Zmenseni matic pro jednotlive barvy
    #Pro cervenou
    Ur_compressed = Ur[:,:r]
    Sr_compressed = Sr[:r]
    VTr_compressed = VTr[:r,:]

    #Pro zelenou
    Ug_compressed = Ug[:,:r]
    Sg_compressed = Sg[:r]
    VTg_compressed = VTg[:r,:]

    #Pro modrou
    Ub_compressed = Ub[:,:r]
    Sb_compressed = Sb[:r]
    VTb_compressed = VTb[:r,:]

    #Rekonstrukce zmensenych matic pro jednotlive barvy
    R_compressed = np.dot(Ur_compressed * Sr_compressed, VTr_compressed)
    G_compressed = np.dot(Ug_compressed * Sg_compressed, VTg_compressed)
    B_compressed = np.dot(Ub_compressed * Sb_compressed, VTb_compressed)

    #Spojeni do jednoho obrazu
    Compressed_image = np.stack((R_compressed, G_compressed, B_compressed), axis = 2)
    Compressed_image = Compressed_image.astype('uint8')

    #Vraceni zmenseneho obrazu, poctu singularnich hodnot, seznamu hodnot ranku, seznamu hodnot kvality v procentech, seznamu singularnich hodnot
    return Compressed_image, n, ranks, quality_retained, S

def Compression_Gray(A, r):
    """
    Provede kompresi nebarevného obrázku použitím sinfulárního rozkladu.

    argumenty:
        A -- vstupní obrázek
        r -- počet zachovaných singulárních hodnot
    """
    #Provedeni singularniho rozkladu
    U, S, VT = np.linalg.svd(A)

    #Spocitani rozptylu zobrazeneho kazdou singularni hodnotou
    total_S = S.sum()
    n = len(S)
    ranks = range(1,  n + 1)
    quality_retained = 100 * np.cumsum(S) / total_S

    #Zmenseni matic podle ranku r
    U_compressed = U[:, :r]
    S_compressed = S[:r]
    VT_compressed = VT[:r, :]

    #Rekonstrukce zpet v jednu matici
    Compressed_image = np.dot(U_compressed * S_compressed, VT_compressed)
    Compressed_image = Compressed_image.astype('uint8')

    #Vraceni zmenseneho obrazu, poctu singularnich hodnot, seznamu hodnot ranku, seznamu hodnot kvality v procentech, seznamu singularnich hodnot
    return Compressed_image, n, ranks, quality_retained,S

def Graph_quality(compressed_image, n, ranks, quality_retained, r, gray):
    """
    Zobrazí komprimovaný obrázek a graf závislosti kvality na ranku.

    argumenty:
        compressed_image -- komprimovaný obrázek
        n -- počet singulárních hodnot v původní matici
        ranks -- seznam hodnot ranku
        quality_retained -- seznam hodnot kvality v procentech
        r -- počet zachovaných singulárních hodnot
        gray -- indikátor, zda je obrázek černobílý (1 pro černobílý, jinak 0)
    """

    figure, axis = plt.subplots(2, 1, figsize = (16, 4))
    #Zobrazeni obrazku
    plt.sca(axis[0]) 
    plt.imshow(compressed_image)
    if gray == 1:
        plt.set_cmap('gray')
    plt.axis('off')
    plt.title('Zmenšený obrázek\nRank: ' + str(r))

    #Zobrazeni grafu
    plt.sca(axis[1])
    plt.plot(ranks, quality_retained, color='blue')
    plt.xlim(0, n)
    plt.ylim([0, 100])
    plt.xlabel('Rank')
    plt.ylabel('Kvalita v %')
    plt.grid()
    plt.title('Závislost ranku na kvalitě')
    plt.show()

def Graph_singular(compressed_image, ranks, n, S, r, gray):
    """
    Zobrazí komprimovaný obrázek a graf závislosti singulárních hodnot na ranku.

    argumenty:
        compressed_image -- komprimovaný obrázek
        ranks -- seznam hodnot ranku
        n -- počet singulárních hodnot v původní matici
        S -- seznam singulárních hodnot
        r -- počet zachovaných singulárních hodnot
        gray -- indikátor, zda je obrázek černobílý (1 pro černobílý, jinak 0)
    """
    figure, axis = plt.subplots(2, 1, figsize = (16, 4))
    #Zobrazeni obrazku
    plt.sca(axis[0])
    plt.imshow(compressed_image)
    if gray == 1:
        plt.set_cmap('gray')
    plt.axis('off')
    plt.title('Zmenšený obrázek\nRank: ' + str(r))

    #Zobrazeni grafu
    plt.sca(axis[1])
    plt.plot(ranks, S, color='blue')
    plt.xlim(0, n)
    plt.ylim([0, max(S)])
    plt.xlabel('Rank')
    plt.ylabel('Singulární hodnota')
    plt.grid()
    plt.title('Závislost ranku na hodnotach singulárních hodnot')
    plt.show()
    
def Compare(A, compressed_image, gray):
    """
    Zobrazí vedle sebe komprimovaný a původní obrázek.

    argumenty
        A -- původní obrázek
        compressed_image -- komprimovaný obrázek
        gray -- indikátor, zda je obrázek černobílý (1 pro černobílý, jinak 0)
    """
    figure, axis = plt.subplots(1, 2, figsize = (16,9))

    #Zobrazení původního obrázku
    plt.sca(axis[0])
    plt.imshow(A)
    if gray == 1:
        plt.set_cmap('gray')
    plt.axis('off')
    plt.title('Původní obrázek')

    #Zobrazení zmenšeného obrázku
    plt.sca(axis[1])
    plt.imshow(compressed_image)
    plt.axis('off')
    plt.title('Zmenšený obrázek')

    plt.show()

def calculate_mse(original_image, compressed_image):
    """
    Vypočítá a vrátí MSE (Střední kvadratickou chybu) mezi původním a komprimovaným obrázkem.

    argumenty
        original_image -- původní obrázek
        compressed_image -- komprimovaný obrázek
    """
    #Oznaceni si rozdilu puvodniho obrazku se zmensenym
    diff = original_image - compressed_image

    #Umocneni na druhou
    squared_diff = diff**2

    #Vypocet prumeru umocnenych rozdilu
    mse = np.mean(squared_diff)
    
    return mse

def calculate_psnr(mse):
    """Vypočítá a vrátí PSNR (Peak Signal-to-Noise Ratio) na základě MSE."""
    #PSNR vypocet. 255 jelikoz se predokladaji 8 bitove obrazky
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    
    return psnr

def calculate_ssim(original_image, compressed_image):
    """
    Vypočítá a vrátí SSIM (Structural Similarity Index) mezi původním a komprimovaným obrázkem.

    argumenty
        original_image -- původní obrázek
        compressed_image -- komprimovaný obrázek
    """
    #Zajistuje minimalizaci problemu se stabilitou
    k1 = 0.01
    k2 = 0.03

    #Konstanta pro porovnani jasu
    c1 = (k1 * 255)**2

    #Konstanta pro porovnani kontrastu
    c2 = (k2 * 255)**2
    
    #Prumerna hodnota puvodniho obrazku
    mu_original = original_image.mean()

    #Prumerna hodnota zmenseneho obrazku
    mu_compressed = compressed_image.mean()
    
    #Rozptyl puvodniho obrazku
    sigma_original_sq = np.mean((original_image - mu_original)**2)

    #Rozptyl zmenseneho obrazku
    sigma_compressed_sq = np.mean((compressed_image - mu_compressed)**2)
    
    sigma12 = np.mean((original_image - mu_original) * (compressed_image - mu_compressed))
    
    numerator = (2 * mu_original * mu_compressed + c1) * (2 * sigma12 + c2)
    denominator = (mu_original**2 + mu_compressed**2 + c1) * (sigma_original_sq + sigma_compressed_sq + c2)
    
    ssim = numerator / denominator
    
    return ssim

#Neznama, ktera kontroluje, zda se program jiz zeptal uzivatele, jestli chce obrazek zmensit
down = 0
#Neznama, ktera kontroluje, zda ma byt obrazek cernobily
gray = 0
i = 0

#While loop, ktery bezi dokud neni uzivatel spokojen s obrazkem
while i != 1:
    input('Nyní načtěte libovolný obraz z vašeho počítače. Klikněte na enter...')

    #Nacteme obrazek
    file_name = Open_file()
    A = plt.imread(file_name)
    A_original = A

    #Nastavime velikost obrazku
    plt.figure(figsize = (16, 9))

    #Zobrazime obraz
    plt.imshow(A)
    plt.axis('off')
    plt.title('Zvolený obrázek. (Pro pokračování zavřete)')
    plt.show()

    #Zjistime velikost obrazu
    img_width, img_height = A.shape[1], A.shape[0]

    #Zjisitme, zda je uzivatel spokojen s obrazkem, pokud ne, spusti se loop znova
    answer = input("Je toto obrázek, se kterým chcete pracovat? Odpovězte 'Ano', nebo 'Ne'\n")

    if answer.lower() == 'ano':
        i = 1

    elif answer.lower() == 'ne':
        continue

    else:
        print('Neznámý příkaz. Zkuste to znovu.')

#Zjisitme, jakym zpusobem chce zadat rank
try:
    answer = int(input('Přejete si:\n[1] Určit kolik prvků (singulárních hodnot) se zachová\n[2] Určit kvalitu výsledného obrazu (v procentech vůči původnímu)\n'))
except ValueError:
    print("Neplatný vstup. Vstup musí být celé číslo.")
    sys.exit(1)

#Zadani ranku rucne
if answer == 1:
    try:
        answer_rank = int(input('Přejete si:\n[1] Zadat rank ručně (Nesmí být více než ' + str(min(img_height, img_width)) + ')\n[2] Zadat (v procentech) kolik prvků zachovat\n'))
    except ValueError:
        print('Neplatný vstup. Vstup musí být celé číslo.')
        sys.exit(1)

    #Zadani ranku cislem
    if answer_rank == 1:
        #Nesmi být větší než libovolná strana obrazku a zaroven musi byt vetsi nez nula
        try:
            r = int(input('Zadejte rank. Nesmí být více než ' + str(min(img_height, img_width)) + '\n'))
        except ValueError:
            print("Neplatný vstup. Vstup musí být celé číslo.")
            sys.exit(1)
        
        if r > min(img_width, img_height) and r <= 0:
            sys.exit(1)

    #Zadani ranku procentualne z mensi strany
    elif answer_rank == 2:
        try:
            relative_rank = int(input('Kolik procent chcete zachovat?\n'))
        except ValueError:
            print("Neplatný vstup. Vstup musí být celé číslo.")
            sys.exit(1)

        relative_rank /= 100

        r = int(relative_rank * min(img_height, img_width))
        print('Rank: ', r, '\n')

#Vypocet ranku podle kvality
elif answer == 2:
    try:
        answer_percent = int(input('Kolik procent kvality chcete zachovat? (Mezi 40 a 100)\n'))
    except ValueError:
        print("Neplatný vstup. Vstup musí být celé číslo.")
        sys.exit(1)

    #Kontrola, ze jsou procenta spravne
    if answer_percent > 40 and answer_percent <= 100:
        #Kontrola, zda zmensit obraz
        answer_downsample = input("Přejete si i zmenšit obraz? (Bude zachován každý i-tý pixel.) Odpovězte 'Ano', nebo 'Ne'\n")
        down = 1

        #Volani funkce, ktera vypocita rank podle kvality
        if answer_downsample.lower() == 'ano':
            try:
                downsample = int(input('O kolik? (Doporučuje se velmi malé číslo a musí být menší než menší strana obrázku)\n'))
            except:
                print("Neplatný vstup. Vstup musí být celé číslo.")
                sys.exit(1)
            
            if downsample > min(img_height, img_width) or downsample < 0:
                print('Neplatný vstup. Vstup musí být v rozměrech obrázku.')
                sys.exit(1)

            else:
                r = Calculate_rank(A, answer_percent, DOWNSAMPLE = downsample)
                down = 2
                print('Rank: ', r)

        elif answer_downsample.lower() == 'ne':
            r = Calculate_rank(A, answer_percent)
            print('Rank: ',r)
        
        else:
            print("Neplatný vstup. Vstup musí být 'Ano', nebo 'Ne'")
            sys.exit(1)

    else:
        print('Neplatný vstup. Procenta musí být mezi 40 a 100.')
        sys.exit(1)
else:
    sys.exit(1)

if down == 2:
    compressed_image, n, ranks, quality_retained, S = Compression_Color(A, r, downsample)

else:
    try: 
        answer_img = int(input('Přejete si s vaším obrázkem pracovat v:\n[1] Původní podobě\n[2] Podobě zmenšené na dvě dimenze (bez barev)\n[3] Šedém spektru\n'))
    except ValueError:
        print("Neplatný vstup. Vstup musí být celé číslo.")
        sys.exit(1)

    #Prace s puvodnim obrazem
    if answer_img == 1:
        #Kontrola, zda se program jiz zeptal na zmenseni
        if down == 0:
            answer_down = input("Přejete si i zmenšit obrázek? (Zachová se každý i-tý pixel.) Odpovězte 'Ano', nebo 'Ne'\n")

            if answer_down.lower() == 'ano':
                try:
                    downsample = int(input('O kolik? (Doporučuje se velmi malé číslo a musí být menší než menší strana obrázku)\n'))
                except ValueError:
                    print("Neplatný vstup. Vstup musí být celé číslo.")
                    sys.exit(1)
            
                if downsample > min(img_height, img_width) or downsample < 0:
                    print('Neplatný vstup. Vstup musí být v rozměrech obrázku.')
                    sys.exit(1)
            
                else:
                    compressed_image, n, ranks, quality_retained, S = Compression_Color(A, r, DOWNSAMPLE = downsample)

            elif answer_down.lower() == 'ne':
                compressed_image, n, ranks, quality_retained, S  = Compression_Color(A, r)
                plt.imshow(compressed_image)
                plt.show
        
            else:
                print("Neplatný vstup. Vstup musí být 'Ano', nebo 'Ne'")
                sys.exit(1)
    
        elif down == 1:
            compressed_image, n, ranks, quality_retained, S = Compression_Color(A, r)
    
        else:
            compressed_image, n, ranks, quality_retained, S = Compression_Color(A, r, downsample)

    #Prace s 2D obrazkem
    elif answer_img == 2:
        A = np.mean(A, axis =- 1)
        compressed_image, n, ranks, quality_retained, S = Compression_Gray(A, r)

    #Prace s cernobilym obrazkem
    elif answer_img == 3:
        A = np.dot(A[...,:3], [0.2989, 0.5870, 0.1140])
        compressed_image, n, ranks, quality_retained, S = Compression_Gray(A, r)
        print(np.shape(compressed_image))
        gray = 1

    else:
        print('Neplatný vstup.')
        sys.exit(1)

#Nacteni obrazku
img = plt.imshow(compressed_image)

if gray == 1:
    plt.set_cmap('gray')

#Zobrazeni obrazku
plt.axis('off')
plt.title('Rank: ' + str(r))
plt.show()

answer = 0

while answer != 6:
    try:
        answer = int(input('Přejete si:\n[1] Zobrazit srovnání původního a zmenšeného obrázku.\n[2] Zobrazit graf závislosti ranku na kvalitě.\n[3] Zobrazit graf závislosti singulárních hodnot na ranku.\n[4] Porovnat kvalitu výsledného obrázku. (mse, psnr, ssim).\n[5] Znovu zobrazit obrázek.\n[6] Skončit.\n'))
    except ValueError:
        print("Neplatný vstup. Vstup musí být celé číslo.")
        sys.exit(1)
    
    if answer == 1:
        Compare(A_original, compressed_image, gray)

    elif answer == 2:
        Graph_quality(compressed_image, n, ranks, quality_retained, r, gray)

    elif answer == 3:
        Graph_singular(compressed_image, ranks, n, S, r, gray)

    elif answer == 4:
        if down == 2:
            print('Jelikož jste se rozhodli zmenšit obrázek, tato funkce nefunguje.')
        else:
            mse = calculate_mse(A, compressed_image)
            psnr = calculate_psnr(mse)
            ssim = calculate_ssim(A, compressed_image)

            print("Quality Metrics:")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
            print(f"Structural Similarity Index (SSIM): {ssim:.4f}")
    elif answer == 5:
        img = plt.imshow(compressed_image)

        if gray == 1:
            plt.set_cmap('gray')

        #Zobrazeni obrazku
        plt.axis('off')
        plt.title('Rank: ' + str(r))
        plt.show()
    elif answer == 6:
        continue

    else:
        print('Neplatný vstup.\n')

sys.exit(1)
