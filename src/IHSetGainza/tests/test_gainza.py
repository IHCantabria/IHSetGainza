# Loading libraries

import os
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
import rasterio
from affine import Affine
from IHSetGainza import IHSetGainza

pt = r'C:\IH-SET\IHSetGainza-github\data\\'
ptsmc = r'C:\IH-SET\IHSetGainza-github\data\Mopla Narrabeen\Narrabeen\Alternativa 1\Mopla\\'

fid10 = open('bitacora.dat', 'w+')

with open(os.path.join(pt, '0_Data', 'casos.dat'), 'r') as file_clavesoluca:
    clavesoluca = file_clavesoluca.read().splitlines()

with open(os.path.join(pt, '0_Data', 'mallas.dat'), 'r') as file_clavesmalla:
    clavesmalla = file_clavesmalla.read().splitlines()

ncasos = 100

with open(os.path.join(pt, '0_Data', 'Casos_Transporte.dat'), 'w') as fid:
    for i in range(ncasos):
        nombre = clavesmalla[i] + clavesoluca[i]
        fid.write(nombre + '\n')

with open(os.path.join(pt, '0_Data', 'Casos_Transporte.dat'), 'r') as file_cases:
    a = file_cases.read()

s = len(a)
k = 1
b = []
f = []

for i in range(0, s - 10, 21):
    b.append(a[i:i + 10])
    k += 1

r = 1
for j in range(0, s, 21):
    f.append(a[j:j + 20])
    r += 1

destination_folder_TOT = 'C:\IH-SET\IHSetGainza-github\data\\1sacarTOT'
if not os.path.exists(destination_folder_TOT):
    os.makedirs(destination_folder_TOT)

IHSetGainza.Step1_sacarTOT(a, b, ptsmc, destination_folder_TOT)

destination_folder_rotura = 'C:\IH-SET\IHSetGainza-github\data\\2rotura'
if not os.path.exists(destination_folder_rotura):
    os.makedirs(destination_folder_rotura)

numeroperfiles = 63
resolucion = 20
resolucionperfil = 2

Tm_ptomed = np.loadtxt(os.path.join(pt, '0_Data', 'Tm_100casos.dat'))
Tp_ptomed = np.loadtxt(os.path.join(pt, '0_Data', 'Tp_100casos.dat'))
Dirm_ptomed = np.loadtxt(os.path.join(pt, '0_Data', 'dir_100casos.dat'))

Xmin = 342413.868 - 100
Ymin = 6265005.804 - 100
Xmax = 345966.874
Ymax = 6269859.636

XYminmax = [Xmin, Ymin, Xmax, Ymax]

IHSetGainza.Step2_rotura(b, f, clavesoluca, numeroperfiles, resolucion, resolucionperfil, Tm_ptomed, Tp_ptomed, XYminmax, pt, ptsmc, destination_folder_TOT, destination_folder_rotura)

destination_folder_reconstruccion = 'C:\IH-SET\IHSetGainza-github\data\\3reconstruccion'
if not os.path.exists(destination_folder_reconstruccion):
    os.makedirs(destination_folder_reconstruccion)

dirmin = 60
dirmax = 170

centers = np.loadtxt(pt + '0_Data\\100casos.dat')
A = np.loadtxt(pt + '0_Data\Crea_dow_CJC\dow.dat')
RES = np.load(destination_folder_rotura + '//RES_ROTURA.npy')

IHSetGainza.Step3_reconstruccion(centers, A, dirmin, dirmax, RES, ncasos, destination_folder_rotura, destination_folder_reconstruccion)

destination_folder_distancia = 'C:\IH-SET\IHSetGainza-github\data\\4distancia'
if not os.path.exists(destination_folder_distancia):
    os.makedirs(destination_folder_distancia)

nombre_perfiles = os.path.join(pt, '0_Data', 'perfiles', 'perfiles.dat')
perfiles = np.loadtxt(nombre_perfiles)

file_m = os.path.join(destination_folder_rotura, 'ini_perfil.npy')
aa = np.load(file_m)

IHSetGainza.Step4_distancia(perfiles, numeroperfiles, aa, destination_folder_reconstruccion, destination_folder_distancia)

destination_folder_eddy = 'C:\IH-SET\IHSetGainza-github\data\\5eddy'
if not os.path.exists(destination_folder_eddy):
    os.makedirs(destination_folder_eddy)

g = 9.8
ro = 1.03 * 10**3
cf = 0.01
m = 0.13
gamma = 0.8
K = 5 / (2 * (3 * (0.8**2) + 8))
M = 10

IHSetGainza.Step5_eddy(ro, g , M, numeroperfiles, destination_folder_reconstruccion, destination_folder_distancia, destination_folder_eddy)

destination_folder_ponderar = 'C:\IH-SET\IHSetGainza-github\data\\6ponderar'
if not os.path.exists(destination_folder_ponderar):
    os.makedirs(destination_folder_ponderar)

IHSetGainza.Step6_ponderar(ro, g, cf, m, gamma, K, M, numeroperfiles, destination_folder_reconstruccion,destination_folder_distancia, destination_folder_eddy, destination_folder_ponderar)

img = rasterio.open(os.path.join(pt, '0_Data', 'Narrabeen_CJC_UTM56S_WI2015.tif'))
if isinstance(img.transform, Affine):
    transform = img.transform
else:
    transform = img.affine

N = img.width
M = img.height
dx = transform.a
dy = transform.e
minx = transform.c
maxy = transform.f

# Read the image data, flip upside down if necessary
red = img.read(1)
green = img.read(2)
blue = img.read(3)
if dy < 0:
    dy = -dy
    red = np.flip(red, 0)
    green = np.flip(green, 0)
    blue = np.flip(blue, 0)

# Generate X and Y grid locations
xdata = minx + dx/2 + dx*np.arange(N)
ydata = maxy - dy/2 - dy*np.arange(M-1,-1,-1)

extent = [xdata[0], xdata[-1], ydata[0], ydata[-1]]
color_image = np.stack((red, green, blue), axis=-1)
plt.imshow(np.flipud(color_image), extent=extent)

# Load data from text file
perfiles = np.loadtxt('C:\IH-SET\IHSetGainza-github\data\\0_data\perfiles\perfiles.dat')
numeroperfiles = len(perfiles) // 2

file_path = os.path.join(pt, '0_Data\perfiles\Perfiles_Narrabeen.txt')
T = np.loadtxt(file_path, skiprows=1)
x1 = T[:, 1]
y1 = T[:, 2]
x2 = T[:, 3]
y2 = T[:, 4]

plt.plot(x1, y1, 'k.')
plt.plot(x2, y2, 'k.')

x0, y0 = 343112.8709, 6269202.0642
plt.plot(x0, y0, 'mo')

costa = [[x0, y0]]

dir = np.loadtxt(destination_folder_ponderar + '\\lc.dat')

# Reverse order
dir = np.flipud(dir)
x1 = np.flipud(x1)
x2 = np.flipud(x2)
y1 = np.flipud(y1)
y2 = np.flipud(y2)

for i in range(numeroperfiles - 3):
    pc = np.polyfit([x0, x0 - np.cos(np.deg2rad(dir[i]))], [y0, y0 - np.sin(np.deg2rad(dir[i]))], 1)
    pp = np.polyfit([np.mean(x1[i+1:i+3]), np.mean(x2[i+1:i+3])], [np.mean(y1[i+1:i+3]), np.mean(y2[i+1:i+3])], 1)
    xc = (pc[1] - pp[1]) / (pp[0] - pc[0])
    yc = pc[0] * xc + pc[1]
    costa.append([xc, yc])
    x0, y0 = xc, yc

costa = np.array(costa)
plt.plot(costa[:, 0], costa[:, 1], 'r', linewidth=2)

plt.show()

