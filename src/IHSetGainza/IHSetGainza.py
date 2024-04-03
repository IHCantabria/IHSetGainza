import os
import numpy as np
from math import *
from scipy.interpolate import griddata
from numba import jit, types
from scipy.signal import savgol_filter

def Step1_sacarTOT(a, b, ptsmc, destination_folder_TOT):
    s = len(a)
    for i in range(s // 21):
        w = b[i]
        File_cong = os.path.join(ptsmc, w + 'REF2.DAT')
        with open(File_cong, 'r') as fid:
            print(fid.readline().split())
            for _ in range(1):
                fid.readline()
            x0y0 = list(map(float, fid.readline().split()))
            angu = float(fid.readline())
            LxLy = list(map(float, fid.readline().split()))
            nfnc = list(map(int, fid.readline().split()))
            DxDy = list(map(float, fid.readline().split()))
        
        numdat = nfnc[0] * nfnc[1]
    
        malla = {
            'xIni': x0y0[0],
            'yIni': x0y0[1],
            'angle': angu
        }

        r = 1
        f = []

        for j in range(0, s, 21):
            f.append(a[j:j + 20])
            r += 1

        grdFileHs = os.path.join(ptsmc, 'SP', f[i] + '_Height.grd')
        hs, x, y, xproy, yproy = readGrd(grdFileHs, malla)
        mat_Hs = np.column_stack((np.reshape(x,(numdat,1)), np.reshape(y,(numdat,1)), np.reshape(xproy,(numdat,1)),
                                 np.reshape(yproy,(numdat,1)), np.reshape(hs,(numdat,1))))
    
        grdFileTopo = os.path.join(ptsmc, 'SP', f[i] + '_Topography_out.grd')
        Topo, x, y, xproy, yproy = readGrd(grdFileTopo, malla)
        mat_Topo = np.column_stack((np.reshape(x,(numdat,1)), np.reshape(y,(numdat,1)), np.reshape(xproy,(numdat,1)),
                                 np.reshape(yproy,(numdat,1)), np.reshape(Topo,(numdat,1))))

        grdFileDir = os.path.join(ptsmc, 'SP', f[i] + '_Direction.grd')
        direc, x, y, xproy, yproy = readGrd(grdFileDir, malla)

        diraux = -np.reshape(direc,(numdat,1))+malla['angle']
        mat_Dir = np.column_stack((np.reshape(x,(numdat,1)), np.reshape(y,(numdat,1)), np.reshape(xproy,(numdat,1)),np.reshape(yproy,(numdat,1)), diraux))
        matriz = np.column_stack((mat_Hs[:,0],mat_Hs[:,1],mat_Hs[:,2],mat_Hs[:,3],mat_Hs[:,4],mat_Topo[:,4], mat_Dir[:,4]))
    
        filename = os.path.join(destination_folder_TOT, f[i] + 'TOT.dat')
        np.savetxt(filename, matriz, fmt='%f')
        
    return

def Step2_rotura(b, f, clavesoluca, numeroperfiles, resolucion, resolucionperfil, Tm_ptomed, Tp_ptomed, XYminmax, pt, ptsmc, destination_folder_TOT, destination_folder_rotura):
    Xmin = XYminmax[0]
    Ymin = XYminmax[1]
    Xmax = XYminmax[2]
    Ymax = XYminmax[3]
    
    nnx = round((Xmax - Xmin) / resolucion)
    nny = round((Ymax - Ymin) / resolucion)
    
    x = np.linspace(Xmin, Xmax, nnx + 1)
    y = np.linspace(Ymin, Ymax, nny + 1)
    [xx, yy] = np.meshgrid(x, y)
    
    ini = np.zeros((len(clavesoluca), numeroperfiles, 2))
    RES = np.zeros((len(clavesoluca), numeroperfiles, 7))
    
    
    perfil = []
    nombre = []
    Puntos = np.zeros((2, 2, numeroperfiles))
        
    for i in range(len(clavesoluca)):
        print(i)
        file_m = os.path.join(destination_folder_TOT, f[i] + 'TOT.dat')
        M = np.loadtxt(file_m)
        w = b[i]
        
        File_cong = os.path.join(ptsmc, w + 'REF2.DAT')
        with open(File_cong, 'r') as fid:
            print(fid.readline().split())
            for _ in range(1):
                fid.readline()        
            x0y0 = list(map(float, fid.readline().split()))
            angu = float(fid.readline())
            LxLy = list(map(float, fid.readline().split()))
            nfnc = list(map(int, fid.readline().split()))
            DxDy = list(map(float, fid.readline().split()))
        
        col = nfnc[0]
        fila = nfnc[1]
        Hs = M[:, 4]
        Diri = M[:, 6]
        x = M[:, 2]
        y = M[:, 3]
        d = M[:, 5]
 
        file_v = os.path.join(destination_folder_TOT, f[i] + 'TOT.dat')
        V = np.loadtxt(file_v)
        min_x = min(V[:, 2])
        min_y = min(V[:, 3])       
        
        a = 0
        o = np.where(
            (V[:, 2] >= Xmin - a) & (V[:, 2] <= Xmax + a) & (V[:, 3] >= Ymin - a) & (V[:, 3] <= Ymax + a)
        )
        
        xi = V[o, 2]
        yi = V[o, 3]
        vi = V[o, 5]
        Hi = V[o, 4]
        Dir = V[o, 6]
    
        xi = xi.ravel()
        yi = yi.ravel()
        vi = vi.ravel()
        Hi = Hi.ravel()
        Dir = Dir.ravel()
        
        try:
            vv = np.full(xx.shape, np.nan)
            Hv = np.full(xx.shape, np.nan)
            Dv = np.full(xx.shape, np.nan)       
            
            vv = griddata((xi - (min(xi)), yi - (min(yi))), vi, (xx - (min(xi)), yy - (min(yi))), method='linear')
            Hv = griddata((xi - (min(xi)), yi - (min(yi))), Hi, (xx - (min(xi)), yy - (min(yi))), method='linear')
            Dv = griddata((xi - (min(xi)), yi - (min(yi))), Dir, (xx - (min(xi)), yy - (min(yi))), method='linear')
        
        except:
            print("Lo hago en surfer")
            with open('bitacora.dat', 'a+') as fid10:
                fid10.write(f'Caso {i}')

            vv = surfergriddata(xi - (min(xi)), yi - (min(yi)), vi, xx - (min(xi)), yy - (min(yi)), method='Triangulation')
            Hv = surfergriddata(xi - (min(xi)), yi - (min(yi)), Hi, xx - (min(xi)), yy - (min(yi)), method='Triangulation')
            Dv = surfergriddata(xi - (min(xi)), yi - (min(yi)), Dir, xx - (min(xi)), yy - (min(yi)), method='Triangulation')
            
            vv[np.where(vv == 1.70141e+038)] = np.nan
            Hv[np.where(Hv == 1.70141e+038)] = np.nan
            Dv[np.where(Dv == 1.70141e+038)] = np.nan
            
        svv = np.isnan(vv)
        vv[svv] = 0
        sHv = np.isnan(Hv)
        Hv[sHv] = 0
        sDv = np.isnan(Dv)
        Dv[sDv] = 0          

        for j in range(numeroperfiles):
            perfil = str(j+1)
            nombre = os.path.join(pt, '0_Data', 'Perfiles', 'perfil' + perfil + '.dat')
            puntos = np.loadtxt(nombre)
            x1 = puntos[0, 0]
            y1 = puntos[1, 0]
            x2 = puntos[0, 1]
            y2 = puntos[1, 1]    
            
            npx = round(abs(x2 - x1) / resolucionperfil)
            xp = np.linspace(x1, x2, npx + 1)
            yp = np.linspace(y1, y2, npx + 1)
            
            vvp = griddata((xx.flatten(), yy.flatten()), vv.flatten(), (xp, yp), method='linear')
            Hvp = griddata((xx.flatten(), yy.flatten()), Hv.flatten(), (xp, yp), method='linear')
            Dvp = griddata((xx.flatten(), yy.flatten()), Dv.flatten(), (xp, yp), method='linear')
            
            for tt in range(len(vvp) - 1):
                if vvp[tt] > 0 and vvp[tt + 1] < 0:
                    ini[i, j, :] = [xp[tt], yp[tt]]
                    break
            
            gamma = Hvp / -vvp
            pos = np.array([])
            for l in range(1, len(gamma) - 1):
                if abs(gamma[l]) > 0.55 and abs(gamma[l + 1]) < 0.55 and abs(gamma[l - 1]) < 0.55:
                    pos = np.append(pos, l + 1)
            if len(pos) == 0:
                rot = abs(gamma - 0.55)
                pos = np.argmin(rot)
        
            if isinstance(pos, np.ndarray): 
                pos = int(pos[0])
                
            RES[i, j, 0] = xp[pos]
            RES[i, j, 1] = yp[pos]
            RES[i, j, 2] = vvp[pos]
            RES[i, j, 3] = Hvp[pos]
            RES[i, j, 4] = Dvp[pos]
            RES[i, j, 5] = Tm_ptomed[i]
            RES[i, j, 6] = Tp_ptomed[i]
            
    filename1 = os.path.join(destination_folder_rotura, 'RES_ROTURA.npy')
    filename2 = os.path.join(destination_folder_rotura, 'ini_perfil.npy')
    np.save(filename1, RES)
    np.save(filename2, ini)
    
    return

def Step3_reconstruccion(centers, A, dirmin, dirmax, RES, ncasos, destination_folder_rotura, destination_folder_reconstruccion):

    maxH, minH = centers[:, 0].max(), centers[:, 0].min()
    maxT, minT = centers[:, 1].max(), centers[:, 1].min()
    centers_n = (centers - [minH, minT, 0]) / [maxH - minH, maxT - minT, 180/np.pi]
    
    fechas = A[:, :6]
    Hs = A[:, 6]
    Tm = A[:, 7]
    Dir = np.deg2rad(A[:, 8])
    
    if dirmax<dirmin:
        calmas = np.where((A[:, 8] > dirmax) & (A[:, 8] < dirmin))
        nocalmas = np.where((A[:, 8] >= dirmin) | (A[:, 8] <= dirmax))
    if dirmax>=dirmin:
        calmas = np.where((A[:, 8] < dirmin) | (A[:, 8] > dirmax))
        nocalmas = np.where((A[:, 8] >= dirmin) & (A[:, 8] <= dirmax))

    Hs = np.delete(Hs, calmas)
    Tm = np.delete(Tm, calmas)
    Dir = np.delete(Dir, calmas)

    datos = np.column_stack((Hs, Tm, Dir))
    datos_n = (datos - [minH, minT, 0]) / [maxH - minH, maxT - minT, 1]
    
    RES = np.load(destination_folder_rotura + '//RES_ROTURA.npy')
    U = RES
    s=np.shape(RES)

    # OUTPUTS
    for jj in range(U.shape[1]):
        print(jj)
        [nx, ny] = np.shape(datos_n)
        Propagaciones2 = np.zeros((ncasos, 10))
        resultados = np.zeros((nx, 10))
        resultados_sin_x_y = np.zeros((nx, 13))
    
        for i in range(ncasos):  # Loop over propagated cases
            Propagaciones2[i, 0] = U[i, jj, 0]  # xx
            Propagaciones2[i, 1] = U[i, jj, 1]  # yy
            Propagaciones2[i, 2] = U[i, jj, 3]  # Hsb (rotura en Vmax)
            Propagaciones2[i, 3] = U[i, jj, 4]  # Dirb direcci� del oleaje en la rotura (Vmax)
            Propagaciones2[i, 4] = U[i, jj, 2]  # hb profundidad de rotura (para Vmax)
            Propagaciones2[i, 5] = 0  # Vmax corriente m�ima en el perfil
            Propagaciones2[i, 6] = 0  # DirV Direcci� asociada corriente m�ima en el perfil
            Propagaciones2[i, 7] = U[i, jj, 5]  # Tm
            Propagaciones2[i, 8] = U[i, jj, 6]  # Tp
            Propagaciones2[i, 9] = 0  # Velocidad media perfil
    
        parametros = 10
        ndireccion = 1
        cdireccion = [3]
    
        resultados = InterpolacionRBF_Parametros(parametros, ndireccion, cdireccion, centers_n, datos_n, Propagaciones2)
    
        nocalmas = np.array(nocalmas)
        resultados_sin_x_y = np.column_stack((fechas[nocalmas[0],:], resultados[:,:9]))

        kk = np.where(resultados_sin_x_y[:,8] < 0)
        resultados_sin_x_y[kk, 8] = 0.001
        del kk

        kk = np.where(resultados_sin_x_y[:, 10] > 0)
        resultados_sin_x_y[kk, 10] = -0.01
        resultados_sin_x_y[:, 10] = -resultados_sin_x_y[:, 10]
    
        filename = os.path.join(destination_folder_reconstruccion, 'SerieGowrotura_Perfil_{}.dat'.format(jj+1))
        np.savetxt(filename, resultados_sin_x_y, fmt='%f')

def Step4_distancia(perfiles, numeroperfiles, aa, destination_folder_reconstruccion, destination_folder_distancia):
    x = perfiles[:, 0]
    y = perfiles[:, 1]
    a = 0

    numeroperfiles = len(x) // 2

    cat_op = np.zeros(numeroperfiles)
    cat_cont = np.zeros(numeroperfiles)
    ang = np.zeros(numeroperfiles)

    for i in range(0, numeroperfiles * 2, 2):
        cat_op[a] = x[i + 1] - x[i]
        cat_cont[a] = y[i + 1] - y[i]
        ang[a] = np.degrees(np.arctan(cat_op[a] / cat_cont[a]))
        a += 1
    
    pos = np.zeros((numeroperfiles, 2))

    for j in range(numeroperfiles):
        y = aa[0, j, 1]
        x = aa[0, j, 0]
        pos[j, :] = [x, y]

    filename = os.path.join(destination_folder_distancia, 'pos.dat')
    np.savetxt(filename, pos, fmt='%f')

    cas = 1 
    k = 0
    perfil = str(k+1)
    nombre = os.path.join(destination_folder_reconstruccion, 'SerieGowrotura_Perfil_' + perfil + '.dat')
    b = np.loadtxt(nombre)
    dist = np.zeros((len(b), numeroperfiles))

    for k in range(numeroperfiles):
        perfil = str(k+1)
        nombre = os.path.join(destination_folder_reconstruccion, 'SerieGowrotura_Perfil_' + perfil + '.dat')
        b = np.loadtxt(nombre)

        for l in range(len(b)):
            yb = b[l, 7]
            dist[l, k] = (yb - pos[k, 1]) / np.cos(np.radians(ang[k]))

    dist = np.abs(dist)
    dist[dist <= 0] = 0.1

    filename = os.path.join(destination_folder_distancia, 'distancia_atan_x.dat')
    np.savetxt(filename, dist, fmt='%f')

def Step5_eddy(ro, g , M, numeroperfiles, destination_folder_reconstruccion, destination_folder_distancia, destination_folder_eddy):
    k = 0
    perfil = str(k+1)
    nombre = os.path.join(destination_folder_reconstruccion, 'SerieGowrotura_Perfil_' + perfil + '.dat')
    np_data = np.loadtxt(nombre)
    nump = len(np_data)

    for per in range(numeroperfiles):
        v = np.zeros(nump)
        perfil = str(per+1)
        nombre1 = os.path.join(destination_folder_reconstruccion, 'SerieGowrotura_Perfil_' + perfil + '.dat')
        A = np.loadtxt(nombre1)
    
        nombre2 = os.path.join(destination_folder_distancia, 'distancia_atan_x.dat')
        dist = np.loadtxt(nombre2)
    
        Xb = dist[:, per]
        hb = A[:, 8]
        Hb = A[:, 10]
        Hrms = Hb / np.sqrt(2)

        for i in range(nump):
            Cg = np.sqrt(g * hb[i])
            eps = (1/8) * ro * g * (Hb[i]**2) * (Cg / Xb[i])
            v[i] = M * ((eps / ro)**(1/3)) * Hrms[i]
        
        filename = os.path.join(destination_folder_eddy, 'eddy_p' + perfil + '.dat')
        np.savetxt(filename, v, fmt='%f')

def Step6_ponderar(ro, g, cf, m, gamma, K, M, numeroperfiles, destination_folder_reconstruccion,destination_folder_distancia, destination_folder_eddy, destination_folder_ponderar):
    f = np.loadtxt(destination_folder_reconstruccion + '/SerieGowrotura_Perfil_1.dat')
    distancia = np.loadtxt(destination_folder_distancia + '/distancia_atan_x.dat')

    H_media = np.zeros((numeroperfiles, 1))
    lb_media = np.zeros((numeroperfiles, 1))
    h_media = np.zeros((numeroperfiles, 1))
    dir_media = np.zeros((numeroperfiles, 1))
    eddy_media = np.zeros((numeroperfiles, 1))
    x_media = np.zeros((numeroperfiles, 1))
    y_media = np.zeros((numeroperfiles, 1))
    
    for i in range(0, numeroperfiles):
        print(i)
        a = np.loadtxt(destination_folder_reconstruccion + f'/SerieGowrotura_Perfil_{i+1}.dat')
        d = np.loadtxt(destination_folder_eddy + f'/eddy_p{i+1}.dat')

        eddy = d
        Hs = a[:, 8]
        h = a[:, 10]
        dir = a[:, 9]
        Xb = distancia[:, i]
        x = a[:, 6]
        y = a[:, 7]

        E = (1 / 8) * ro * g * Hs**2 * np.sqrt(g * h)

        Etot = np.sum(E)

        pond = E / Etot
        H_pond = pond * Hs
        lb_pond = pond * Xb
        h_pond = pond * h
        dir_pond = pond * dir
        eddy_pond = pond * eddy
        x_pond = pond * x
        y_pond = pond * y

        H_media[i, 0] = np.sum(H_pond) / np.sum(pond)
        lb_media[i, 0] = np.sum(lb_pond) / np.sum(pond)
        h_media[i, 0] = np.sum(h_pond) / np.sum(pond)
        dir_media[i, 0] = np.sum(dir_pond) / np.sum(pond)
        eddy_media[i, 0] = np.sum(eddy_pond) / np.sum(pond)
        x_media[i, 0] = np.sum(x_pond) / np.sum(pond)
        y_media[i, 0] = np.sum(y_pond) / np.sum(pond)

    H_media = savgol_filter(H_media.ravel(), 5, 1)
    lb_media = savgol_filter(lb_media.ravel(), 5, 1)
    eddy_media = savgol_filter(eddy_media.ravel(), 5, 1)
    h_media = savgol_filter(h_media.ravel(), 5, 1)
    x_media = savgol_filter(x_media.ravel(), 5, 1)
    y_media = savgol_filter(y_media.ravel(), 5, 1)
    dir_media = savgol_filter(dir_media.ravel(), 5, 1)

    np.savetxt(destination_folder_ponderar + '\\H_pond_smooth.dat', H_media, fmt='%.6f')
    np.savetxt(destination_folder_ponderar + '\\xb_pond_smooth.dat', lb_media, fmt='%.6f')
    np.savetxt(destination_folder_ponderar + '\\hh_pond_smooth.dat', h_media, fmt='%.6f')
    np.savetxt(destination_folder_ponderar + '\\dir_pond_smooth.dat', dir_media, fmt='%.6f')
    np.savetxt(destination_folder_ponderar + '\\eddy_pond_smooth.dat', eddy_media, fmt='%.6f')
    np.savetxt(destination_folder_ponderar + '\\x_pond_smooth.dat', x_media, fmt='%.6f')
    np.savetxt(destination_folder_ponderar + '\\y_pond_smooth.dat', y_media, fmt='%.6f')

    H = np.column_stack((x_media, y_media, H_media))
    np.savetxt(destination_folder_ponderar + '\\H.dat', H, fmt='%.6f')

    gradiente_ = np.zeros((1, numeroperfiles))
    
    for j in range(1, numeroperfiles - 2):
        print(j)
        B = np.array(np.loadtxt(destination_folder_ponderar + '\\H.dat')[j, :])
        A = np.array(np.loadtxt(destination_folder_ponderar + '\\H.dat')[j - 1, :])
        C = np.array(np.loadtxt(destination_folder_ponderar + '\\H.dat')[j + 1, :])
        v = 0
        
        dch_x = (B[0] + C[0]) / 2
        dch_y = (B[1] + C[1]) / 2
        h_dch = [B[2], C[2]]
        x_dch = [B[0], C[0]]
    
        if x_dch[0] == x_dch[1]:
            x_dch = [B[v, 1], C[v, 1]]
            hs_dch = np.interp(dch_y, x_dch, h_dch)
        else:
            hs_dch = np.interp(dch_x, x_dch, h_dch)

        izk_x = (B[0] + A[0]) / 2
        izk_y = (B[1] + A[1]) / 2
        h_izk = [B[2], A[2]]
        x_izk = [B[0], A[0]]
    
        if x_izk[0] == x_izk[1]:
            x_izk = [B[1], A[1]]
            hs_izk = np.interp(izk_y, x_izk, h_izk)
        else:
            hs_izk = np.interp(izk_x, x_izk, h_izk)

        HH = hs_izk - hs_dch

        if -0.001 < HH < 0.001:
            HH = 0

        d_x = izk_x - dch_x
        d_y = izk_y - dch_y
        distt = np.sqrt(d_x**2 + d_y**2)

        if distt < 1:
            HH = 0

        diferencia = HH / distt

        gradiente_[v, j] = diferencia

    np.savetxt(destination_folder_ponderar + '\\gradiente.dat', gradiente_, fmt='%.6f')

    FxT = 0
    FyT = 0
    xb = np.loadtxt(destination_folder_ponderar + '\\xb_pond_smooth.dat')
    H = np.loadtxt(destination_folder_ponderar + '\\H_pond_smooth.dat')
    gradd = np.loadtxt(destination_folder_ponderar + '\\gradiente.dat')
    hb = np.loadtxt(destination_folder_ponderar + '\\hh_pond_smooth.dat')
    eddy = np.loadtxt(destination_folder_ponderar + '\\eddy_pond_smooth.dat')
    dir = np.loadtxt(destination_folder_ponderar + '\\dir_pond_smooth.dat')
    dir = np.array(np.abs(dir))
    
    C = np.zeros((numeroperfiles-1, 1))
    A = np.zeros((numeroperfiles-1, 1))
    B_ = np.zeros((numeroperfiles-1, 1))
    B = np.zeros((numeroperfiles-1, 1))
    ang_dif = np.zeros((numeroperfiles-1, 1))
    sn_tub = np.zeros((numeroperfiles-1, 1))
    ang_tot = np.zeros((numeroperfiles-1, 1))
    lc = np.zeros((numeroperfiles-1, 1))

    for i in range(numeroperfiles-1):
        C[i, 0] = (np.pi / cf) * (5 / 32) * g * (m**2) * gamma * (xb[i]**2) * (1 / (np.sqrt(g * hb[i])))
        A[i, 0] = (np.pi / cf) * ((g * m)**(1/2)) * (2 / 3) * (xb[i]**(3/2)) * gradd[i] * K * (((1 / 8) * (gamma**2)) + 1)

        B_[i, 0] = M * (20 * (np.pi**2) * eddy[i]) / (16 * (cf**2) * gamma)
        B[i, 0] = B_[i, 0] * (K * gamma * (gradd[i]**2))

        ang_dif[i, 0] = np.arcsin((A[i, 0] + B[i, 0]) / C[i, 0])
        sn_tub[i, 0] = np.arcsin(A[i, 0] / C[i, 0])

        ang_tot[i, 0] = dir[i] + ang_dif[i, 0]
        lc[i, 0] = ang_tot[i, 0]

    np.savetxt(destination_folder_ponderar + '\\lc.dat', lc, fmt='%.6f')

def readGrd(grdFile, malla, var='Estandar'):
    data = None
    with open(grdFile, 'r') as fid:
        fid.readline()
        xy = list(map(int, fid.readline().split()))
        Ly = list(map(float, fid.readline().split()))
        Lx = list(map(float, fid.readline().split()))
        fid.readline()
        data = np.fromfile(fid, dtype=float, sep=' ').reshape(xy[1], xy[0])

    data = np.rot90(data)
    
    xMalla = np.linspace(Lx[0], Lx[1], xy[1])
    yMalla = np.linspace(Ly[0], Ly[1], xy[0])
    
    xMalla, yMalla = np.meshgrid(xMalla, yMalla)
    
    xMallaRot = xMalla * np.cos(np.deg2rad(malla['angle'])) - yMalla * np.sin(np.deg2rad(malla['angle']))
    yMallaRot = yMalla * np.cos(np.deg2rad(malla['angle'])) + xMalla * np.sin(np.deg2rad(malla['angle']))
    
    xproy = malla['xIni'] + xMallaRot
    yproy = malla['yIni'] + yMallaRot
   
    return data, xMalla, yMalla, xproy, yproy

def surfergriddata(x, y, z, xi, yi, method='linear'):
    points = np.column_stack((x, y))
    grid = np.column_stack((xi, yi))
    zi = griddata(points, z, grid, method=method)
    return zi
        
def InterpolacionRBF_Parametros(parametros, ndireccion, cdireccion, subset_n, PCs_n, Propagaciones):
    resultados = np.zeros((len(PCs_n), parametros))
    PropY = np.zeros(len(Propagaciones))
    
    for j in range(parametros):
        if j in cdireccion:
            
            m = len(Propagaciones)  
            DirX = np.zeros(m)
            DirY = np.zeros(m)
            temp = np.pi / 2 - np.deg2rad(Propagaciones[:, j])
            dd = np.where(temp < -np.pi)
            temp[dd] = 2 * np.pi + temp[dd]
            DirX = np.cos(temp)
            DirY = np.sin(temp)            
            DirX=DirX.reshape(1,-1)        
            DirY=DirY.reshape(1,-1)
            
            coeff_DirX = rbfcreate_modificado(subset_n.T, DirX,'RBFFunction','gaussian')
            coeff_DirX = rbfcreate_modificado(subset_n.T, DirX,'RBFFunction','gaussian')
            DirXp = rbfinterp_modificado(PCs_n.T, coeff_DirX)
            
            coeff_DirY = rbfcreate_modificado(subset_n.T, DirY,'RBFFunction','gaussian')
            DirYp = rbfinterp_modificado(PCs_n.T, coeff_DirY)
            
            Dirp = np.arctan2(DirYp, DirXp) * 180 / np.pi
            resultados[:, j] = Dirp
        else:
            PropY=(Propagaciones[:, j])
            PropY=PropY.reshape(1,-1)
            
            coeff = rbfcreate_modificado(subset_n.T, PropY,'RBFFunction','gaussian')
            resultados[:, j] = rbfinterp_modificado(PCs_n.T, coeff)
   
    return resultados

def rbfcreate_modificado(x, y, *args):
    if len(args) == 0:
        print("x: [ dim by n matrix of coordinates for the nodes ]")
        print("y: [   1 by n vector of values at nodes ]")
        print("RBFFunction: [ gaussian  | thinplate | cubic | multiquadrics | {linear} ]")
        print("RBFConstant: [ positive scalar     ]")
        print("RBFSmooth: [ positive scalar {0} ]")
        print("Stats: [ on | {off} ]")
        return

    Names = ["RBFFunction", "RBFConstant", "RBFSmooth", "Stats"]
    m, n = len(Names), x.shape[1]
    names = [name.lower() for name in Names]

    options = {name: None for name in Names}
    for j in range(m):
       options[(Names[j].strip())]=[]

    nXDim, nXCount = x.shape
    nYDim, nYCount = y.shape

    if nXCount != nYCount:
        raise ValueError("x and y should have the same number of rows")

    if nYDim != 1:
        raise ValueError("y should be n by 1 vector")

    options["x"] = x
    options["y"] = y

    options["RBFFunction"] = "linear"
    options["RBFConstant"] = (np.prod(np.max(x, axis=1) - np.min(x, axis=1)) / nXCount) ** (1 / nXDim)
    options["RBFSmooth"] = 0
    options["Stats"] = "off"

    i = 0
    nargs = len(args)
    expectval = False

    if nargs % 2 != 0:
        raise ValueError("Arguments must occur in name-value pairs.")

    while i < nargs:
        arg = args[i]

        if not expectval:
            if not isinstance(arg, str):
                raise ValueError(f"Expected argument {i+1} to be a string property name.")

            lowArg = arg.lower()
            j = [index for index, name in enumerate(names) if lowArg in name]

            if not j:
                raise ValueError(f"Unrecognized property name '{arg}'")
            elif len(j) > 1:
                k = [index for index, name in j if lowArg == name]

                if len(k) == 1:
                    j = k
                else:
                    msg = f"Ambiguous property name '{arg}' ({(Names[j[0]].strip())}"

                    for idx in j[1:]:
                        msg += f", {(Names[idx].strip())}"

                    msg += ")."
                    raise ValueError(msg)
            j = j[0]
            expectval = True
        else:
            options[(Names[j].strip())] = arg
            expectval = False
        i += 1

    if expectval:
        raise ValueError(f"Expected value for property '{arg}'")

    switch = {
        "linear": rbfphi_linear,
        "cubic": rbfphi_cubic,
        "multiquadric": rbfphi_multiquadrics,
        "thinplate": rbfphi_thinplate,
        "gaussian": rbfphi_gaussian,
    }

    options["rbfphi"] = switch.get(options["RBFFunction"].lower(), rbfphi_linear)

    phi = options["rbfphi"]

    A = rbfAssemble(x, phi, options["RBFConstant"], options["RBFSmooth"])
    b = np.vstack([y.T, np.zeros((nXDim + 1, 1))])
    rbfcoeff = np.linalg.solve(A, b)

    options["rbfcoeff"] = rbfcoeff

    if options["Stats"] == "on":
        print(f"{n} point RBF interpolation was created")
    
    return options 

def rbfinterp_modificado(x, options):
    phi = options['rbfphi']
    rbfconst = options['RBFConstant']
    nodes = options['x']
    rbfcoeff = options['rbfcoeff']

    dim, n = nodes.shape
    dimPoints, nPoints = x.shape

    if dim != dimPoints:
        raise ValueError("x should have the same number of rows as the array used to create the RBF interpolation")

    f = np.zeros(nPoints)
    r = np.zeros(n)

    for i in range(nPoints):
        # print(i)
        s = 0
        r = np.linalg.norm(x[:, i][:, np.newaxis] - nodes, axis=0)
        s = rbfcoeff[n] + np.sum(rbfcoeff[:n] * phi(r, rbfconst))

        for k in range(dim):
            s = s + rbfcoeff[k + n + 1] * x[k, i]

        f[i] = s
        

    return f

def rbfAssemble(x, phi, const, smooth):
    dim, n = x.shape
    A = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i):
            # r = norma(x[:, i-1], x[:, j]) 
            r = np.linalg.norm(x[:, i-1] - x[:, j])
            temp = phi(r, const)
            A[i-1, j] = temp
            A[j-1, i] = temp
        A[i-1, i-1] = A[i-1, i-1] - smooth 
        
    P = np.hstack([np.ones((n, 1)), x.T])
    A = np.vstack([np.hstack([A, P]), np.hstack([P.T, np.zeros((dim + 1, dim + 1))])]) 
    
    return A

def rbfphi_linear(r, const):
    return r

def rbfphi_cubic(r, const):
    return r ** 3

def rbfphi_gaussian(r, const):
    return np.exp(-0.5 * (r ** 2) / (const ** 2))

def rbfphi_multiquadrics(r, const):
    return np.sqrt(1 + (r ** 2) / (const ** 2))

def rbfphi_thinplate(r, const):
    return r ** 2 * np.log(r + 1)