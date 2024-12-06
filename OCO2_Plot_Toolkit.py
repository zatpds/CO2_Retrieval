import numpy as np
from matplotlib import pyplot as plt
import h5py
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from netCDF4 import Dataset


def base_map_gen(lat1, lat2, long1, long2, date, ax):
    # Use Albedo data to generate...
    # ax object passing is necessary, since we always wish to overlay the things...
    try:
        s = np.genfromtxt('/net/mnemo/data/shc/Bulk_Data/Worldwide_Data/Albedo/' + str(int(date / 100)) + '.CSV', delimiter=',')
    except:
        print('Data for the specific month not found. Using default (2017 Jan) data instead.')
        s = np.genfromtxt('/net/mnemo/data/shc/Bulk_Data/Worldwide_Data/Albedo/201701.CSV', delimiter=',')
    # s = np.genfromtxt('E:\Desktop\Jul_16_NewMethodAttempts\Albedo/2017_01_01_Albedo.CSV', delimiter=',')
    long = np.delete(s[0], 0, 0)
    lat = np.delete(s[:, 0], 0, 0)
    data = np.delete(s, 0, 0)
    data = np.delete(data, 0, 1)
    data = np.array(data)

    idx1 = np.logical_and((lat > lat1 - 0.2), (lat < lat2 + 0.2))
    # The interpolation from edge losses one or two grids of information.
    # The information would be safely compensated by 0.2, twice of the grid size.
    tmp = list(filter(lambda i: idx1[i], range(len(idx1))))
    idx1_1 = min(tmp)
    idx1_2 = max(tmp)
    idx2 = np.logical_and((long > long1 - 0.3), (long < long2 + 0.3))
    tmp = list(filter(lambda i: idx2[i], range(len(idx2))))
    idx2_1 = min(tmp)
    idx2_2 = max(tmp)

    idx = np.logical_or(data < 0, data > 1) # Removing the filling values
    data[idx] = 0
    fig, ax = plt.subplots()
    ax.contourf(long[idx2_1:idx2_2], lat[idx1_1:idx1_2], data[idx1_1:idx1_2, idx2_1:idx2_2], cmap='gray')
    return ax


def mark_cities(lats, longs, weights, ax):
    if weights==[]:
        s = np.ones(len(lats)) * 10
    else:
        s = (weights - min(weights)) / (max(weights) - min(weights)) * 10
    for i in range(len(lats)):
        ax.plot([longs[i]], [lats[i]], marker='o', markersize=s[i], color='red')
    return ax


def track_gen_oco(lat1, lat2, l2file, ax, option):
    # Takes the standard level 2 filename for the plot.
    # It gets the reading itself,since the vertex structures would be lost after the L1B Interpolation
    # options: AOD, XCO2
    f = h5py.File(l2file, 'r')
    lat = f["RetrievalGeometry/retrieval_vertex_latitude"]
    lats1 = np.array(lat[:, 1, 0])
    idx = np.logical_and(lats1>lat1, lats1<lat2)
    lats1 = np.array(lat[idx, 1, 0])
    lats2 = np.array(lat[idx, 1, 1])
    lats3 = np.array(lat[idx, 1, 2])
    lats4 = np.array(lat[idx, 1, 3])
    long = f["RetrievalGeometry/retrieval_vertex_longitude"]
    long1 = np.array(long[idx, 1, 0])
    long2 = np.array(long[idx, 1, 1])
    long3 = np.array(long[idx, 1, 2])
    long4 = np.array(long[idx, 1, 3])

    if option=='AOD':
        AOD = f["AerosolResults/aerosol_total_aod"]
        data = np.array(AOD[idx])
    else:
        XCO2 = f["RetrievalResults/xco2"]
        data = np.array(XCO2[idx])
    p = []
    for i in range(len(lats1)):
        polygon = Polygon([(long1[i], lats1[i]), (long2[i], lats2[i]), (long3[i], lats3[i]), (long4[i], lats4[i])])
        p.append(polygon)
    p = PatchCollection(p, cmap='seismic', alpha=0.3)
    colors = np.array(data)
    p.set_array(colors)
    ax.add_collection(p)
    return ax


def track_gen_oco_lite(lat1, lat2, long1, long2, l2_lite_file, ax):
    # Takes the standard level 2 filename for the plot.
    # It gets the reading itself,since the vertex structures would be lost after the L1B Interpolation
    # No option, only XCO2
    f = Dataset(l2_lite_file)
    lat = f["vertex_latitude"]
    lats = lat[:, 0]
    long = f["vertex_longitude"]
    longs = long[:, 0]
    idx = np.logical_and(np.logical_and(lats>lat1, lats<lat2), np.logical_and(longs>long1, longs<long2))
    lats1 = np.array(lat[idx, 0])
    lats2 = np.array(lat[idx, 1])
    lats3 = np.array(lat[idx, 2])
    lats4 = np.array(lat[idx, 3])
    long1 = np.array(long[idx, 0])
    long2 = np.array(long[idx, 1])
    long3 = np.array(long[idx, 2])
    long4 = np.array(long[idx, 3])

    xco2 = f["xco2"]
    data = np.array(xco2)
    data = data[idx]
    p = []
    for i in range(len(lats1)):
        polygon = Polygon([(long1[i], lats1[i]), (long2[i], lats2[i]), (long3[i], lats3[i]), (long4[i], lats4[i])])
        p.append(polygon)
    p = PatchCollection(p, cmap='seismic', alpha=0.3)
    colors = np.array(data)
    p.set_array(colors)
    ax.add_collection(p)
    return ax

