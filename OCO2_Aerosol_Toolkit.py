'''
This generalizes the utilities from the ocean case, and also adds more for the land case.
NOTE: There are some pieces of code serving the purpose of the debugging and saves some data to my directory, which
means you don't have the permission... Kindly search for "np.savetxt" and try to change it.
REQUIREMENTS: scipy version newer than 0.17.0
numpy version up to date
'''
import h5py
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d, interp2d
from netCDF4 import Dataset
import glob
import os


def dirnew(dirn):
    try:
        os.makedirs(dirn)
    except:
        print('Directory %s already exists. Proceeding without creating new directory...'%dirn)

def l1b_read_area(filename, lat1, lat2,lon1,lon2):
    # This is for the old l1b files that has different fields
    # Takes the L1b File name, returns the STANDARD L1B dict file.
    # Adapted from Zhao's MATLAB codes...
    # Taking only channel 2, indexed as 1
    # Taking only data between latitude 1 and latitude 2
    f = h5py.File(filename, 'r')
    # 6438 * 8
    latitudes = f['SoundingGeometry/sounding_latitude']
    latitudes = latitudes[:, 1]
    longitudes = f['SoundingGeometry/sounding_longitude']
    longitudes = longitudes[:, 1]
    idx = np.logical_and(np.logical_and((latitudes>lat1), (latitudes<lat2)),np.logical_and((longitudes>lon1), (longitudes<lon2)))
    sid_lists = f['SoundingGeometry/sounding_id']
    sid_lists = sid_lists[:, 1]
    land_frac = f['SoundingGeometry/sounding_land_fraction']
    land_frac = land_frac[:, 1]
    # 6438 * 8 *1016
    ls = []
    Rad_1 = f['SoundingMeasurements/radiance_o2']
    Rad_1 = Rad_1[:, 1 , :]
    ls.append(len(Rad_1[0, :]))
    Rad_2 = f['SoundingMeasurements/radiance_strong_co2']
    Rad_2 = Rad_2[:, 1, :]
    ls.append(len(Rad_2[0, :]))
    Rad_3 = f['SoundingMeasurements/radiance_weak_co2']
    Rad_3 = Rad_3[:, 1, :]
    ls.append(len(Rad_3[0, :]))
    zenith = f['SoundingGeometry/sounding_solar_zenith']
    zenith = zenith[:, 1]
    zenith = np.deg2rad(zenith)
    zenith = np.cos(zenith) # So only the cosine value is carried over...
    DCS = f['InstrumentHeader/dispersion_coef_samp']
    surf_rough = f['SoundingGeometry/sounding_surface_roughness']
    surf_rough = surf_rough[:, 1]
    cont1 = f['SoundingMeasurements/rad_continuum_o2']
    cont1 = cont1[:, 1]
    cont2 = f['SoundingMeasurements/rad_continuum_sco2']
    cont2 = cont2[:, 1]
    cont3 = f['SoundingMeasurements/rad_continuum_wco2']
    cont3 = cont3[:, 1]
    for i in range(len(Rad_1[:,0])):
        tmp = np.sort(Rad_1[i,:])
        cont1[i] = sum(tmp[-20:-1]) / 19
        tmp = np.sort(Rad_2[i,:])
        cont2[i] = sum(tmp[-20:-1]) / 19
        tmp = np.sort(Rad_3[i,:])
        cont3[i] = sum(tmp[-20:-1]) / 19
    wls = []
    for j in range(3):
        tmp = range(ls[j])
        tmp = np.array(tmp)
        tmp = tmp + 1
        dband = DCS[j, 1, :]
        lambdas = np.zeros(ls[j])
        for ll in range(len(dband)):
            lambdas = np.add(lambdas, np.multiply(np.power(tmp, ll), dband[ll]))
        wls.append(lambdas)

    L1BDict = {
        "Latitude": latitudes[idx],
        "Longitude": longitudes[idx],
        "Sounding_ID": sid_lists[idx],
        "O2A_Radiance": Rad_1[idx, :],
        "CO2S_Radiance": Rad_2[idx, :],
        "CO2W_Radiance": Rad_3[idx, :],
        "O2A_Wavelength": wls[0],
        "CO2W_Wavelength": wls[1],
        "CO2S_Wavelength": wls[2],
        "Land_Fraction": land_frac[idx],
        "Zenith" : zenith[idx],
        "Roughness": surf_rough[idx],
        "O2A_Cont": cont1[idx],
        "CO2S_Cont": cont2[idx],
        "CO2W_Cont": cont3[idx]
    }
    return L1BDict


def l1b_read(filename, lat1, lat2):
    # Takes the L1b File name, returns the STANDARD L1B dict file.
    # Adapted from Zhao's MATLAB codes...
    # Taking only channel 2, indexed as 1
    # Taking only data between latitude 1 and latitude 2
    f = h5py.File(filename, 'r')
    # 6438 * 8
    latitudes = f['SoundingGeometry/sounding_latitude']
    latitudes = latitudes[:, 1]
    idx = np.logical_and((latitudes>lat1), (latitudes<lat2))
    longitudes = f['SoundingGeometry/sounding_longitude']
    longitudes = longitudes[:, 1]
    sid_lists = f['SoundingGeometry/sounding_id']
    sid_lists = sid_lists[:, 1]
    land_frac = f['SoundingGeometry/sounding_land_fraction']
    land_frac = land_frac[:, 1]
    # 6438 * 8 *1016
    ls = []
    Rad_1 = f['SoundingMeasurements/radiance_o2']
    Rad_1 = Rad_1[:, 1 , :]
    ls.append(len(Rad_1[0, :]))
    Rad_2 = f['SoundingMeasurements/radiance_strong_co2']
    Rad_2 = Rad_2[:, 1, :]
    ls.append(len(Rad_2[0, :]))
    Rad_3 = f['SoundingMeasurements/radiance_weak_co2']
    Rad_3 = Rad_3[:, 1, :]
    ls.append(len(Rad_3[0, :]))
    zenith = f['SoundingGeometry/sounding_solar_zenith']
    zenith = zenith[:, 1]
    zenith = np.deg2rad(zenith)
    zenith = np.cos(zenith) # So only the cosine value is carried over...
    DCS = f['InstrumentHeader/dispersion_coef_samp']
    surf_rough = f['SoundingGeometry/sounding_surface_roughness']
    surf_rough = surf_rough[:, 1]
    cont1 = f['SoundingMeasurements/rad_continuum_o2']
    cont1 = cont1[:, 1]
    cont2 = f['SoundingMeasurements/rad_continuum_strong_co2']
    cont2 = cont2[:, 1]
    cont3 = f['SoundingMeasurements/rad_continuum_weak_co2']
    cont3 = cont3[:, 1]
    idxes = np.array(range(len(cont3)))
    for i in range(len(Rad_1[:,0])):
        tmp = np.sort(Rad_1[i,:])
        cont1[i] = sum(tmp[-20:-1]) / 19
        tmp = np.sort(Rad_2[i,:])
        cont2[i] = sum(tmp[-20:-1]) / 19
        tmp = np.sort(Rad_3[i,:])
        cont3[i] = sum(tmp[-20:-1]) / 19
    wls = []
    for j in range(3):
        tmp = range(ls[j])
        tmp = np.array(tmp)
        tmp = tmp + 1
        dband = DCS[j, 1, :]
        lambdas = np.zeros(ls[j])
        for ll in range(len(dband)):
            lambdas = np.add(lambdas, np.multiply(np.power(tmp, ll), dband[ll]))
        wls.append(lambdas)

    L1BDict = {
        "Latitude": latitudes[idx],
        "Longitude": longitudes[idx],
        "Sounding_ID": sid_lists[idx],
        "O2A_Radiance": Rad_1[idx, :],
        "CO2S_Radiance": Rad_2[idx, :],
        "CO2W_Radiance": Rad_3[idx, :],
        "O2A_Wavelength": wls[0],
        "CO2W_Wavelength": wls[1],
        "CO2S_Wavelength": wls[2],
        "Land_Fraction": land_frac[idx],
        "Zenith" : zenith[idx],
        "Roughness": surf_rough[idx],
        "O2A_Cont": cont1[idx],
        "CO2S_Cont": cont2[idx],
        "CO2W_Cont": cont3[idx],
        "Indices": idxes[idx]
    }
    return L1BDict


def l1b_read_v10r(filename, lat1, lat2):
    # Takes the L1b File name, returns the STANDARD L1B dict file.
    # Adapted from Zhao's MATLAB codes...
    # Taking only channel 2, indexed as 1
    # Taking only data between latitude 1 and latitude 2
    f = h5py.File(filename, 'r')
    # 6438 * 8
    latitudes = f['SoundingGeometry/sounding_latitude']
    latitudes = latitudes[:, 1]
    idx = np.logical_and((latitudes>lat1), (latitudes<lat2))
    longitudes = f['SoundingGeometry/sounding_longitude']
    longitudes = longitudes[:, 1]
    sid_lists = f['SoundingGeometry/sounding_id']
    sid_lists = sid_lists[:, 1]
    land_frac = f['SoundingGeometry/sounding_land_fraction']
    land_frac = land_frac[:, 1]
    # 6438 * 8 *1016
    ls = []
    Rad_1 = f['SoundingMeasurements/radiance_o2']
    Rad_1 = Rad_1[:, 1 , :]
    ls.append(len(Rad_1[0, :]))
    Rad_2 = f['SoundingMeasurements/radiance_strong_co2']
    Rad_2 = Rad_2[:, 1, :]
    ls.append(len(Rad_2[0, :]))
    Rad_3 = f['SoundingMeasurements/radiance_weak_co2']
    Rad_3 = Rad_3[:, 1, :]
    ls.append(len(Rad_3[0, :]))
    zenith = f['SoundingGeometry/sounding_solar_zenith']
    zenith = zenith[:, 1]
    zenith = np.deg2rad(zenith)
    zenith = np.cos(zenith) # So only the cosine value is carried over...
    DCS = f['InstrumentHeader/dispersion_coef_samp']
    surf_rough = f['SoundingGeometry/sounding_surface_roughness']
    surf_rough = surf_rough[:, 1]
    cont1 = f['SoundingMeasurements/rad_continuum_o2']
    cont1 = cont1[:, 1]
    idxes = np.array(range(len(cont1)))
    for i in range(len(Rad_1[:,0])):
        tmp = np.sort(Rad_1[i,:])
        cont1[i] = sum(tmp[-20:-1]) / 19
    wls = []
    for j in range(3):
        tmp = range(ls[j])
        tmp = np.array(tmp)
        tmp = tmp + 1
        dband = DCS[j, 1, :]
        lambdas = np.zeros(ls[j])
        for ll in range(len(dband)):
            lambdas = np.add(lambdas, np.multiply(np.power(tmp, ll), dband[ll]))
        wls.append(lambdas)

    L1BDict = {
        "Latitude": latitudes[idx],
        "Longitude": longitudes[idx],
        "Sounding_ID": sid_lists[idx],
        "O2A_Radiance": Rad_1[idx, :],
        "CO2S_Radiance": Rad_2[idx, :],
        "CO2W_Radiance": Rad_3[idx, :],
        "O2A_Wavelength": wls[0],
        "CO2W_Wavelength": wls[1],
        "CO2S_Wavelength": wls[2],
        "Land_Fraction": land_frac[idx],
        "Zenith" : zenith[idx],
        "Roughness": surf_rough[idx],
        "O2A_Cont": cont1[idx],
        "Indices": idxes[idx]
    }
    return L1BDict


def cal_read(filename):
    # Takes the CAL .mat file name, returns the STANDARD CALIPSO dict file.
    # Adapted from Zhao and mine MATLAB codes...
    nc = loadmat(filename)
    CAL_lat = nc['CALIPSOLatitude']
    CAL_lat = CAL_lat[:, 1]
    CAL_AOD = nc['AOD_760']
    CAL_long = nc['CALIPSOLongitude']
    CAL_long = CAL_long[:, 1]
    CAL_ALH = nc['EffectiveALH']
    CAL_AOD[np.isnan(CAL_ALH)] = np.nan
    Surf_type = nc['IGBP_Surface_Type']
    Cloud = nc['Column_Optical_Depth_Cloud_532']
    CALDict = {
        "AOD": CAL_AOD,
        "Latitude": CAL_lat,
        "Longitude": CAL_long,
        "ALH": CAL_ALH,
        "Surf_Type": Surf_type,
        "Cloud": Cloud
    }
    for key in CALDict:
        if CALDict[key].ndim==2:
            CALDict[key] = CALDict[key].reshape(len(CALDict[key]),)
    return CALDict


def cal_concatenate(caldict1, caldict2):
    # concatenate two caldicts. Used due to the fact that sometimes two CALIPSO date are matched with one OCO track.
    caldict = caldict1
    for key in caldict1:
        tmp = caldict1[key]
        tmp2 = caldict2[key]
        for i in range(len(tmp2)):
            tmp = np.append(tmp, tmp2[i])
        caldict[key] = tmp
    return caldict

def cont_level_eval(L1bDict, sid_idx, Band):
    # Use the mean of channels 850~900 to evaluate the continuum level... Taking the OCO2 data!
    # Convention: Band takes 0, 1, and 2
    if Band==0:
        spectrum = L1bDict['O2A_Radiance'][sid_idx, :]
    else:
        if Band==1:
            spectrum = L1bDict['CO2W_Radiance'][sid_idx, :]
        else:
            spectrum = L1bDict['CO2S_Radiance'][sid_idx, :]
    spectrum = spectrum / 1.0e20
    sum_a = np.sum(np.power(spectrum, 5))
    sum_w = np.sum(np.power(spectrum, 4))
    return sum_a / sum_w


def lat_interp_to_l1b(dict, target_latitudes):
    origin_latitudes = dict["Latitude"]
    for key in dict:
        if key=="Latitude":
            dict[key] = target_latitudes
            continue
        tmp = dict[key]
        mdl = interp1d(origin_latitudes, tmp, kind='linear', fill_value="extrapolate")
        dict[key] = mdl(target_latitudes)
    return dict


def mid_level_eval(L1bDict, sid_idx, Band):
    # Self-sorted should not be too different from clear-sky-sorted.
    if Band==0:
        spectrum = L1bDict['O2A_Radiance'][sid_idx, :]
    else:
        if Band==1:
            spectrum = L1bDict['CO2W_Radiance'][sid_idx, :]
        else:
            spectrum = L1bDict['CO2S_Radiance'][sid_idx, :]
    s_tmp = sorted(spectrum)
    s_tmp = np.array(s_tmp)
    s_tmp = s_tmp / np.mean(s_tmp[range(len(s_tmp) - 50, len(s_tmp) - 1)])
    concern_range = [
        [150, 250],
        [150, 250],
        [150, 250]
    ] # The range that characterizing! For CO2 bands this awaits exploration...
    concern_range = np.array(concern_range)
    res = np.mean(s_tmp[concern_range[Band, 0]:concern_range[Band, 1]])
    if res>0.8:
        np.savetxt('/net/mnemo/data/shc/tmp/Anomal_M.csv', s_tmp, delimiter=',')
        np.savetxt('/net/mnemo/data/shc/tmp/spec.csv', spectrum, delimiter=',')
    return res


def mid_level_eval_v2(L1bDict, sid_idx, Band, concern_range):
    # Self-sorted should not be too different from clear-sky-sorted.
    if Band==0:
        spectrum = L1bDict['O2A_Radiance'][sid_idx, :]
    else:
        if Band==1:
            spectrum = L1bDict['CO2W_Radiance'][sid_idx, :]
        else:
            spectrum = L1bDict['CO2S_Radiance'][sid_idx, :]
    s_tmp = sorted(spectrum)
    s_tmp = np.array(s_tmp)
    s_tmp = s_tmp / np.mean(s_tmp[range(len(s_tmp) - 50, len(s_tmp) - 1)])
    concern_range = np.array(concern_range)
    res = np.mean(s_tmp[concern_range[0]:concern_range[1]])
    return res


def remv_invalid_soundings(l1bdict):
    print('Preprocessing: L1B Quality Filtering')
    threshold = 0 # min value < this would be the invalid...
    spectrum = l1bdict['O2A_Radiance'] # Validity should be the same for all three bands
    i = 0
    while (i<len(spectrum[:,0])):
        if min(spectrum[i, :])<threshold:
            for key in l1bdict:
                if key[len(key)-10:len(key)]!='Wavelength':
                    l1bdict[key] = np.delete(l1bdict[key], i, 0)
            spectrum = np.delete(spectrum, i, 0)
        else:
            i = i + 1
    return l1bdict


def remv_ocean_soundings(l1bdict):
    print('Preprocessing: L1B Ocean Filtering')
    threshold = 95 # min value < this would be considered as the ocean
    Land_Frac = l1bdict['Land_Fraction'] # Land fraction is a 1D ary
    i = 0
    while (i<len(Land_Frac)):
        if Land_Frac[i]<threshold:
            for key in l1bdict:
                if key[len(key) - 10:len(key)] != 'Wavelength':
                    l1bdict[key] = np.delete(l1bdict[key], i, 0)
            Land_Frac = np.delete(Land_Frac, i, 0)
        else:
            i = i + 1
    return l1bdict


def albedo_construc(date):
    print('Preprocess: Reading and constructing interpolated albedo data...')
    # Use the latitude and longitude in l1b dictionary, to construct information interpolated from albedo csv files.
    # Returning the albedo values along the track. If any fill values, jump over and interpolate later...
    try:
        if np.mod(int(date / 100), 100)<10:
            k = '0'
        else:
            k = ''
        s = np.genfromtxt(
            '/net/mnemo/data/shc/Bulk_Data/For_Antong/Albedo/2016' + k + str(np.mod(int(date / 100), 100)) + '.CSV',
            delimiter=',')
    except:
        print('Data for the May not found. Using default alternative (June) data instead.')
        s = np.genfromtxt('/net/mnemo/data/shc/Bulk_Data/For_Antong/Albedo/201606.CSV', delimiter=',')
    # We are going to fix the used albedo for the Saudi Arabia case...
    long = np.delete(s[0], 0, 0)
    lat = np.delete(s[:, 0], 0, 0)
    data = np.delete(s, 0, 0)
    data = np.delete(data, 0, 1)
    idx = np.logical_or(data < 0, data > 1)
    data[idx] = 0
    mdl = interp2d(long, lat, data, kind='linear', fill_value=0)
    return mdl


def albedo_interp(l1bdict, mdl):
    print('Other: Interpolating the albedo data to the current track...')
    lat = l1bdict['Latitude']
    long = l1bdict['Longitude']
    alb = []
    for i in range(len(lat)):
        alb.append(float(mdl(long[i], lat[i]))) # mdl by default returns an array
    alb = np.reshape(alb, (len(alb), ))
    return alb


def l2_read(filename):
    # Returns the STANDARD l2 Dictionary!
    # We keep the same latitude, as by right the L2 results should be the same as L1b in geometry.
    f = h5py.File(filename, 'r')
    lat = f["RetrievalGeometry/retrieval_latitude"]
    lat = np.array(lat)
    long = f["RetrievalGeometry/retrieval_longitude"]
    long = np.array(long)
    AOD = f["AerosolResults/aerosol_total_aod"]
    AOD = np.array(AOD)
    XCO2 = f["RetrievalResults/xco2"]
    XCO2 = np.array(XCO2)
    '''
    o2_albedo = f["AlbedoResults/albedo_o2_fph"]
    o2_albedo_slope = f["AlbedoResults/albedo_slope_o2"]
    co2w_albedo = f["AlbedoResults/albedo_weak_co2_fph"]
    co2w_albedo_slope = f["AlbedoResults/albedo_slope_weak_co2"]
    co2s_albedo = f["AlbedoResults/albedo_strong_co2_fph"]
    co2s_albedo_slope = f["AlbedoResults/albedo_slope_strong_co2"]
    '''
    l2dict = {
        "XCO2": XCO2,
        "AOD" : AOD,
        "Latitude" : lat,
        "Longitude" : long
        #"O2_Albedo" : o2_albedo,
        #"O2_Albedo_Slope" : o2_albedo_slope,
        #"CO2_Strong_Albedo": co2s_albedo,
        #"CO2_Strong_Albedo_Slope": co2s_albedo_slope,
        #"CO2_Weak_Albedo": co2w_albedo,
        #"CO2_Weak_Albedo_Slope": co2w_albedo_slope
    }
    return l2dict


def l2_lite_read(filname, l1bdict):
    # Longitude must be within the limit, since this contains all tracks of a day...
    # This completes the whole process of reading and interpolating.
    # Returns an array only, with a list of XCO2 data
    data = Dataset(filname)
    long = data['longitude']
    lat = data['latitude']
    lat1 = min(l1bdict["Latitude"])
    lat2 = max(l1bdict["Latitude"])
    long1 = min(l1bdict["Longitude"])
    long2 = max(l1bdict["Longitude"])
    xco2 = data['xco2']
    idx = np.logical_and(np.logical_and(lat>=lat1, lat<=lat2), np.logical_and(long>=long1, long<=long2))
    lat = lat[idx]
    xco2 = xco2[idx]
    mdl = interp1d(lat, xco2, kind='linear', fill_value="extrapolate")
    return mdl(l1bdict["Latitude"])


def l2_lite_obtain(date):
    import wget
    l2litenames = '/net/mnemo/data/shc/Bulk_Data/OCO_LITE/OCO2LtCO2v9-156411244753.txt'
    toget = 0
    for line in open(l2litenames, 'r'):
        if str(date)[2:9] in line:
            toget = line
            break
    if toget==0:
        raise Exception('ERROR: No mathced L2_LITE file for the date')
    filename = wget.download(toget, out='/net/mnemo/data/shc/Bulk_Data/OCO_LITE/')
    return filename


def SigBin(series, numsig):
    # Takes the time series and the number of sigmas for marking the difference
    # Output is the series of data indexed with 1, 2, or 3. (high,mid,low)
    # An optimization of the number of sigmas chosen is included, which slows down the process.
    # The optimizer is proven to be not taking too much time.....
    half_window_size = int(len(series)/20)
    out_ind = np.zeros(len(series))
    mline = []
    for i in range(len(series)): # Find out the median for comparison
        ref_level = np.median(series[max(0, i - half_window_size):min(len(series), i + half_window_size)])
        mline.append(ref_level)
    series_new = np.array(series) - np.array(mline) # Detrending: avoid influence from rapidly varying overall trend.
    sss = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    min_score = 999
    for ss in sss:
        for i in range(len(series)):
            ct1 = 0
            ct2 = 0
            ct3 = 0
            ref_level = mline[i]
            ref_sig = np.std(series_new[max(0, i - half_window_size):min(len(series), i + half_window_size)])
            ref_sig = ref_sig * ss
            if series[i]>=ref_level + numsig * ref_sig:
                out_ind[i] = 1
                ct1+=1
            else:
                if series[i] <= ref_level - numsig * ref_sig:
                    out_ind[i] = 3
                    ct3+=1
                else:
                    out_ind[i] = 2
                    ct2+=1
        if np.std([ct1, ct2, ct3])<min_score: # Optimization metric: the standard deviation of the three counts.
            min_score = np.std([ct1, ct2, ct3])
            sig_st = ss
    hline = []
    lline = []
    for i in range(len(series)):
        ref_level = mline[i]
        ref_sig = np.std(series_new[max(0, i - half_window_size):min(len(series), i + half_window_size)])
        ref_sig = ref_sig * sig_st
        if series[i] >= ref_level + numsig * ref_sig:
            out_ind[i] = 1
        else:
            if series[i] <= ref_level - numsig * ref_sig:
                out_ind[i] = 3
            else:
                out_ind[i] = 2
        hline.append(ref_level + numsig * ref_sig)
        lline.append(ref_level - numsig * ref_sig)
    return out_ind


def diff_spectrum(l1bdict, low_idx, Band):
    if Band==0:
        spectrum = l1bdict['O2A_Radiance']
    else:
        if Band==1:
            spectrum = l1bdict['CO2W_Radiance']
        else:
            spectrum = l1bdict['CO2S_Radiance']
    ref_low = spectrum[low_idx, :]
    ref_low = sorted(ref_low)
    ref_low = np.array(ref_low)
    ref_low = ref_low / np.mean(ref_low[range(len(ref_low) - 50, len(ref_low) - 1)])
    results = []
    for i in range(len(spectrum)):
        tmp = spectrum[i, :]
        tmp = sorted(tmp)
        tmp = np.array(tmp)
        tmp = tmp / np.mean(tmp[range(len(tmp) - 50, len(tmp) - 1)])
        results.append(tmp - ref_low)
    np.savetxt('/net/mnemo/data/shc/tmp/diffspectrums.csv', results, delimiter=',')
    return np.array(results)


def diff_spectrum_2(l1bdict, refdict, low_idx, Band):
    # Taking a different date...
    if Band==0:
        spectrum = l1bdict['O2A_Radiance']
        sref = refdict['O2A_Radiance']
    else:
        if Band==1:
            spectrum = l1bdict['CO2W_Radiance']
            sref = refdict['CO2W_Radiance']
        else:
            spectrum = l1bdict['CO2S_Radiance']
            sref = refdict['CO2S_Radiance']
    ref_low = sref[low_idx, :]
    ref_low = sorted(ref_low)
    ref_low = np.array(ref_low)
    ref_low = ref_low / np.mean(ref_low[range(len(ref_low) - 50, len(ref_low) - 1)])
    results = []
    for i in range(len(spectrum)):
        tmp = spectrum[i, :]
        tmp = sorted(tmp)
        tmp = np.array(tmp)
        tmp = tmp / np.mean(tmp[range(len(tmp) - 50, len(tmp) - 1)])
        results.append(tmp - ref_low)
    np.savetxt('/net/mnemo/data/shc/tmp/diffspectrums.csv', results, delimiter=',')
    return np.array(results)


def sigbin_levels_eval(diffs, idx, band, indices):
    concern_range = [
        [150, 250],
        [150, 250],
        [150, 250]
    ]
    concern_range = np.array(concern_range)
    tmp = diffs[idx, concern_range[band, 0]:concern_range[band, 1]]
    indices = indices[concern_range[band, 0]:concern_range[band, 1]]
    a = np.mean(tmp[indices==1])
    b = np.mean(tmp[indices == 2])
    c = np.mean(tmp[indices == 3])
    return a, b, c


def add_albedo_l1b(l1bdict, date):
    # Takes the l1b dictionary, and the albedo term is added into the l1b dictionary
    # The albedo information is taken from the albedo file and interpolated to the l1b track.
    # Albedo information must be pre-processed by the MATLAB utility
    filn = '/net/mnemo/data/shc/OCO_codes/MATLAB_Utilities/Surface_Albedo/' + str(date) + '.mat'
    nc = loadmat(filn)
    mod_lat = nc['lats']
    mod_long = nc['longs']
    o2 = nc['o2']
    co2s = nc['co2s']
    lats = l1bdict["Latitude"]
    longs = l1bdict["Longitude"]
    mdl_o2 = interp2d(mod_long, mod_lat, o2, kind='linear', fill_value=0)
    mdl_co2 = interp2d(mod_long, mod_lat, co2s, kind='linear', fill_value=0)
    o2 = []
    co2 = []
    for i in range(len(lats)):
        o2.append(float(mdl_o2(longs[i], lats[i])))
        co2.append(float(mdl_co2(longs[i], lats[i])))
    l1bdict.update({'O2_Albedo' : o2,
                    'CO2S_Albedo' : co2})
    return l1bdict



def search_file(date, dirn, mode):
    # date is a NUMBER for searching. dirn is the string of directory name.
    # mode: 'L1b', 'CALIPSO', 'IDP', 'Met', 'L2', 'mat'...
    # Actually the header is not needed...
    if mode=='Met':
        header = 'oco2_L2Met'
    else:
        if mode=='L2':
            header = 'oco2_L2Std'
        else:
            if mode=='L1b':
                header = 'oco2_L1bSc'
            else:
                if mode=='CALIPSO':
                    header = 'CAL_LID_L2_05kmAPro'
                else:
                    if mode=='IDP':
                        header = 'oco2_L2IDP'
                    else:
                        header = ''
    s = glob.glob(dirn + '/' + header + '*' + str(date)[2:8] + '*')
    return s[0]


def get_sid_for_lat(l1bdict, lat):
    latitude = l1bdict['Latitude']
    sid_list = l1bdict['Sounding_ID']
    # Look up the sounding id in the id list...
    # Use instrument #2 only, indexed as 1
    # do it when change direction...
    for i in range(1, len(sid_list)):
        if (latitude[i] - lat) * (latitude[i - 1] - lat) < 0.0:
            return str(sid_list[i])


def write_txt(rad, wl, filn):
    # write to txt file with collated rad and wl
    with open(filn + '.txt', 'w') as f:
        for i in range(len(rad)):
            f.write('%.5f'%wl[i] + ' ' * 8 +  '%d\n'%int(rad[i]))