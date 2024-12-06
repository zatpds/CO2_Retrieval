import l2_afp
import OCO2_Aerosol_Toolkit
import OCO2_Solution_Toolkit
import numpy as np
import multiprocessing as mp

manip = 1

appen = '_Error_Contained'

def parallel_process_unit(aod, alh, l1bFile, metFile, IDPFile, L1bDict, date, n, ntot):
    sol_rec = []
    flags = []
    nums = []
    if n==ntot - 1:
        endnum = len(aod)
    else:
        endnum = (n + 1) * int(len(aod) / ntot)
    for ii in range(n * int(len(aod) / ntot), endnum):
        flag = 1

        if aod[ii] < 0:
            aod[ii] = 0.001
        if alh[ii] < 0:
            alh[ii] = 0.5
        l2_obj = OCO2_Solution_Toolkit.construc_constrained_l2_obj(l1bFile, metFile, IDPFile,
                                                                   str(L1bDict["Sounding_ID"][ii]))
        if manip==1:
            l2_obj = OCO2_Solution_Toolkit.constrain_alh(l2_obj, np.exp(-alh[ii] / 8))
            l2_obj = OCO2_Solution_Toolkit.constrain_aod(l2_obj, np.log(aod[ii]))
        if manip==2:
            l2_obj = OCO2_Solution_Toolkit.constrain_alh(l2_obj, np.exp(-alh[ii] / 8))
        try:
            Kupdate = l2_obj.Kupdate
            x, _, _ = OCO2_Solution_Toolkit.l2_obj_solve(l2_obj, Kupdate)
            sol_rec.append(x[-1])
            print('Done for sounding ID ' + str(L1bDict["Sounding_ID"][ii]))
        except:
            print(str(L1bDict["Sounding_ID"][ii]) + 'does not yield valid retrieval.')
            flag = 0
        flags.append(flag)
        nums.append(ii)
    np.savez('./' + str(date) + appen + '/' + 'constrained_sol' + str(n) + '.npz', L1bDict["Sounding_ID"],
             L1bDict["Latitude"], L1bDict["Longitude"], sol_rec, aod, alh, flags, nums)


place = 'Sasan'

if place=='Riyadh':
    dates = [20161029, 20160826, 20160522, 20160319, 20160115]
if place=='NewDelhi':
    dates = [20151024, 20150904] # 20160110, 20160103,
if place=='Sasan':
    dates = [20141023]
if place=='Cal':
    dates = [20171008, 20171015, 20171022, 20180112, 20180425, 20180621, 20180628]
    # Dates where no valid AOD data available are peeled off!
    #
if place=='LA':
    dates = [20180808, 20180603, 20171125, 20171006] # Target mode on the city!

for date in dates:
    OCO2_Aerosol_Toolkit.dirnew(str(date) + appen)
    print('Doing for date:%d' % date)
    mdl = OCO2_Aerosol_Toolkit.albedo_construc(date)
    if place=='Riyadh':
        dirn = './Results_Riyadh/NN/'
        l1bFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/Riyadh/OCO/ND/L1b', 'L1b')
        l2File = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/Riyadh/OCO/ND/L2', 'L2')
        matFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/OCO_codes/MATLAB_Utilities/Riyadh',
                                                   'mat')
        metFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/Riyadh/OCO/ND/Met', 'Met')
        IDPFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/Riyadh/OCO/ND/IDP', 'IDP')
        lat1 = 20
        lat2 = 30
    if place=='NewDelhi':
        dirn = './Results_NewDelhi/NN/'
        l1bFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/New_Delhi/OCO_Data/ND/L1b', 'L1b')
        l2File = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/New_Delhi/OCO_Data/ND/L2', 'L2')
        matFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/OCO_codes/MATLAB_Utilities/New_Delhi', 'mat')
        metFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/New_Delhi/OCO_Data/ND/Met', 'Met')
        IDPFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/New_Delhi/OCO_Data/ND/IDP', 'IDP')
        lat1 = 20
        lat2 = 30
    if place=='Sasan':
        dirn = './Results_Sasan/NN/'
        l1bFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/Plume_Cases/Sasan/OCO/L1b',
                                                   'L1b')
        l2File = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/Plume_Cases/Sasan/OCO/L2', 'L2')
        metFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/Plume_Cases/Sasan/OCO/Met', 'Met')
        IDPFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/Plume_Cases/Sasan/OCO/IDP', 'IDP')
        matFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/OCO_codes/MATLAB_Utilities/Plume_Sasan',
                                                   'mat')
        lat1 = 20
        lat2 = 30
    if place=='Cal':
        dirn = './California/NN/'
        l1bFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/Los_Angeles/ND', 'L1b')
        l2File = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/Los_Angeles/ND', 'L2')
        matFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/OCO_codes/MATLAB_Utilities/Los_Angeles/ND', 'mat')
        lat1 = 32.7
        lat2 = 37.8
    if place=='LA':
        dirn = './LA/NN/'
        l1bFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/Los_Angeles/TG', 'L1b')
        l2File = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/Bulk_Data/Los_Angeles/TG', 'L2')
        matFile = OCO2_Aerosol_Toolkit.search_file(date, '/net/fusi/raid06/shc/OCO_codes/MATLAB_Utilities/Los_Angeles/TG', 'mat')
        lat1 = 32.7
        lat2 = 37.8
    L1bDict = OCO2_Aerosol_Toolkit.l1b_read(l1bFile, lat1, lat2)
    L1bDict = OCO2_Aerosol_Toolkit.remv_invalid_soundings(L1bDict)
    L1bDict = OCO2_Aerosol_Toolkit.remv_ocean_soundings(L1bDict)
    CalDict = OCO2_Aerosol_Toolkit.cal_read(matFile)
    CalDict = OCO2_Aerosol_Toolkit.lat_interp_to_l1b(CalDict, L1bDict["Latitude"])
    L2Dict = OCO2_Aerosol_Toolkit.l2_read(l2File)
    L2Dict = OCO2_Aerosol_Toolkit.lat_interp_to_l1b(L2Dict, L1bDict["Latitude"])
    albedo = OCO2_Aerosol_Toolkit.albedo_interp(L1bDict, mdl)
    # Since L1b dictionary is the most dense in terms of grid, we interpolate other data onto it...
    C_level = []
    M_level = []
    for i in range(len(L1bDict["Latitude"])):
        C_level.append(OCO2_Aerosol_Toolkit.cont_level_eval(L1bDict, i, 0))
        M = []
        for j in range(2):
            M.append(OCO2_Aerosol_Toolkit.mid_level_eval_v2(L1bDict, i, j, [150, 250]))
            M.append(OCO2_Aerosol_Toolkit.mid_level_eval_v2(L1bDict, i, j, [350, 450]))
            M.append(OCO2_Aerosol_Toolkit.mid_level_eval_v2(L1bDict, i, j, [550, 650]))
        M_level.append(M)
    az = albedo * L1bDict['Zenith']
    aod, alh = OCO2_Solution_Toolkit.aod_alh_retrieval(C_level, M_level, az)
    nmax = 8
    processes = [mp.Process(target=parallel_process_unit, args=(aod, alh, l1bFile, metFile, IDPFile, L1bDict, date, n,
                                                                nmax)) for n in range(nmax)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


