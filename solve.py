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


place = 'Riyadh'

if place=='Riyadh':
    dates = [20161029, 20160826, 20160522, 20160319, 20160115]

for date in dates:
    OCO2_Aerosol_Toolkit.dirnew(str(date) + appen)
    print('Doing for date:%d' % date)
    mdl = OCO2_Aerosol_Toolkit.albedo_construc(date)
    
    if place=='Riyadh':
        dirn = './Results_Riyadh/NN/'
        l1bFile = OCO2_Aerosol_Toolkit.search_file(date, './Riyadh/L1b', 'L1b')
        l2File = OCO2_Aerosol_Toolkit.search_file(date, './Riyadh/L2', 'L2')
        matFile = OCO2_Aerosol_Toolkit.search_file(date, './net/fusi/raid06/shc/OCO_codes/MATLAB_Utilities/Riyadh',
                                                   'mat')
        metFile = OCO2_Aerosol_Toolkit.search_file(date, './Riyadh/Met', 'Met')
        IDPFile = OCO2_Aerosol_Toolkit.search_file(date, './Riyadh/L1b/IDP', 'IDP')
        lat1 = 20
        lat2 = 30
        
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


