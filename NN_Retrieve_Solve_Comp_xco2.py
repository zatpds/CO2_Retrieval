import numpy as np
import torch.nn as nn
import torch
import OCO2_Solution_Toolkit
import OCO2_Aerosol_Toolkit

def NormalizeWith(inp,inp_ref):
    mmax = np.max(inp_ref)
    mmin = np.min(inp_ref)
    oup = (inp - mmin) / (mmax - mmin)
    return oup


def PredictFromSpectrum(spectrumIn,albIn,class_p):
    # Input is just the 3*1016 spectrum
    o2spec = spectrumIn[0,:]
    co2spec = spectrumIn[1,:]
    tmp = np.sort(o2spec)
    clev = np.mean(tmp[965:1015])
    mlev = []
    mlev.append(np.mean(tmp[150:250]) / clev)
    mlev.append(np.mean(tmp[350:450]) / clev)
    mlev.append(np.mean(tmp[550:650]) / clev)
    tmp = np.sort(co2spec)
    clev2 = np.mean(tmp[965:1015])
    mlev.append(np.mean(tmp[150:250]) / clev2)
    mlev.append(np.mean(tmp[350:450]) / clev2)
    mlev.append(np.mean(tmp[550:650]) / clev2)
    
    # Zenith and albedo, in the current version fixed
    # As we are using the synthetic spectra
    zeni = 0.99986595514
    # Ground BRDF Soil A-Band BRDF Weight Intercept = 4.154259374847318. What is that?
    alb = 0.33 # 0.33 # 4.154259374847318 / 4pi for now...
    # Normalize in the same way as the oup_to_py files get normalized
    mlev_ref = np.loadtxt("/home/shc/OCO2_Land_Investigation/Torch_Learn_Repo_2/oup_to_py/all_mlev")
    mlev = np.array(NormalizeWith(mlev,mlev_ref))
    clev_ref = np.loadtxt("/home/shc/OCO2_Land_Investigation/Torch_Learn_Repo_2/oup_to_py/all_clev")
    clev = NormalizeWith(clev,clev_ref)
    zeni_ref = np.loadtxt("/home/shc/OCO2_Land_Investigation/Torch_Learn_Repo_2/oup_to_py/all_zeni")
    zeni = NormalizeWith(zeni,zeni_ref)
    alb_ref = np.loadtxt("/home/shc/OCO2_Land_Investigation/Torch_Learn_Repo_2/oup_to_py/all_albs")
    alb = NormalizeWith(alb,alb_ref)
    inps = np.concatenate(([clev,alb,zeni],mlev),axis=0)
    # Load the 5 classification networks and do the poll
    vote = torch.from_numpy(np.zeros((2,))).float()
    for ii in range(5):
        pt_name = '/home/shc/OCO2_Land_Investigation/Torch_Learn_Repo_2/class_nets/class_' + str(ii + 1) + '.pt'
        model = nn.Sequential(
            nn.Linear(9, 10),
            nn.Sigmoid(),
            nn.Linear(10, 2),
            nn.Sigmoid()
        )
        model.load_state_dict(torch.load(pt_name))
        model.eval()
        tmp = model(torch.from_numpy(inps).float())
        vote = vote + tmp
    aod_model = nn.Sequential(
        nn.Linear(9, 10),
        nn.Sigmoid(),
        nn.Linear(10, 2)
    )
    alh_model = nn.Sequential(
        nn.Linear(9, 10),
        nn.Sigmoid(),
        nn.Linear(10, 2)
    )
    aod = []
    aod_sig = []
    alh = []
    alh_sig = []
    if vote[0]>vote[1]:
        # Type 0, use first class
        class_rec = 1
        for jj in range(8):
            aod_net = '/home/shc/OCO2_Land_Investigation/Torch_Learn_Repo_2/fit_nets_linscale/ensemble_' + str(jj + 1) + '/aod_class_1.pt'
            alh_net = '/home/shc/OCO2_Land_Investigation/Torch_Learn_Repo_2/fit_nets_linscale/ensemble_' + str(jj + 1) + '/alh_class_1.pt'
            aod_model.load_state_dict(torch.load(aod_net))
            aod_model.eval()
            alh_model.load_state_dict(torch.load(alh_net))
            alh_model.eval()
            aod_now = aod_model(torch.from_numpy(inps).float())
            alh_now = alh_model(torch.from_numpy(inps).float())
            aod_now = aod_now.detach().numpy()
            alh_now = alh_now.detach().numpy()
            # Section 1 need to transform...
            aod_now[1] = abs(aod_now[1] / aod_now[0])
            aod_now[0] = np.log(aod_now[0])
            aod.append(aod_now[0])
            aod_sig.append(aod_now[1])
            alh.append(alh_now[0])
            alh_sig.append(alh_now[1])
    else:
        # Type 1, use second class
        class_rec = 2
        for jj in range(8):
            aod_net = '/home/shc/OCO2_Land_Investigation/Torch_Learn_Repo_2/fit_nets_linscale/ensemble_' + str(jj + 1) + '/aod_class_2.pt'
            alh_net = '/home/shc/OCO2_Land_Investigation/Torch_Learn_Repo_2/fit_nets_linscale/ensemble_' + str(jj + 1) + '/alh_class_2.pt'
            aod_model.load_state_dict(torch.load(aod_net))
            aod_model.eval()
            alh_model.load_state_dict(torch.load(alh_net))
            alh_model.eval()
            aod_now = aod_model(torch.from_numpy(inps).float())
            alh_now = alh_model(torch.from_numpy(inps).float())
            aod_now = aod_now.detach().numpy()
            alh_now = alh_now.detach().numpy()
            aod.append(aod_now[0])
            aod_sig.append(aod_now[1])
            alh.append(alh_now[0])
            alh_sig.append(alh_now[1])
    fin_aod = np.nanmean(aod)
    fin_aod_sig = np.nanmean(np.array(aod_sig) * np.array(aod_sig)) + \
                  np.nanmean(np.array(aod) * np.array(aod)) - fin_aod * fin_aod
    fin_alh = np.nanmean(alh)
    fin_alh_sig = np.nanmean(np.array(alh_sig) * np.array(alh_sig)) + \
                  np.nanmean(np.array(alh) * np.array(alh)) - fin_alh * fin_alh
    diag = inps # Diagnostic variable
    return class_rec, fin_aod, fin_aod_sig, fin_alh, fin_alh_sig, diag


import OCO2_Aerosol_Toolkit
import l2_afp


spec_data_path = '/net/fusi/raid06/shc/Bulk_Data/OCO_Forward/Riyadh_new/co2_plume_0_rd.npy' # Just need to care about AOD up to 1
spec_data = np.load(spec_data_path,allow_pickle=True)
sol_noc_rec = []
sol_ret_rec = []
config_file = l2_afp.utils.get_lua_config_files()['default']
merradir = '/home/zcz/OCO2_FP_RTRetrieval_AronneWrapper/examples/FORWARD_ZZ/MERRA'
abscodir = '/home/zcz/OCO2_FP_RTRetrieval_AronneWrapper/examples/FORWARD_ZZ/ABSCO'

date = 20160522

l1bFile = OCO2_Aerosol_Toolkit.search_file(date, '/home/shc/OCO2_Land_Investigation/Data/Riyadh/L1b_Mod', 'L1b')
l2File = OCO2_Aerosol_Toolkit.search_file(date, '/home/shc/OCO2_Land_Investigation/Data/Riyadh/L2', 'L2')
metFile = OCO2_Aerosol_Toolkit.search_file(date, '/home/shc/OCO2_Land_Investigation/Data/Riyadh/Met', 'Met')
IDPFile = OCO2_Aerosol_Toolkit.search_file(date, '/home/shc/OCO2_Land_Investigation/Data/Riyadh/IDP', 'IDP')
# <Go to latitude 29 for the retrieval, as it is where the fake plume is the largest>
lat1 = 29
lat2 = 29.1
# OCO2_Aerosol_Toolkit.dirnew(dirn)
L1bDict = OCO2_Aerosol_Toolkit.l1b_read(l1bFile, lat1, lat2)
L1bDict = OCO2_Aerosol_Toolkit.remv_invalid_soundings(L1bDict)
sid = str(L1bDict['Sounding_ID'][0])

noc_flags = []
ret_flags = []

for i in range(spec_data.shape[0]):
    tmp = spec_data[i,:,:]
    class_rec, aod, aod_sig, alh, alh_sig, diag = PredictFromSpectrum(tmp,33,0)
    # Retrieval using two different ways, 1: non-constrained
    l2_obj = l2_afp.wrapped_fp(
        l1bFile, metFile, config_file, merradir, abscodir,
        sounding_id=sid, imap_file=IDPFile,
        enable_console_log=False)
    # Set the Dust to different levels.
    x_a = l2_obj.get_x()
    S_a = l2_obj.get_Sa()
    Kupdate = l2_obj.Kupdate
    flag = 1
    try:
        x, _, _ = OCO2_Solution_Toolkit.l2_obj_solve(l2_obj, Kupdate)
    except Exception as e:
        print(str(i) + '-noc does not yield valid retrieval.')
        print('Problem below:')
        print(e)
        flag = 0
    noc_flags.append(flag)
    sol_noc_rec.append(x[-1])
    # Set other aerosols to 0 forcefully! Only keep the primary aerosol source...
    x_a[26] = -9999
    x_a[29] = -9999
    x_a[32] = -9999
    x_a[35] = -9999
    x_a[39] = 0
    l2_obj.set_x(x_a)
    l2_obj = OCO2_Solution_Toolkit.constrain_alh_err(l2_obj, aod, aod_sig)
    l2_obj = OCO2_Solution_Toolkit.constrain_aod_err(l2_obj, alh, alh_sig)
    Kupdate = l2_obj.Kupdate
    flag = 1
    try:
        x, _, _ = OCO2_Solution_Toolkit.l2_obj_solve(l2_obj, Kupdate)
    except Exception as e:
        print(str(i) + '-ret does not yield valid retrieval.')
        print('Problem below:')
        print(e)
        flag = 0
    ret_flags.append(flag)
    sol_ret_rec.append(x[-1])

np.save('./Synthetic_Retrieval/Apr6_xco2_noc.npy',sol_noc_rec)
np.save('./Synthetic_Retrieval/Apr6_xco2_ret.npy',sol_ret_rec)
np.save('./Synthetic_Retrieval/Apr6_xco2_noc_flags.npy',noc_flags)
np.save('./Synthetic_Retrieval/Apr6_xco2_ret_flags.npy',ret_flags)