'''
Initial design intended for the constrained solution. Passing forward and back the l2_afp objects
All Operations:
l2_obj.Kupdate                        l2_obj.get_aerosol_total_ref_OD       l2_obj.get_sample_indexes             l2_obj.num_samples
l2_obj.L2Run                          l2_obj.get_altitude_levels            l2_obj.get_solar_distance             l2_obj.set_Sa
l2_obj.aerosol_names                  l2_obj.get_gas_column_numdensity      l2_obj.get_state_variable_names       l2_obj.set_x
l2_obj.forward_run                    l2_obj.get_gas_profile_numdensity     l2_obj.get_stokes_coefficients        l2_obj.solve
l2_obj.forward_run_highres            l2_obj.get_gas_vmr_profile            l2_obj.get_temperature_profile        l2_obj.write_h5_output_file
l2_obj.get_Sa                         l2_obj.get_geometry                   l2_obj.get_x                          l2_obj.write_h5_output_file_FPformat
l2_obj.get_Se_diag                    l2_obj.get_noise                      l2_obj.get_y
l2_obj.get_aerosol_layer_OD           l2_obj.get_pressure_levels            l2_obj.jacobian_run

'''
import numpy as np
import l2_afp
import os
from scipy.io import loadmat


def constrain_aod(l2_obj, aod):
    var_name = l2_obj.get_state_variable_names()
    for j in range(len(var_name)):
        if var_name[j][0:13]=='Aerosol Shape':
            break
    x = l2_obj.get_x()
    x[j] = aod
    x[j + 3] = -10
    l2_obj.set_x(x)
    Sa = l2_obj.get_Sa()
    Sa[j, j] = Sa[j, j] / 4
    Sa[j + 3, j + 3] = Sa[j, j] / 4
    l2_obj.set_Sa(Sa)
    return l2_obj


def constrain_alh(l2_obj, alh):
    var_name = l2_obj.get_state_variable_names()
    for j in range(len(var_name)):
        if var_name[j][0:13] == 'Aerosol Shape':
            break
    x = l2_obj.get_x()
    x[j + 1] = alh
    x[j + 4] = alh
    l2_obj.set_x(x)
    Sa = l2_obj.get_Sa()
    Sa[j + 1, j + 1] = Sa[j + 1, j + 1] / 4
    Sa[j + 4, j + 4] = Sa[j + 1, j + 1] / 4
    l2_obj.set_Sa(Sa)
    return l2_obj

def constrain_aod_err(l2_obj, aod, err):
    var_name = l2_obj.get_state_variable_names()
    for j in range(len(var_name)):
        if var_name[j][0:13]=='Aerosol Shape':
            break
    x = l2_obj.get_x()
    x[j] = aod
    x[j + 3] = aod-3
    l2_obj.set_x(x)
    Sa = l2_obj.get_Sa()
    Sa[j, j] = err
    # Sa[j + 3, j + 3] = Sa[j+3, j+3] / 4
    l2_obj.set_Sa(Sa)
    return l2_obj


def constrain_alh_err(l2_obj, alh, err):
    var_name = l2_obj.get_state_variable_names()
    for j in range(len(var_name)):
        if var_name[j][0:13] == 'Aerosol Shape':
            break
    x = l2_obj.get_x()
    x[j + 1] = alh
    x[j + 4] = alh
    l2_obj.set_x(x)
    Sa = l2_obj.get_Sa()
    Sa[j + 1, j + 1] = err
    # Sa[j + 4, j + 4] = Sa[j + 4, j + 4] / 4
    l2_obj.set_Sa(Sa)
    return l2_obj


def l2_obj_solve(l2_obj, Kupdate):
    x0 = l2_obj.get_x()
    y = l2_obj.get_y()
    Sa = l2_obj.get_Sa()
    Se_diag = l2_obj.get_Se_diag()
    Se = l2_afp.utils.diagcmatrix(Se_diag)
    x_i_list, Fx_i_list, S_i_list = l2_afp.bayesian_nonlinear_solver(
        Se, Sa, y, x0, Kupdate, start_gamma=10.0,
        max_iteration_ct=5, debug_write=False,
        debug_write_prefix='test_l2_afp_match_run',
        match_l2_fp_costfunc=True)
    return x_i_list, Fx_i_list, S_i_list


def l2_obj_update(l2_obj, key, xin, sin):
    x = l2_obj.get_x()
    if xin != 0:
        x[key] = xin
    l2_obj.set_x(x)
    Sa = l2_obj.get_Sa()
    if sin != 0:
        Sa[key, key] = sin
    l2_obj.set_Sa(Sa)
    return l2_obj


def construc_l2_obj(l1bfile, metfile, idpfile, sounding_id):
    merradir = '/net/fusi/raid06/zcz/OCO2_FP_RTRetrieval/RtRetrievalFramework-B8.1.00/build_pythonwrapper/l2_afp-master/examples/FORWARD_ZZ/MERRA'
    abscodir = '/net/fusi/raid06/zcz/OCO2_FP_RTRetrieval/RtRetrievalFramework-B8.1.00/build_pythonwrapper/l2_afp-master/examples/FORWARD_ZZ/ABSCO'
    config_file = l2_afp.utils.get_lua_config_files()['default']
    l2_obj = l2_afp.wrapped_fp(
        l1bfile, metfile, config_file, merradir, abscodir,
        sounding_id=sounding_id, imap_file=idpfile,
        enable_console_log=False)
    return l2_obj


def construc_constrained_l2_obj(l1bfile, metfile, idpfile, sounding_id):
    merradir = '/net/fusi/raid06/zcz/OCO2_FP_RTRetrieval/RtRetrievalFramework-B8.1.00/build_pythonwrapper/l2_afp-master/examples/FORWARD_ZZ/MERRA'
    abscodir = '/net/fusi/raid06/zcz/OCO2_FP_RTRetrieval/RtRetrievalFramework-B8.1.00/build_pythonwrapper/l2_afp-master/examples/FORWARD_ZZ/ABSCO'
    config_file = l2_afp.utils.get_lua_config_files()['fixed_aerosol']
    l2_obj = l2_afp.wrapped_fp(
        l1bfile, metfile, config_file, merradir, abscodir,
        sounding_id=sounding_id, imap_file=idpfile,
        enable_console_log=False)
    return l2_obj



def aod_alh_retrieval(c, m, az):
    # Albedo Zenith (N * 1), C_level (N * 1), M_level (N * 6)
    # From Neurak Network. in the order of [c m az]
    # Outputs both N*1, AOD and ALH
    inps = np.zeros((len(c), 8))
    inps[:, 0] = c
    inps[:, 1:7] = m
    inps[:, 7] = az
    np.savetxt('./NN_inputs.csv', inps, delimiter=',')
    os.system('cat use_trained_NN.m | matlab -nodesktop -nosplash')
    flag = 0
    while flag==0:
        try:
            outs = loadmat('./NN_outputs.mat')
            os.system('rm -rf ./NN_outputs.mat')
            print('MATLAB execution done.')
            flag = 1
        except:
            continue
    aod = outs['AOD']
    alh = outs['ALH']
    return aod, alh


def constrain_alh_aod_lua_config(aod, alh):
    from tempfile import mkstemp
    from shutil import move
    from os import fdopen, remove
    f = ('/home/shc/.local/lib/python2.7/site-packages/l2_afp-0.0.1-py2.7.egg/l2_afp/lua_configs/'+
         'custom_config_fixed_aerosol.lua')
    fh, abs_path = mkstemp()
    newl = '    return ConfigCommon.lua_to_blitz_double_1d({%.3f, %.3f, 0.2}) --Mark1 \n'%(np.log(aod), np.exp(-1.0 * alh / 8))
    with fdopen(fh, 'w') as new_file:
        with open(f) as old_file:
            for line in old_file:
                if line.__contains__('--Mark1'):
                    new_file.write(newl)
                else:
                    new_file.write(line)
    # Remove original file
    remove(f)
    # Move new file
    move(abs_path, f)


def set_y_l1b_file(y,idx,l1bfile):
    # y is 3*1016
    import h5py
    f = h5py.File(l1bfile, 'r+')
    o2_rad = f['SoundingMeasurements/radiance_o2']
    o2_rad[idx,1,:] = np.squeeze(y[0,:])
    co2w_rad = f['SoundingMeasurements/radiance_weak_co2']
    co2w_rad[idx,1,:] = np.squeeze(y[1,:])
    co2s_rad = f['SoundingMeasurements/radiance_strong_co2']
    co2s_rad[idx,1,:] = np.squeeze(y[2,:])
    f.close()

