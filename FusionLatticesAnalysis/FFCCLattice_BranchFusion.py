import numpy as np
import matplotlib.pyplot as plt
import os

import multiprocessing
from itertools import repeat

from pymatching import Matching

cwd = os.getcwd()
saving_folder = os.path.join(cwd, "SavingFolder")

try:
    from LatticeFunctions.FFCCLattice_Fusion import FFCCLattice_BranchedFusions
    from linear_algebra_inZ2 import loss_decoding_gausselim_fast_trackqbts
    from misc_functions import merge_multiedges_in_Hmat_faster, get_multiedge_errorprob, get_Hmat_weights
except:
    from .LatticeFunctions.FFCCLattice_Fusion import FFCCLattice_BranchedFusions
    from .linear_algebra_inZ2 import loss_decoding_gausselim_fast_trackqbts
    from .misc_functions import merge_multiedges_in_Hmat_faster, get_multiedge_errorprob, get_Hmat_weights

###########################################################################################
######## Functions for full parallelized decoder of errors & losses for Raussendorf lattice


def decoder_single_run_lossandEC(num_primal_qbts, H_withlogop, qbt_syndr_mat, err_p, loss_p, error_type='iid',
                                 num_ec_runs_per_loss_trial=1):
    #### Sample lost qubits and update H with supercells
    lost_qubits = np.random.binomial(1, loss_p, num_primal_qbts).astype(np.uint8)
    num_lost_qbts = np.sum(lost_qubits)
    if num_lost_qbts:
        ## Lost first
        lostfirst_qbts_order = np.flip(np.argsort(lost_qubits))
        Hwithlogop_ordered = H_withlogop[:, lostfirst_qbts_order]
        qbt_syndr_mat_dec = qbt_syndr_mat[lostfirst_qbts_order]

        Hwithlogop_ordered_rowechelon, qbt_syndr_mat_dec = loss_decoding_gausselim_fast_trackqbts(Hwithlogop_ordered,
                                                                                                  qbt_syndr_mat_dec,
                                                                                                  num_lost_qbts)

        new_logop = Hwithlogop_ordered_rowechelon[-1]

        if np.any(new_logop[:num_lost_qbts]):
            return 1 * num_ec_runs_per_loss_trial
    else:
        Hwithlogop_ordered_rowechelon = H_withlogop
        qbt_syndr_mat_dec = qbt_syndr_mat
        new_logop = Hwithlogop_ordered_rowechelon[-1]

    if err_p > 1.e-10:

        new_logop = new_logop[num_lost_qbts:]
        new_qbt_syndr_mat_dec = qbt_syndr_mat_dec[num_lost_qbts:]

        if num_lost_qbts > 0:
            first_nonlostsyndr = (int(num_lost_qbts) - np.argmax(
                (np.any(Hwithlogop_ordered_rowechelon[:num_lost_qbts, :num_lost_qbts], axis=1))[::-1]))
            NewH_unfiltered = Hwithlogop_ordered_rowechelon[first_nonlostsyndr:-1, num_lost_qbts:]
            new_qbt_syndr_mat_dec = new_qbt_syndr_mat_dec - first_nonlostsyndr
        else:
            NewH_unfiltered = Hwithlogop_ordered_rowechelon[:-1]

        new_qbt_syndr_mat_dec, new_ixs, inverse_ixs, occ_counts, has_no_zeroed_qbt = merge_multiedges_in_Hmat_faster(
            new_qbt_syndr_mat_dec)
        NewH = NewH_unfiltered[:, new_ixs]
        new_logop = new_logop[new_ixs]

        if NewH.shape[0] < 2:
            return 1 * num_ec_runs_per_loss_trial

        error_probs = get_multiedge_errorprob(error_type=error_type, p=err_p, occ_counts=occ_counts)
        NewH_weights = get_Hmat_weights(error_type=error_type, multiedge_error_probs=error_probs)

        matching = Matching(NewH, spacelike_weights=NewH_weights, error_probabilities=error_probs)

        ##### Noise simulation
        num_ec_fails = 0
        for _ in range(num_ec_runs_per_loss_trial):
            noise, syndrome = matching.add_noise()
            correction = matching.decode(syndrome)
            num_ec_fails += np.dot(new_logop, (correction + noise)) % 2
        return num_ec_fails

    else:
        return 0


def decoder_successprob_error_vs_loss_single_vals(noise_vals, num_primal_qbts, H_withlogop, log_op_array, qbt_syndr_mat,
                                                  error_type, num_loss_trials, num_ec_runs_per_loss_trial):
    (error_p, loss_p) = noise_vals
    num_errors = 0
    if loss_p > 0:
        for loss_trial in range(num_loss_trials):
            num_errors += decoder_single_run_lossandEC(num_primal_qbts, H_withlogop, qbt_syndr_mat, error_p, loss_p,
                                                       error_type=error_type,
                                                       num_ec_runs_per_loss_trial=num_ec_runs_per_loss_trial)
    else:
        if error_type == 'iid':

            new_qbt_syndr_mat_dec, new_ixs, inverse_ixs, occ_counts, has_no_zeroed_qbt = merge_multiedges_in_Hmat_faster(
                qbt_syndr_mat)
            NewH = H_withlogop[:-1, new_ixs]
            new_logop = H_withlogop[-1, new_ixs]
            if NewH.shape[0] < 2:
                return 1 * num_ec_runs_per_loss_trial

            error_probs = get_multiedge_errorprob(error_type=error_type, p=error_p, occ_counts=occ_counts)
            NewH_weights = get_Hmat_weights(error_type=error_type, multiedge_error_probs=error_probs)

            matching = Matching(NewH, spacelike_weights=NewH_weights, error_probabilities=error_probs)

        else:
            raise ValueError('Only accepted error types are: ["iid"]')
        for _ in range(num_loss_trials * num_ec_runs_per_loss_trial):
            noise, syndrome = matching.add_noise()
            correction = matching.decode(syndrome)
            num_errors += np.dot(new_logop, (correction + noise)) % 2
    return (num_errors / (num_loss_trials * num_ec_runs_per_loss_trial))


def decoder_successprob_error_vs_loss_list_parallelized(error_vs_loss_list, L, log_op_axis='z', error_type='iid',
                                                        num_loss_trials=1000, num_ec_runs_per_loss_trial=1):
    if log_op_axis == 'x':
        log_op_ix = 0
    elif log_op_axis == 'y':
        log_op_ix = 1
    elif log_op_axis == 'z':
        log_op_ix = 2
    else:
        raise ValueError('log_op_axis needs to be in [x, y, z]')

    Lattice = FFCCLattice_BranchedFusions(L, L, L)
    log_op_qbts0 = Lattice.log_ops_qbts[log_op_ix]
    log_op_array = np.array([1 if x in log_op_qbts0 else 0 for x in range(Lattice.num_primal_qbts)],
                            dtype=np.uint8)

    #### Get matching matrix and define Pymatching matching object
    H_matrix = Lattice.get_matching_matrix()
    qbt_syndr_mat = np.where(H_matrix.T)[1].reshape((Lattice.num_primal_qbts, 2)).astype(dtype=np.int32)
    H_withlogop = np.vstack([H_matrix, log_op_array])

    pool = multiprocessing.Pool()
    success_probs = pool.starmap(decoder_successprob_error_vs_loss_single_vals,
                                 zip(error_vs_loss_list, repeat(Lattice.num_primal_qbts), repeat(H_withlogop),
                                     repeat(log_op_array), repeat(qbt_syndr_mat), repeat(error_type),
                                     repeat(num_loss_trials), repeat(num_ec_runs_per_loss_trial)))

    return np.array(success_probs)



if __name__ == '__main__':
    from timeit import default_timer
    import pickle

    ######################################################################
    ####### Test only losses - parallelized decoder
    ######################################################################

    loss_min = 0.11
    loss_max = 0.16

    num_trials = 1000

    num_steps = 10

    L_list = np.arange(3, 8, 2)

    eras_ps = np.linspace(loss_min, loss_max, num_steps)
    err_vs_eras_vals = np.array([(0, eras_p) for eras_p in eras_ps])

    print(err_vs_eras_vals)
    plt.figure()
    start_t = default_timer()
    for L_ix, L in enumerate(L_list):
        print('   Doing L=', L)
        this_data = \
            decoder_successprob_error_vs_loss_list_parallelized(err_vs_eras_vals, L, num_loss_trials=num_trials,
                                                                num_ec_runs_per_loss_trial=1)

        plt.errorbar(eras_ps, this_data,
                     yerr=(this_data * (1 - this_data) / num_trials) ** 0.5,
                     label="L={}".format(L))

    end_t = default_timer()
    print('Completed in ', end_t - start_t, ' s')
    plt.yscale('log')
    plt.xlabel("Loss rate")
    plt.ylabel("Logical error rate")
    plt.legend()
    plt.savefig(os.path.join(saving_folder, 'FFCCLatticeBranchFusion_lossonly.png'))

    ######################################################################
    ####### Test only errors - parallelized decoder
    ######################################################################


    err_min = 0.001
    err_max = 0.017

    log_op_dir = 'z'

    num_trials = 3000

    num_steps = 10

    L_list = np.arange(4, 9, 2)

    err_ps = np.linspace(err_min, err_max, num_steps)
    err_vs_eras_vals = np.array([(err_p, 0.) for err_p in err_ps])

    print(err_vs_eras_vals)
    plt.figure()
    start_t = default_timer()
    for L_ix, L in enumerate(L_list):
        print('   Doing L=', L)
        this_data = \
            decoder_successprob_error_vs_loss_list_parallelized(err_vs_eras_vals, L, log_op_axis=log_op_dir,
                                                                num_loss_trials=num_trials,
                                                                num_ec_runs_per_loss_trial=1)

        plt.errorbar(err_ps, this_data,
                     yerr=(this_data * (1 - this_data) / num_trials) ** 0.5,
                     label="L={}".format(L))

    end_t = default_timer()
    print('Completed in ', end_t - start_t, ' s')

    plt.yscale('log')
    plt.xlabel("Physical error rate")
    plt.ylabel("Logical error rate")
    plt.legend()
    plt.savefig(os.path.join(saving_folder, 'FFCCLatticeBranchFusion_errorsonly.png'))


    ######################################################################
    ####### Full 2D scan with losses & errors
    ######################################################################


    # loss_min = 0.05
    # loss_max = 0.145
    #
    # err_min = 0.005
    # err_max = 0.016
    #
    # num_loss_trials = 2000
    # num_ec_runs_per_loss_trial = 20
    #
    # num_steps = 15
    # num_scans = 5
    #
    # log_op_dir = 'z'
    #
    # L_list = np.arange(4, 13, 4)
    #
    #
    # angles_vals = np.linspace(0, np.pi / 2., num_scans)
    # x_vals = np.linspace(0, 1, num_steps)
    #
    # err_vs_eras_vals_allvals = []
    # all_data = np.array([[None] * len(L_list)] * num_scans)
    #
    # for scan_ix, angle_val in enumerate(angles_vals):
    #     print('\nStarting scan', scan_ix, 'of', num_scans)
    #     err_vs_eras_vals = np.array([(((err_max - err_min) * x + err_min) * np.cos(angle_val),
    #                                   ((loss_max - loss_min) * x + loss_min) * np.sin(angle_val)) for x in x_vals])
    #     err_vs_eras_vals_allvals.append(err_vs_eras_vals)
    #     print(err_vs_eras_vals)
    #     plt.figure()
    #     start_t = default_timer()
    #     for L_ix, L in enumerate(L_list):
    #         print('   Doing L=', L)
    #         this_data = decoder_successprob_error_vs_loss_list_parallelized(err_vs_eras_vals, L, log_op_axis=log_op_dir,
    #                                                                         num_loss_trials=num_loss_trials,
    #                                                                         num_ec_runs_per_loss_trial=num_ec_runs_per_loss_trial)
    #         all_data[scan_ix][L_ix] = np.array(this_data)
    #         plt.errorbar(x_vals, this_data,
    #                      yerr=(this_data * (1 - this_data) / (num_loss_trials * num_ec_runs_per_loss_trial)) ** 0.5,
    #                      label="L={}".format(L))
    #     end_t = default_timer()
    #     print('This scan took:', end_t - start_t, ' s')
    #     plt.yscale('log')
    #     plt.xlabel("Physical error rate")
    #     plt.ylabel("Logical error rate")
    #     plt.legend()
    #     plt.savefig(os.path.join(saving_folder,
    #                              'PlotDuring2DScan_FFCCLatticeBranchFusion_scan'+str(scan_ix)+'of'+str(num_scans)+ '_logopaxis_' + log_op_dir +'.png'))
    #
    # file_name = 'error_vs_erasure_FFCCLatticeBranchFusion_Lmax' + str(max(L_list)) + '_logopaxis_' + log_op_dir +'_numlosstrials' + str(
    #     num_loss_trials) + '_numecrunsperlosstrial' + str(num_ec_runs_per_loss_trial) + '_numsteps' + \
    #             str(num_steps) + '_numscans' + str(num_scans) + '.pickle'
    # data_to_save = [L_list, err_vs_eras_vals_allvals, all_data]
    #
    # with open(os.path.join(saving_folder, file_name), 'wb') as handle:
    #     pickle.dump(data_to_save, handle)
