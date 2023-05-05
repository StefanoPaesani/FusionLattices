import numpy as np
import matplotlib.pyplot as plt
import os

import multiprocessing
from itertools import repeat

from pymatching import Matching

cwd = os.getcwd()
saving_folder = os.path.join(cwd, "SavingFolder")

from LatticeFunctions.FFCCLattice_Fusion import FFCCLattice_BranchedFusions
from linear_algebra_inZ2 import loss_decoding_gausselim_fast_trackqbts


###########################################################################################
######## Functions for full parallelized decoder of errors & losses for Raussendorf lattice

# Biased case
def p_zz_xx(loss_p, p_fail):
    return 1 - (1-p_fail) * ((1-loss_p)**(1/p_fail)), 1 - ((1-loss_p)**(1/p_fail))

### Unbiased case
def p_zz_xx_unbiased(loss_p, p_fail):
    return 1- (1-p_fail*0.5) * ((1-loss_p)**(1/p_fail)), 1- (1-p_fail*0.5) * ((1-loss_p)**(1/p_fail))


def decoder_single_run_lossandEC(num_primal_qbts, good_bad_fusions, H_withlogop, qbt_syndr_mat, loss_array):
    #### Sample lost qubits and update H with supercells


    lost_qubits = np.random.binomial(1, loss_array).astype(np.uint8)
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
            return 1
        else:
            return 0

    else:
        return 0


def decoder_successprob_error_vs_loss_single_vals(noise_params, num_primal_qbts, good_bad_fusions, H_withlogop, qbt_syndr_mat,
                                                  failure_type, error_type, num_trials):
    loss_p, p_fail = noise_params

    num_errors = 0

    if failure_type=='Biased':
        p_zz, p_xx = p_zz_xx(loss_p, p_fail)
    elif failure_type=='Unbiased':
        p_zz, p_xx = p_zz_xx_unbiased(loss_p, p_fail)
    else:
        raise ValueError('failure_type not supported')

    (good_fusions, bad_fusions) = good_bad_fusions

    loss_array = np.zeros(num_primal_qbts)
    loss_array[good_fusions] = np.ones(len(good_fusions))*p_zz
    loss_array[bad_fusions] = np.ones(len(bad_fusions))*p_xx

    for loss_trial in range(num_trials):
        num_errors += decoder_single_run_lossandEC(num_primal_qbts, good_bad_fusions, H_withlogop, qbt_syndr_mat, loss_array)
    return num_errors / num_trials


def decoder_successprob_error_vs_loss_list_parallelized(loss_vs_fail_list, L, failure_type='Biased',
                                                        log_op_axis='z', error_type='iid',
                                                        num_trials=10000):
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

    good_fusions = np.array([cell[[0, 1, 2, 3, 4, 5, 6, 7, 8]] for cell in Lattice.cells_qbts_struct]).flatten()
    bad_fusions = np.array([cell[[9, 10, 11, 12, 13, 14, 15, 16, 17]] for cell in Lattice.cells_qbts_struct]).flatten()

    good_bad_fusions = (good_fusions, bad_fusions)

    #### Get matching matrix and define Pymatching matching object
    H_matrix = Lattice.get_matching_matrix()
    qbt_syndr_mat = np.where(H_matrix.T)[1].reshape((Lattice.num_primal_qbts, 2)).astype(dtype=np.int32)
    H_withlogop = np.vstack([H_matrix, log_op_array])

    pool = multiprocessing.Pool()
    success_probs = pool.starmap(decoder_successprob_error_vs_loss_single_vals,
                                 zip(loss_vs_fail_list, repeat(Lattice.num_primal_qbts),repeat(good_bad_fusions), repeat(H_withlogop), repeat(qbt_syndr_mat), repeat(failure_type), repeat(error_type),
                                     repeat(num_trials),))

    return np.array(success_probs)



if __name__ == '__main__':
    from timeit import default_timer
    import pickle

    ######################################################################
    ####### Test only losses - parallelized decoder
    ######################################################################

    failure_type = 'Biased'   #'Unbiased'     #'Biased'

    p_fail = 0.25

    ###### Unbiased, 25% fusion
    if p_fail == 0.25:
        if failure_type == 'Biased':
            loss_min = 0.003
            loss_max = 0.0065
        elif failure_type == 'Unbiased':
            loss_min = 0.0005
            loss_max = 0.003
        else:
            loss_min = 0.001
            loss_max = 0.0065
    else:
        loss_min = 0.001
        loss_max = 0.0065


    num_trials = 1000

    num_steps = 6


    L_list = np.arange(3, 8, 2)

    eras_ps = np.linspace(loss_min, loss_max, num_steps)
    loss_vs_fail_list = np.array([(eras_p, p_fail) for eras_p in eras_ps])

    print(loss_vs_fail_list)
    plt.figure()
    start_t = default_timer()
    for L_ix, L in enumerate(L_list):
        print('   Doing L=', L)
        this_data = \
            decoder_successprob_error_vs_loss_list_parallelized(loss_vs_fail_list, L, failure_type=failure_type, num_trials=num_trials)

        plt.errorbar(eras_ps, this_data,
                     yerr=(this_data * (1 - this_data) / num_trials) ** 0.5,
                     label="L={}".format(L))

    end_t = default_timer()
    print('Completed in ', end_t - start_t, ' s')

    plt.yscale('log')
    plt.xlabel("Loss rate")
    plt.ylabel("Logical error rate")
    plt.legend()
    plt.savefig(os.path.join(saving_folder, 'FFCCLatticeBranchFusion_withBias_' + failure_type + '_pfail'+str(int(p_fail*100))+'.png'))


