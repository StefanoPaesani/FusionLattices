import numpy as np
import ctypes
import platform ## Used in the c++ wrappers testing on which operating system we are working on

#########################################################################################################
###### Wrapper functions to make Gaussian elimination.
###### Sped-up version targeting loss decoding, binary operations performed in Cpp

os_system = platform.system()
if os_system == 'Windows':
    LTcpp_header = ctypes.cdll.LoadLibrary('./libLossDec_win.dll')
    print('Loaded C++ linear algebra functions for Windows OS')
elif os_system == 'Linux':
    LTcpp_header = ctypes.cdll.LoadLibrary('./libLossDec.so')
    print('Loaded C++ linear algebra functions for Linux OS')
else:
    raise ValueError('Os system not supported: only Windows or Linux')
LTcpp_header.LossDecoder_GaussElimin.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.c_int, ctypes.c_int, ctypes.c_int]
LTcpp_header.LossDecoder_GaussElimin_print.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.c_int, ctypes.c_int, ctypes.c_int]
LTcpp_header.LossDecoder_GaussElimin_trackqbts.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int]

def loss_decoding_gausselim_fast(m, num_lost_qbts, print = False):
    if m.dtype != np.uint8:
        raise ValueError("The c++ function works only for binary matrices with numpy.uint8 datatype entries.")
    nr, nc = m.shape
    # start_t = default_timer()

    REF_m = m.copy()
    if print:
        LTcpp_header.LossDecoder_GaussElimin_print(REF_m.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                             nr, nc, num_lost_qbts)
    else:
        LTcpp_header.LossDecoder_GaussElimin(REF_m.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                             nr, nc, num_lost_qbts)
    return REF_m

def loss_decoding_gausselim_fast_trackqbts(m, qbt_syndr_mat, num_lost_qbts):
    if m.dtype != np.uint8:
        raise ValueError("The c++ function works only for binary matrices with numpy.uint8 datatype entries.")
    nr, nc = m.shape

    REF_m = m.copy()
    REF_qbt_syndr_mat = qbt_syndr_mat.copy()
    LTcpp_header.LossDecoder_GaussElimin_trackqbts(REF_m.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                   REF_qbt_syndr_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                   nr, nc, num_lost_qbts)
    return REF_m, REF_qbt_syndr_mat


