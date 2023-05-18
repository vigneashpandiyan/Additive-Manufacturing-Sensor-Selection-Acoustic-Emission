# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:22:57 2023

@author: srpv
"""
import torch, math
import pandas as pd
import numpy as np

#%%

def find_IMF(x, 
             num_sifting : int = 10, 
             thres_num_extrema : int = 2):
    '''
        Extracting an intrinsic mode function using the sifting process.
        
        Parameters:
        -------------
        x : Tensor, of shape (..., # sampling points )
            Signal data. 
        num_sifting : int, optional.
            The number of sifting times. 
            ( Default : 10 )
        thres_num_extrema : int, optional 
            If (#maxima in `x`) or (#minima in `x`) <= `thres_num_extrema`,  `x` will be 
            considered as a signal residual and thus an all-zero function will be the resulting IMF.
            ( Default: 2 )
            
        Returns:
        -------------
        imf : Tensor, of shape (..., # sampling points)
              The extrated intrinsic mode functions for each signal.
              It will be on the same device as `x`.
    '''
    assert num_sifting > 0, "The number of sifting times should be at least one."
    
    x = torch.as_tensor(x).double()
    device = x.device
    N = x.shape[-1]
    batch_dim = x.shape[:-1] # the batch dimensions
    x = x.view(-1, N)
    batch_num = x.shape[0] # the number of batches
    is_residual = torch.zeros(batch_num, dtype = torch.bool, device = device)
    
    evaluate_points = (torch.arange(N, device = device).view(1, -1) + \
                       (2 * N) * torch.arange(batch_num, device = device).view(-1, 1)).view(-1)

    for _ in range(num_sifting):

        # constructing the envelope by interpolation using cubic Hermite spline
        tmp, tmpleft, tmpright = x[..., 1:-1], x[..., :-2], x[..., 2:]
        
        # ---- the upper envelope ----
        maxima_bool = torch.cat( ( (x[..., 0] >= x[..., 1]).view(-1, 1), 
                                    (tmp >= tmpright) & (tmp >= tmpleft), 
                                    (x[..., -1] >= x[..., -2]).view(-1, 1), 
                                    torch.ones((batch_num, 1), dtype = torch.bool, device = device)
                                    ), dim = 1 )
        is_residual.logical_or_( maxima_bool.sum(dim = -1) - 1 <= thres_num_extrema)
        maxima = maxima_bool.nonzero(as_tuple = False).double()
        zero_grad_pos = (maxima[:, 1] < N).logical_not()
        x_maxima = torch.zeros(maxima.shape[0], device = x.device, dtype = x.dtype)
        x_maxima[zero_grad_pos.logical_not()] = x[maxima_bool[:, :N]]
        del maxima_bool
        maxima[zero_grad_pos, 1] = N + (N-1)/2
        maxima = maxima[:, 1] + maxima[:, 0] * 2 * N 
        maxima = torch.cat( (torch.tensor(-(N+1)/2, device = device).view(1),  maxima) )
        x_maxima = torch.cat( (torch.tensor(0, device = device).view(1),  x_maxima) )
        zero_grad_pos = torch.cat( (torch.tensor(0, device = device).view(1),  zero_grad_pos) )
        envelope_up = _Interpolate(maxima, x_maxima, evaluate_points, zero_grad_pos).view(batch_num, -1)
        del maxima, x_maxima, zero_grad_pos
        
        # ---- the lower envelope ----
        minima_bool = torch.cat( ( ( x[..., 0] <= x[..., 1]).view(-1, 1), 
                                     (tmp <= tmpright) & (tmp <= tmpleft), 
                                     (x[..., -1] <= x[..., -2]).view(-1, 1), 
                                     torch.ones((batch_num, 1), dtype = torch.bool, device = device)
                                    ), dim = 1 )
        is_residual.logical_or_( minima_bool.sum(dim = -1) - 1 <= thres_num_extrema)
        del tmp, tmpleft, tmpright
        minima = minima_bool.nonzero(as_tuple = False).double()
        zero_grad_pos = (minima[:, 1] < N).logical_not()
        x_minima = torch.zeros(minima.shape[0], device = x.device, dtype = x.dtype)
        x_minima[zero_grad_pos.logical_not()] = x[minima_bool[:, :N]]
        del minima_bool
        minima[zero_grad_pos, 1] = N + (N-1)/2
        minima = minima[:, 1] + minima[:, 0] * 2 * N 
        minima = torch.cat( (torch.tensor(-(N+1)/2, device = device).view(1),  minima) )
        x_minima = torch.cat( (torch.tensor(0, device = device).view(1),  x_minima) )
        zero_grad_pos = torch.cat( (torch.tensor(0, device = device).view(1),  zero_grad_pos) )
        envelope_down = _Interpolate(minima, x_minima, evaluate_points, zero_grad_pos).view(batch_num, -1)
        del minima, x_minima, zero_grad_pos
            
        # sift and obtain an IMF candidate
        x = x - (envelope_up + envelope_down) / 2
    
    x[is_residual] = 0
    return x.view(batch_dim + torch.Size([N]))

def emd(x, 
        num_imf : int = 10, 
        ret_residual : bool = False, 
        **kwargs):
    '''
        Perform empirical mode decomposition.
        
        Parameters:
        -------------
        x : Tensor, of shape (..., # sampling points)
            Signal data.
        num_imf : int, optional. 
            The number of IMFs to be extracted from `x`.
            ( Default: 10 )
        num_sifting , thres_num_extrema : int, optional.
            See `help(find_IMF)`
        ret_residual : bool, optional. ( Default: False )
            Whether to return the residual signal as well.
        
        Returns:
        -------------
        imfs                 if `ret_residual` is False;
        (imfs, residual)     if `ret_residual` is True.
        
        imfs : Tensor, of shape ( ..., num_imf, # sampling points )
            The extrated IMFs. 
        residual : Tensor, of shape ( ...,  # sampling points )
            The residual term.
    '''
    x = torch.as_tensor(x).double()
    
    imfs = []
    for _ in range(num_imf):
        imf = find_IMF(x, **kwargs)
        imfs.append(imf)
        x = x - imf
    imfs = torch.stack(imfs, dim = -2)
    
    return (imfs, x) if ret_residual else imfs

def ComputeEMD(x, fs, 
                  num_imf : int = 10, 
                  **kwargs):
    '''
        Perform Hilbert-Huang transform on the signal `x`, and return the amplitude and 
        instantaneous frequency function of each intrinsic mode.
        
        Parameters:
        -----------
        x : Tensor, of shape (..., # sampling points)
            Signal data. 
        fs : real. 
            Sampling frequencies in Hz.
        num_imf : int, optional. 
            The number of IMFs to be extracted from `x`.
            ( Default: 10 )
        num_sifting , thres_num_extrema : int, optional.
            See `help(find_IMF)`
            
        Returns:
        -----------
        (imfs, imfs_env, imfs_freq) - 1
        
        imfs : Tensor, of shape (..., num_imf, # sampling points)
            IMFs obtained from `emd`.
        imfs_env : Tensor, of shape (..., num_imf, # sampling points - 1)
            The envelope functions of all IMFs.
        imfs_freq :Tensor, of shape (..., num_imf, # sampling points - 1)
            The instantaneous frequency functions of all IMFs, measured in 'Hz'.
    '''
    imfs = emd(x, num_imf = num_imf, **kwargs)
    # imfs_env, imfs_freq = get_envelope_frequency(imfs, fs, **kwargs)
    # return imfs, imfs_env, imfs_freq
    return imfs

# Function `_Hermite_spline` and `_Interpolate` are borrowed from @chausies and @Julius's answers 
# to the question "How to do cubic spline interpolation and integration in Pytorch" on stackoverflow.com, 
# with some modifications.     -- Daichao Chen, 2021.3.14 
# (see https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch)

A = torch.tensor([[1, 0, -3, 2],
                  [0, 1, -2, 1],
                  [0, 0, 3, -2],
                  [0, 0, -1, 1]] )

def _Hermite_spline(t):
    '''
        Helper function for cubic Hermite spline interpolation. 
    '''
    global A
    A = torch.as_tensor(A, dtype = t.dtype, device=t.device)
    return A @ (t.view(1, -1) ** torch.arange(4, device=t.device).view(-1, 1) )

def _Interpolate(x, y, xs, zero_grad_pos = None):
    '''
        Cubic Hermite spline interpolation for finding upper and lower envolopes 
        during the sifting process. 
    '''
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat( ( m[0].unsqueeze(0), 
                     (m[1:] + m[:-1]) / 2, 
                     m[-1].unsqueeze(0) )
                    )
    if (zero_grad_pos is not None):
        m[zero_grad_pos] = 0
    
    idxs = torch.searchsorted(x, xs) - 1
    
    dx = x[idxs + 1] - x[idxs]
    
    h = _Hermite_spline((xs - x[idxs]) / dx)
    return    h[0] * y[idxs] \
            + h[1] * m[idxs] * dx  \
            + h[2] * y[idxs + 1] \
            + h[3] * m[idxs + 1] * dx
            
