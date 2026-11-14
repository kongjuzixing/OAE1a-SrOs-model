#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the selected OAE1a Sr–Os long-tail forward solution.

Usage (Windows, from the extracted archive):
    python code/reproduce_longtail.py

Default outputs: D:\\DESK on Windows, archive/recomputed on other OSs.
This reconstructs the previously fitted forcing functions from the archived
shape parameters, and integrates Sr and Os element/isotope mass balances.
It does NOT re-optimize from the observational data.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import os

BASE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(r'D:\DESK') if os.name == 'nt' else BASE / 'recomputed'


def get_shape(t, z):
    Bpeak, Bfinal, tauB, shapeB, qf0, qfend, tauQ, shapeQ, Jpeak, tpeak, pRise, tauf, taus, w = z
    a, b, v = np.ones(len(t)), np.ones(len(t)), np.zeros(len(t))
    early = (t >= -100) & (t < 0)
    r = (t[early] + 100) / 100
    b[early] = 1 + (Bpeak - 1) * r
    qpeak = 1 + (Bfinal - 1) * qf0
    a[early] = b[early] / (1 + (qpeak - 1) * r)
    after = t >= 0
    x = t[after]
    eb, ebend = np.exp(-(x / tauB) ** shapeB), np.exp(-(1000 / tauB) ** shapeB)
    b[after] = Bfinal + (Bpeak - Bfinal) * (eb - ebend) / (1 - ebend)
    eq, eqend = np.exp(-(x / tauQ) ** shapeQ), np.exp(-(1000 / tauQ) ** shapeQ)
    q = 1 + (Bfinal - 1) * (qf0 + (qfend - qf0) * (1 - eq) / (1 - eqend))
    a[after] = b[after] / q
    up = (t >= -100) & (t < tpeak)
    v[up] = Jpeak * np.maximum(0, (t[up] + 100) / (tpeak + 100)) ** pRise
    down = t >= tpeak
    u = t[down] - tpeak
    v[down] = Jpeak * (w * np.exp(-u / tauf) + (1 - w) * np.exp(-u / taus))
    return a, b, v


def integrate(t, a, b, v):
    # Sr values: element fluxes mol/yr; evolve in 1-kyr intervals.
    SR_N0 = 1.25e17
    SR_R0 = 0.707504125
    SR_RIVER = 66.1e9
    SR_DIAG = 5.5e9
    SR_R_RIVER, SR_R_HYDRO, SR_R_DIAG = 0.711, 0.7031, SR_R0
    SR_HYDRO = (SR_RIVER * (SR_R_RIVER - SR_R0)
                + SR_DIAG * (SR_R_DIAG - SR_R0)) / (SR_R0 - SR_R_HYDRO)
    sr_k = (SR_RIVER + SR_HYDRO + SR_DIAG) * 1000 / SR_N0  # kyr^-1
    sr_e = np.exp(-sr_k)
    sr_f = -np.expm1(-sr_k) / sr_k

    # Os values: inventory t and fluxes t/kyr.
    OS_N0 = 13000.0
    OS_R0 = 0.5285493070355841
    OS_CONT, OS_COSMIC = 295.0, 17.6
    OS_R_CONT, OS_R_HYDRO, OS_R_COSMIC, OS_R_OJN = 0.8, 0.126, 0.126, 0.146
    OS_HYDRO = (OS_CONT * (OS_R_CONT - OS_R0)
                + OS_COSMIC * (OS_R_COSMIC - OS_R0)) / (OS_R0 - OS_R_HYDRO)
    os_k = (OS_CONT + OS_HYDRO + OS_COSMIC) / OS_N0  # kyr^-1
    os_e = np.exp(-os_k)
    os_f = -np.expm1(-os_k) / os_k

    n = len(t)
    ns, no = np.ones(n), np.ones(n)
    ds, do = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        j = i - 1
        js = (SR_RIVER * a[j] + SR_HYDRO * b[j] + SR_DIAG) * 1000 / SR_N0
        rs_shift = (SR_RIVER * a[j] * (SR_R_RIVER - SR_R0)
                    + SR_HYDRO * b[j] * (SR_R_HYDRO - SR_R0)
                    + SR_DIAG * (SR_R_DIAG - SR_R0)) * 1000 / SR_N0
        jo = (OS_CONT * a[j] + OS_HYDRO * b[j] + OS_COSMIC + v[j]) / OS_N0
        ro_shift = (OS_CONT * a[j] * (OS_R_CONT - OS_R0)
                    + OS_HYDRO * b[j] * (OS_R_HYDRO - OS_R0)
                    + OS_COSMIC * (OS_R_COSMIC - OS_R0)
                    + v[j] * (OS_R_OJN - OS_R0)) / OS_N0
        ns[i] = sr_e * ns[j] + sr_f * js
        ds[i] = sr_e * ds[j] + sr_f * rs_shift
        no[i] = os_e * no[j] + os_f * jo
        do[i] = os_e * do[j] + os_f * ro_shift

    return {
        'Sr_pred': SR_R0 + ds / ns,
        'Os_pred': OS_R0 + do / no,
        'Sr_stock_rel': ns,
        'Os_stock_rel': no,
        'b_over_a': b/a,
        'Sr_river_Gmol_yr': SR_RIVER / 1e9 * a,
        'Sr_hydro_Gmol_yr': SR_HYDRO / 1e9 * b,
        'Os_river_t_kyr': OS_CONT*a,
        'Os_hydro_t_kyr': OS_HYDRO*b,
        'Os_OJN_share_instantaneous': v/(OS_CONT*a+OS_HYDRO*b+OS_COSMIC+v),
        'Os_cumulative_OJN_fraction': np.cumsum(v)/np.sum(v),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(BASE / 'results' / 'OAE1a_longtail_fullwindow_minus1000_to1000.csv')
    t = reference['time_kyr'].to_numpy(dtype=float)
    if len(t) != 2001 or t[0] != -1000 or t[-1] != 1000 or not np.all(np.diff(t) == 1):
        raise ValueError('Expected the archived 1-kyr grid from -1000 to +1000 kyr')
    param = pd.read_csv(BASE / 'data' / 'OAE1a_longtail_fitted_shape_parameters.csv')
    z = param.set_index('parameter').loc[
        ['Bpeak','Bfinal','tauB','shapeB','qf0','qfend','tauQ','shapeQ',
         'Jpeak','tpeak','pRise','tau_fast','tau_slow','w'], 'value'
    ].to_numpy(float)
    a, b, v = get_shape(t, z)
    calc = integrate(t, a, b, v)
    out = reference[['time_kyr','Sr_target','Os_target']].copy()
    out['Sr_pred'] = calc['Sr_pred']
    out['Os_pred'] = calc['Os_pred']
    out['a_continental'] = a
    out['b_submarine'] = b
    out['b_over_a'] = calc['b_over_a']
    out['OJN_direct_t_kyr'] = v
    for key in ['Sr_stock_rel','Os_stock_rel','Os_OJN_share_instantaneous',
                'Sr_river_Gmol_yr','Sr_hydro_Gmol_yr','Os_river_t_kyr',
                'Os_hydro_t_kyr','Os_cumulative_OJN_fraction']:
        out[key] = calc[key]
    out = out[reference.columns]

    # Agreement with archived fit; recovered shape parameters are numerical
    # reconstructions of the historical fitted forcing curves, not independently re-optimized.
    for col, tol in [('a_continental',1e-6),('b_submarine',1e-6),
                     ('OJN_direct_t_kyr',1e-5),('Sr_pred',1e-9),('Os_pred',1e-9),
                     ('Sr_stock_rel',1e-7),('Os_stock_rel',1e-7)]:
        diff = np.max(abs(out[col].to_numpy() - reference[col].to_numpy()))
        print(f'Validation max abs difference {col}: {diff:.6g}')
        if diff > tol:
            raise AssertionError(f'{col}: recomputation deviates from archived solution')

    sel = t >= -100
    sr_rmse = np.sqrt(np.mean((out.loc[sel,'Sr_pred'] - out.loc[sel,'Sr_target'])**2))
    os_rmse = np.sqrt(np.mean((out.loc[sel,'Os_pred'] - out.loc[sel,'Os_target'])**2))
    ncie = (t >= 0) & (t <= 248)
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    ncie_share = trapz(v[ncie], t[ncie]) / trapz(v[sel], t[sel])
    summary = pd.DataFrame([{
        'time_min_kyr':-1000, 'time_max_kyr':1000,
        'Sr_RMSE':sr_rmse, 'Os_RMSE':os_rmse,
        'max_a':a.max(), 'max_b':b.max(),
        'OJN_peak_t_kyr':t[v.argmax()], 'OJN_peak_flux_t_kyr':v.max(),
        'NCIE_OJN_cumulative_share':ncie_share,
        'max_Sr_stock_rel':out.Sr_stock_rel.max(),
        'max_Os_stock_rel':out.Os_stock_rel.max()
    }])
    outpath = args.output/'OAE1a_longtail_recomputed.csv'
    summarypath = args.output/'OAE1a_longtail_metrics_recomputed.csv'
    out.to_csv(outpath, index=False)
    summary.to_csv(summarypath, index=False)
    print('\nSaved:',outpath, '\nSaved:',summarypath)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
