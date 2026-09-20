"""
Inverse Problem Resolution via Levenberg-Marquardt Algorithm for V-AGMD
Determining Effective Air Gap and Critical Membrane Parameters from Experimental Data
Master's Thesis / International Journal Publication Standard
"""

import os
import subprocess
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import t

# Ensure output directories exist
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Thermophysical density correlation
def rho(temperature, salinity):
    a = [9.999e2, 2.034e-2, -6.162e-3, 2.261e-5, -4.657e-8]
    b = [8.020e2, -2.001, 1.677e-2, -3.060e-5, -1.613e-5]
    temp = temperature
    t_part = a[0] + a[1]*temp + a[2]*temp**2 + a[3]*temp**3 + a[4]*temp**4
    s_part = b[0]*salinity + b[1]*salinity*temp + b[2]*salinity*temp**2 + b[3]*salinity*temp**3 + b[4]*(salinity**2)*(temp**2)
    return t_part + s_part

def simulate_case(row, air_gap=0.0009, pore_diameter=3.5e-7, membrane_porosity=0.7, membrane_thickness=0.00015):
    """Runs vagmd0Dmodel for a specific experimental case with custom parameters."""
    T_f_in = row['Feed temperature at the inlet (°C)']
    T_c_in = row['Coolant temperature at the inlet (°C)']
    real_flow_mL = row['Real Flow (mL/min)']
    P_vac = row['Vacuum pressure (Pa)']
    
    dens_f = rho(T_f_in, 0.0003)
    dens_c = rho(T_c_in, 0.0003)
    m_dot_f = dens_f * (real_flow_mL * 1e-6 / 60.0)
    m_dot_c = dens_c * (real_flow_mL * 1e-6 / 60.0)
    
    cmd = [
        './bin/vagmd0Dmodel',
        '-membrane_area', '0.007853975',
        '-entry_temperature_feed', str(T_f_in),
        '-entry_temperature_cool', str(T_c_in),
        '-feed_mass_flow_rate', str(m_dot_f),
        '-cool_mass_flow_rate', str(m_dot_c),
        '-vacuum_pressure', str(P_vac),
        '-air_gap_thickness', str(air_gap),
        '-feed_channel_height', '0.002',
        '-cold_channel_height', '0.002',
        '-channel_width', '0.0886226551170749',
        '-number_channels', '1',
        '-spacer_porosity', '1.0',
        '-gap_spacer_porosity', '0.636620310097753',
        '-wall_thickness', '0.002',
        '-spacer_conductivity', '0.27',
        '-wall_conductivity', '18.0',
        '-polymer_conductivity', '0.3',
        '-pore_diameter', str(pore_diameter),
        '-membrane_porosity', str(membrane_porosity),
        '-membrane_thickness', str(membrane_thickness),
        '-entry_salinity_feed', '0.0003',
        '-entry_salinity_cool', '0.0003'
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return {'perm_g_min': 0.0, 'T_f_out': 0.0, 'T_c_out': 0.0, 'dist_L_h': 0.0}
    
    report = pd.read_csv('./results/report.csv', header=None)
    try:
        perm_row = report[report[0] == 'Permeate flow rate =']
        if len(perm_row) > 0:
            perm_g_min = float(perm_row[1].values[0])
        else:
            mass_flux = float(report[report[0] == 'Mass flux ='][1].values[0])
            perm_g_min = mass_flux * 0.007853975 * 1000.0 / 60.0
    except Exception:
        mass_flux = float(report[report[0] == 'Mass flux ='][1].values[0])
        perm_g_min = mass_flux * 0.007853975 * 1000.0 / 60.0
        
    T_f_out = float(report[report[0] == 'Feed temperature at the outlet of the module ='][1].values[0])
    T_c_out = float(report[report[0] == 'Coolant temperature at the outlet of the module ='][1].values[0])
    dist_L_h = float(report[report[0] == 'Distillate flow rate ='][1].values[0])
    
    return {
        'perm_g_min': perm_g_min,
        'T_f_out': T_f_out,
        'T_c_out': T_c_out,
        'dist_L_h': dist_L_h
    }

def main():
    print("=== STARTING LEVENBERG-MARQUARDT INVERSE PROBLEM ESTIMATION ===")
    
    df_master = pd.read_csv('Experimento_bancada_geral.csv')
    n_cases = len(df_master)
    print(f"Loaded {n_cases} experimental cases.")
    
    # -------------------------------------------------------------
    # 1. BASELINE: DEFAULT NOMINAL PARAMETERS (Air Gap = 2.0 mm)
    # -------------------------------------------------------------
    default_gap = 0.002 # 2.0 mm
    res_default = []
    for idx, row in df_master.iterrows():
        out = simulate_case(row, air_gap=default_gap)
        res_default.append(out)
    df_default = pd.DataFrame(res_default)
    
    # -------------------------------------------------------------
    # 2. INVERSE LEVEL 1: GLOBAL OPTIMAL AIR GAP
    # -------------------------------------------------------------
    print("\n--- Level 1: Global Effective Air Gap Optimization ---")
    def residual_global(x):
        gap = x[0]
        resids = []
        for idx, row in df_master.iterrows():
            y_exp = row['Permeate mass flow rate (g/min)']
            unc = max(row['Permeate mass flow rate uncertainty (g/min)'], 1e-4)
            out = simulate_case(row, air_gap=gap)
            resids.append((out['perm_g_min'] - y_exp) / unc)
        return np.array(resids)
    
    opt_global = least_squares(residual_global, x0=[0.0009], bounds=(0.00005, 0.01), method='trf', ftol=1e-7, xtol=1e-7)
    delta_g_global = opt_global.x[0]
    
    # Uncertainty Calculation for Global Gap
    J_glob = opt_global.jac
    cov_glob = np.linalg.inv(J_glob.T.dot(J_glob))
    u_std_glob = np.sqrt(cov_glob[0, 0])
    dof_glob = n_cases - 1
    k_glob = t.ppf(0.975, dof_glob) if dof_glob > 0 else 2.00
    U_glob_mm = (k_glob * u_std_glob) * 1000.0  # Convert to mm

    print(f"Global Optimal Effective Air Gap: {delta_g_global*1000:.4f} ± {U_glob_mm:.4f} mm (Cost: {opt_global.cost:.4f})")
    
    res_global = []
    for idx, row in df_master.iterrows():
        out = simulate_case(row, air_gap=delta_g_global)
        res_global.append(out)
    df_global = pd.DataFrame(res_global)
    
    # -------------------------------------------------------------
    # 3. INVERSE LEVEL 2: FLOW-DEPENDENT EFFECTIVE AIR GAP (800, 1000, 1200)
    # -------------------------------------------------------------
    print("\n--- Level 2: Flow-Dependent Effective Air Gap Optimization ---")
    flow_targets = df_master['Target Flow'].unique()
    flow_gaps = {}
    flow_gaps_unc = {}
    
    for flow in flow_targets:
        df_sub = df_master[df_master['Target Flow'] == flow]
        def residual_flow(x):
            gap = x[0]
            resids = []
            for idx, row in df_sub.iterrows():
                y_exp = row['Permeate mass flow rate (g/min)']
                unc = max(row['Permeate mass flow rate uncertainty (g/min)'], 1e-4)
                out = simulate_case(row, air_gap=gap)
                resids.append((out['perm_g_min'] - y_exp) / unc)
            return np.array(resids)
        
        opt_flow = least_squares(residual_flow, x0=[0.0009], bounds=(0.00005, 0.002), method='trf')
        delta_g_flow = opt_flow.x[0]
        flow_gaps[flow] = delta_g_flow
        
        # Uncertainty Calculation for Flow-Dependent Gap
        J_flow = opt_flow.jac
        cov_flow = np.linalg.inv(J_flow.T.dot(J_flow))
        u_std_flow = np.sqrt(cov_flow[0, 0])
        dof_flow = len(df_sub) - 1
        k_flow = t.ppf(0.975, dof_flow) if dof_flow > 0 else 2.00
        U_flow_mm = (k_flow * u_std_flow) * 1000.0
        flow_gaps_unc[flow] = U_flow_mm
        
        print(f"Flow {flow}: Optimal Effective Air Gap = {delta_g_flow*1000:.4f} ± {U_flow_mm:.4f} mm")
        
    res_flow_opt = []
    for idx, row in df_master.iterrows():
        opt_gap = flow_gaps[row['Target Flow']]
        out = simulate_case(row, air_gap=opt_gap)
        res_flow_opt.append(out)
    df_flow_opt = pd.DataFrame(res_flow_opt)
    
    # -------------------------------------------------------------
    # 4. COMPILE MASTER COMPARISON TABLE
    # -------------------------------------------------------------
    df_master['Sim Permeate Default (g/min)'] = df_default['perm_g_min']
    df_master['Sim Permeate Global LM (g/min)'] = df_global['perm_g_min']
    df_master['Sim Permeate Flow LM (g/min)'] = df_flow_opt['perm_g_min']
    
    df_master['Sim Tf_out Default (°C)'] = df_default['T_f_out']
    df_master['Sim Tc_out Default (°C)'] = df_default['T_c_out']
    df_master['Sim Tf_out Flow LM (°C)'] = df_flow_opt['T_f_out']
    df_master['Sim Tc_out Flow LM (°C)'] = df_flow_opt['T_c_out']
    
    # Percentage Errors
    df_master['Rel Error Default (%)'] = 100.0 * (df_master['Sim Permeate Default (g/min)'] - df_master['Permeate mass flow rate (g/min)']) / df_master['Permeate mass flow rate (g/min)']
    df_master['Rel Error Flow LM (%)'] = 100.0 * (df_master['Sim Permeate Flow LM (g/min)'] - df_master['Permeate mass flow rate (g/min)']) / df_master['Permeate mass flow rate (g/min)']
    
    df_master.to_csv('results/inverse_problem_results.csv', index=False)
    
    # -------------------------------------------------------------
    # 5. COMPREHENSIVE PERFORMANCE METRICS (RMSE, MAE, MAPE, R2)
    # -------------------------------------------------------------
    def get_metrics(y_true, y_pred):
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        mae = np.mean(np.abs(y_pred - y_true))
        mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100.0
        max_err = np.max(np.abs(y_pred - y_true))
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1.0 - (ss_res / ss_tot)
        return rmse, mae, mape, max_err, r2

    y_exp = df_master['Permeate mass flow rate (g/min)'].values
    
    m_def = get_metrics(y_exp, df_master['Sim Permeate Default (g/min)'].values)
    m_glob = get_metrics(y_exp, df_master['Sim Permeate Global LM (g/min)'].values)
    m_flow = get_metrics(y_exp, df_master['Sim Permeate Flow LM (g/min)'].values)
    
    metrics_summary = pd.DataFrame([
        {'Model Configuration': 'Default Nominal Parameters (2.00 mm gap)', 'RMSE (g/min)': m_def[0], 'MAE (g/min)': m_def[1], 'MAPE (%)': m_def[2], 'Max Error (g/min)': m_def[3], 'R²': m_def[4]},
        {'Model Configuration': f'Global LM Calibrated Gap ({delta_g_global*1000:.3f} mm)', 'RMSE (g/min)': m_glob[0], 'MAE (g/min)': m_glob[1], 'MAPE (%)': m_glob[2], 'Max Error (g/min)': m_glob[3], 'R²': m_glob[4]},
        {'Model Configuration': 'Flow-Dependent LM Calibrated Gaps', 'RMSE (g/min)': m_flow[0], 'MAE (g/min)': m_flow[1], 'MAPE (%)': m_flow[2], 'Max Error (g/min)': m_flow[3], 'R²': m_flow[4]}
    ])
    
    metrics_summary.to_csv('results/model_performance_metrics.csv', index=False)
    
    # LaTeX Metrics Table
    latex_metrics = metrics_summary.to_latex(
        index=False,
        float_format="%.4f",
        caption="Statistical Comparison of Model Performance for Permeate Flow Rate Estimation across Experimental Runs",
        label="tab:model_performance"
    )
    with open('results/model_performance_metrics.tex', 'w') as f:
        f.write(latex_metrics)
        
    print("\n=== MODEL PERFORMANCE METRICS SUMMARY ===")
    print(metrics_summary.to_string(index=False))
    
    # Save Flow Gap summary with uncertainties
    flow_gap_df = pd.DataFrame([
        {
            'Target Flow': k, 
            'Nominal Gap (mm)': 2.00, 
            'Effective Gap LM (mm)': v*1000.0,
            'Expanded Uncertainty (± mm)': flow_gaps_unc[k],
            'Reduction (%)': 100.0*(2.00 - v*1000.0)/2.00
        }
        for k, v in flow_gaps.items()
    ])
    flow_gap_df.to_csv('results/flow_dependent_air_gaps.csv', index=False)
    print("\n=== FLOW-DEPENDENT EFFECTIVE AIR GAPS ===")
    print(flow_gap_df.to_string(index=False))
    
    print("\n=== INVERSE PROBLEM ESTIMATION FINISHED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()