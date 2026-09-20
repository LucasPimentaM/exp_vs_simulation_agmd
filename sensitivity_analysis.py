"""
Formal Sensitivity Analysis for Vacuum-Enhanced Air Gap Membrane Distillation (V-AGMD)
Master's Thesis / International Publication Standards
"""

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Thermophysical density of seawater / brine
def rho(temperature, salinity):
    a = [9.999e2, 2.034e-2, -6.162e-3, 2.261e-5, -4.657e-8]
    b = [8.020e2, -2.001, 1.677e-2, -3.060e-5, -1.613e-5]
    t = temperature
    t_part = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4
    s_part = b[0]*salinity + b[1]*salinity*t + b[2]*salinity*t**2 + b[3]*salinity*t**3 + b[4]*(salinity**2)*(t**2)
    return t_part + s_part

def run_simulation(p):
    """Executes vagmd0Dmodel binary and retrieves outputs in g/min."""
    cmd = ['./bin/vagmd0Dmodel']
    for k, v in p.items():
        cmd.extend([f'-{k}', str(v)])
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"C simulation failed: {res.stderr}")
    
    report = pd.read_csv('./results/report.csv', header=None)
    
    # Read permeate in g/min directly or compute from mass flux
    try:
        perm_row = report[report[0] == 'Permeate flow rate =']
        if len(perm_row) > 0:
            perm_g_min = float(perm_row[1].values[0])
        else:
            mass_flux = float(report[report[0] == 'Mass flux ='][1].values[0])
            perm_g_min = mass_flux * float(p.get('membrane_area', 0.007853975)) * 1000.0 / 60.0
    except Exception:
        mass_flux = float(report[report[0] == 'Mass flux ='][1].values[0])
        perm_g_min = mass_flux * float(p.get('membrane_area', 0.007853975)) * 1000.0 / 60.0
        
    T_f_out = float(report[report[0] == 'Feed temperature at the outlet of the module ='][1].values[0])
    T_c_out = float(report[report[0] == 'Coolant temperature at the outlet of the module ='][1].values[0])
    heat_flux = float(report[report[0] == 'Heat flux ='][1].values[0])
    
    return {
        'perm_g_min': perm_g_min,
        'T_f_out': T_f_out,
        'T_c_out': T_c_out,
        'heat_flux': heat_flux
    }

def main():
    print("=== STARTING FORMAL PARAMETRIC SENSITIVITY ANALYSIS ===")
    
    # Load consolidated dataset
    df_exp = pd.read_csv('Experimento_bancada_geral.csv')
    
    # Reference baseline condition (mean over all 18 real experimental points)
    mean_flow_mL = df_exp['Real Flow (mL/min)'].mean()
    mean_T1 = df_exp['Feed temperature at the inlet (°C)'].mean()
    mean_T4 = df_exp['Coolant temperature at the inlet (°C)'].mean()
    mean_P2 = df_exp['Vacuum pressure (Pa)'].mean()
    
    dens_f = rho(mean_T1, 0.0003)
    dens_c = rho(mean_T4, 0.0003)
    m_dot_f = dens_f * (mean_flow_mL * 1e-6 / 60.0)
    m_dot_c = dens_c * (mean_flow_mL * 1e-6 / 60.0)
    
    base_params = {
        'membrane_area': 0.007853975,
        'membrane_thickness': 0.00015,
        'membrane_porosity': 0.7,
        'pore_diameter': 3.5e-7,
        'polymer_conductivity': 0.3,
        'feed_channel_height': 0.002,
        'cold_channel_height': 0.002,
        'channel_width': 0.0886226551170749,
        'spacer_porosity': 1.0,
        'gap_spacer_porosity': 0.636620310097753,
        'air_gap_thickness': 0.002,
        'wall_thickness': 0.002,
        'spacer_conductivity': 0.27,
        'wall_conductivity': 18.0,
        'number_channels': 1,
        'feed_mass_flow_rate': m_dot_f,
        'cool_mass_flow_rate': m_dot_c,
        'entry_temperature_feed': mean_T1,
        'entry_temperature_cool': mean_T4,
        'vacuum_pressure': mean_P2,
        'entry_salinity_feed': 0.0003,
        'entry_salinity_cool': 0.0003
    }
    
    base_out = run_simulation(base_params)
    J0 = base_out['perm_g_min']
    print(f"Base Permeate Flow Rate at Mean Operating Conditions: {J0:.4f} g/min")
    
    param_meta = [
        ('entry_temperature_feed', r'Feed inlet temperature ($T_{f,in}$)', '°C', 'Operating'),
        ('entry_temperature_cool', r'Coolant inlet temperature ($T_{c,in}$)', '°C', 'Operating'),
        ('air_gap_thickness', r'Air gap thickness ($\delta_g$)', 'm', 'Apparatus / Gap'),
        ('membrane_porosity', r'Membrane porosity ($\varepsilon_m$)', '-', 'Membrane'),
        ('membrane_thickness', r'Membrane thickness ($\delta_m$)', 'm', 'Membrane'),
        ('pore_diameter', r'Mean pore diameter ($d_p$)', 'm', 'Membrane'),
        ('gap_spacer_porosity', r'Gap spacer porosity ($\varepsilon_g$)', '-', 'Apparatus / Gap'),
        ('feed_channel_height', r'Feed channel height ($h_f$)', 'm', 'Apparatus / Module'),
        ('feed_mass_flow_rate', r'Feed mass flow rate ($\dot{m}_f$)', 'kg/s', 'Operating'),
        ('spacer_conductivity', r'Spacer thermal conductivity ($k_s$)', 'W/(m·K)', 'Apparatus / Spacer'),
        ('cold_channel_height', r'Coolant channel height ($h_c$)', 'm', 'Apparatus / Module'),
        ('cool_mass_flow_rate', r'Coolant mass flow rate ($\dot{m}_c$)', 'kg/s', 'Operating'),
        ('wall_conductivity', r'Condensing wall conductivity ($k_w$)', 'W/(m·K)', 'Apparatus / Wall'),
        ('wall_thickness', r'Condensing wall thickness ($\delta_w$)', 'm', 'Apparatus / Wall'),
        ('vacuum_pressure', r'Vacuum pressure ($P_{vac}$)', 'Pa', 'Operating'),
        ('polymer_conductivity', r'Polymer conductivity ($k_{poly}$)', 'W/(m·K)', 'Membrane')
    ]
    
    # 1. Local Elasticity (Sensitivity Coefficients)
    sensitivity_rows = []
    
    for key, label, unit, category in param_meta:
        val0 = base_params[key]
        if key == 'vacuum_pressure':
            delta = 100.0 # 100 Pa perturbation
        elif val0 != 0:
            delta = 0.02 * abs(val0) # 2% step for central finite difference
        else:
            delta = 0.01
            
        p_plus = base_params.copy()
        p_plus[key] = val0 + delta
        p_minus = base_params.copy()
        p_minus[key] = val0 - delta
        
        out_plus = run_simulation(p_plus)
        out_minus = run_simulation(p_minus)
        
        dJ_dp = (out_plus['perm_g_min'] - out_minus['perm_g_min']) / (2.0 * delta)
        
        if val0 != 0:
            elasticity = (val0 / J0) * dJ_dp
        else:
            elasticity = (delta / J0) * dJ_dp
            
        sensitivity_rows.append({
            'Parameter Key': key,
            'Parameter Label': label,
            'Category': category,
            'Nominal Value': val0,
            'Unit': unit,
            'dJ/dp (g/min/unit)': dJ_dp,
            'Elasticity (Dimensionless)': elasticity,
            'Abs Elasticity': abs(elasticity)
        })
        
    df_sens = pd.DataFrame(sensitivity_rows).sort_values(by='Abs Elasticity', ascending=False).reset_index(drop=True)
    df_sens['Rank'] = df_sens.index + 1
    
    # Save sensitivity table
    df_sens.to_csv('results/sensitivity_ranking.csv', index=False)
    
    # Export LaTeX Table
    latex_table = df_sens[['Rank', 'Parameter Label', 'Category', 'Nominal Value', 'Unit', 'Elasticity (Dimensionless)']].to_latex(
        index=False,
        escape=False,
        float_format="%.4e",
        caption="Dimensionless Parametric Sensitivity Coefficients (Elasticities) for Permeate Flow Rate in AGMD Model",
        label="tab:sensitivity_ranking"
    )
    with open('results/sensitivity_ranking.tex', 'w') as f:
        f.write(latex_table)
        
    print("\n--- SENSITIVITY RANKING (TOP TO BOTTOM) ---")
    print(df_sens[['Rank', 'Parameter Key', 'Category', 'Elasticity (Dimensionless)']].to_string(index=False))
    
    # 2. Generate Publication-Quality Tornado Diagram (600 DPI)
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=600)
    
    df_plot = df_sens.sort_values(by='Elasticity (Dimensionless)', ascending=True).reset_index(drop=True)
    
    y_pos = np.arange(len(df_plot))
    colors = ['#d95f02' if x < 0 else '#1b9e77' for x in df_plot['Elasticity (Dimensionless)']]
    
    bars = ax.barh(y_pos, df_plot['Elasticity (Dimensionless)'], color=colors, height=0.65, edgecolor='black', linewidth=0.8, alpha=0.9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['Parameter Label'])
    ax.axvline(0, color='black', linewidth=1.0, linestyle='-')
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    ax.set_xlabel(r'Dimensionless Elasticity $\left( E_{\theta} = \frac{\theta_0}{\dot{m}_{\mathrm{perm},0}} \frac{\partial \dot{m}_{\mathrm{perm}}}{\partial \theta} \right)$')
    ax.set_title('Parametric Sensitivity of Permeate Flow Rate in AGMD Model', fontweight='bold', pad=12)
    
    # Annotate values
    for bar, val in zip(bars, df_plot['Elasticity (Dimensionless)']):
        if val >= 0:
            ax.text(val + 0.08, bar.get_y() + bar.get_height()/2.0, f'{val:+.3f}', 
                    va='center', ha='left', fontsize=9, fontweight='bold', color='#116644')
        else:
            ax.text(val - 0.08, bar.get_y() + bar.get_height()/2.0, f'{val:+.3f}', 
                    va='center', ha='right', fontsize=9, fontweight='bold', color='#993300')
            
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    
    plt.tight_layout()
    fig.savefig('figures/tornado_sensitivity.png', dpi=600, bbox_inches='tight')
    fig.savefig('figures/tornado_sensitivity.svg', bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/tornado_sensitivity.png and .svg")
    
    # 3. Parametric Perturbation Response Curves (-50% to +50%)
    fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=600)
    
    key_params_curve = [
        ('air_gap_thickness', r'Air gap thickness ($\delta_g$)', '#e41a1c', '-'),
        ('membrane_porosity', r'Membrane porosity ($\varepsilon_m$)', '#377eb8', '--'),
        ('membrane_thickness', r'Membrane thickness ($\delta_m$)', '#4daf4a', '-.'),
        ('entry_temperature_feed', r'Feed inlet temp ($T_{f,in}$)', '#984ea3', ':'),
        ('pore_diameter', r'Pore diameter ($d_p$)', '#ff7f00', '-')
    ]
    
    delta_percentages = np.linspace(-40, 40, 17) # -40% to +40%
    
    for key, label, color, lstyle in key_params_curve:
        val0 = base_params[key]
        response_perm = []
        for pct in delta_percentages:
            p_test = base_params.copy()
            val_perturbed = val0 * (1.0 + pct / 100.0)
            if key == 'membrane_porosity':
                val_perturbed = min(max(val_perturbed, 0.1), 0.95)
            p_test[key] = val_perturbed
            try:
                out = run_simulation(p_test)
                response_perm.append(out['perm_g_min'])
            except Exception:
                response_perm.append(np.nan)
                
        # Normalized response J / J0
        norm_response = np.array(response_perm) / J0
        ax.plot(delta_percentages, norm_response, label=label, color=color, linestyle=lstyle, linewidth=2.0, marker='o', markersize=4)
        
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.0)
    ax.axvline(0.0, color='gray', linestyle=':', linewidth=1.0)
    ax.set_xlabel('Parameter Variation (%)')
    ax.set_ylabel(r'Normalized Permeate Rate $\left( \dot{m}_{\mathrm{perm}} / \dot{m}_{\mathrm{perm},0} \right)$')
    ax.set_title('Nonlinear Permeate Response to Critical Parameter Perturbations', fontweight='bold', pad=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig('figures/parameter_perturbation_curves.png', dpi=600, bbox_inches='tight')
    fig.savefig('figures/parameter_perturbation_curves.svg', bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/parameter_perturbation_curves.png and .svg")
    
    print("=== SENSITIVITY ANALYSIS COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()
