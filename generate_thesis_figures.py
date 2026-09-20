"""
Master's Thesis / International Journal Publication Figures Generator (Vector PDF, English)
Vacuum-Enhanced Air Gap Membrane Distillation (V-AGMD)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure Publication-Grade Aesthetics
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.titlesize'] = 14

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load master results
df_master = pd.read_csv('results/inverse_problem_results.csv')

# Define distinct styling for each flow rate using Low, Medium, High
flow_styles = {
    '800 ml/min': {'color': '#1f77b4', 'marker': 'o', 'label': 'Low Flow'},
    '1000 ml/min': {'color': '#2ca02c', 'marker': 's', 'label': 'Medium Flow'},
    '1200 ml/min': {'color': '#d62728', 'marker': '^', 'label': 'High Flow'}
}

# ==============================================================================
# FIGURE 1A: PARITY PLOT - DEFAULT NOMINAL PARAMETERS (2.00 mm)
# ==============================================================================
def plot_fig1a_parity_permeate_default():
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    
    y_exp = df_master['Permeate mass flow rate (g/min)']
    
    # Range setup
    min_val = 0.2
    max_val = 2.4
    line_x = np.linspace(min_val, max_val, 100)
    
    ax.plot(line_x, line_x, 'k-', linewidth=1.2, label='Parity line')
    ax.plot(line_x, line_x * 1.20, 'k--', linewidth=0.9, alpha=0.7, label='+/- 20% Error bound')
    ax.plot(line_x, line_x * 0.80, 'k--', linewidth=0.9, alpha=0.7)
    
    for flow, style in flow_styles.items():
        sub = df_master[df_master['Target Flow'] == flow]
        ax.errorbar(
            sub['Sim Permeate Default (g/min)'],
            sub['Permeate mass flow rate (g/min)'],
            yerr=sub['Permeate mass flow rate uncertainty (g/min)'],
            fmt=style['marker'],
            color=style['color'],
            ecolor=style['color'],
            capsize=3.0,
            markersize=6.5,
            markeredgecolor='black',
            markeredgewidth=0.8,
            label=style['label']
        )
        
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel('Numerical Permeate Flow Rate [g/min]')
    ax.set_ylabel('Experimental Permeate Flow Rate [g/min]')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig('figures/fig1a_parity_permeate_default.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/fig1a_parity_permeate_default.pdf")

# ==============================================================================
# FIGURE 1B: PARITY PLOT - CALIBRATED MODEL (Flow-Dependent Air Gap LM)
# ==============================================================================
def plot_fig1b_parity_permeate_calibrated():
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    
    y_exp = df_master['Permeate mass flow rate (g/min)']
    
    # Range setup
    min_val = 0.2
    max_val = 2.4
    line_x = np.linspace(min_val, max_val, 100)
    
    ax.plot(line_x, line_x, 'k-', linewidth=1.2, label='Parity line')
    ax.plot(line_x, line_x * 1.20, 'k--', linewidth=0.9, alpha=0.7, label='+/- 20% Error bound')
    ax.plot(line_x, line_x * 0.80, 'k--', linewidth=0.9, alpha=0.7)
    
    for flow, style in flow_styles.items():
        sub = df_master[df_master['Target Flow'] == flow]
        ax.errorbar(
            sub['Sim Permeate Flow LM (g/min)'],
            sub['Permeate mass flow rate (g/min)'],
            yerr=sub['Permeate mass flow rate uncertainty (g/min)'],
            fmt=style['marker'],
            color=style['color'],
            ecolor=style['color'],
            capsize=3.0,
            markersize=6.5,
            markeredgecolor='black',
            markeredgewidth=0.8,
            label=style['label']
        )
        
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel('Numerical Permeate Flow Rate [g/min]')
    ax.set_ylabel('Experimental Permeate Flow Rate [g/min]')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig('figures/fig1b_parity_permeate_calibrated.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/fig1b_parity_permeate_calibrated.pdf")

# ==============================================================================
# FIGURE 2: PARITY PLOT - OUTLET TEMPERATURES
# ==============================================================================
def plot_fig2_parity_temperatures():
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    
    T_min = 18.0
    T_max = 85.0
    line_T = np.linspace(T_min, T_max, 100)
    
    ax.plot(line_T, line_T, 'k-', linewidth=1.2, label='Parity line')
    ax.plot(line_T, line_T + 1.0, 'k:', linewidth=0.8, alpha=0.6, label='+/- 1.0 °C Error bound')
    ax.plot(line_T, line_T - 1.0, 'k:', linewidth=0.8, alpha=0.6)
    
    # Feed outlet temperature
    ax.errorbar(
        df_master['Sim Tf_out Flow LM (°C)'],
        df_master['Feed temperature at the outlet (°C)'],
        yerr=df_master['Feed temperature at the outlet uncertainty (°C)'],
        fmt='o',
        color='#d95f02',
        ecolor='#d95f02',
        capsize=3.0,
        markersize=6.5,
        markeredgecolor='black',
        markeredgewidth=0.8,
        label='Feed outlet temperature'
    )
    
    # Coolant outlet temperature
    ax.errorbar(
        df_master['Sim Tc_out Flow LM (°C)'],
        df_master['Coolant temperature at the outlet (°C)'],
        yerr=df_master['Coolant temperature at the outlet uncertainty (°C)'],
        fmt='s',
        color='#7570b3',
        ecolor='#7570b3',
        capsize=3.0,
        markersize=6.5,
        markeredgecolor='black',
        markeredgewidth=0.8,
        label='Coolant outlet temperature'
    )
    
    ax.set_xlim(T_min, T_max)
    ax.set_ylim(T_min, T_max)
    ax.set_xlabel('Numerical Temperature [°C]')
    ax.set_ylabel('Experimental Temperature [°C]')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig('figures/fig2_parity_outlet_temperatures.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/fig2_parity_outlet_temperatures.pdf")

# ==============================================================================
# FIGURE 3: PERMEATE FLOW RATE VS REAL FEED INLET TEMPERATURE
# ==============================================================================
def plot_fig3_permeate_vs_temp():
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    
    for flow, style in flow_styles.items():
        sub = df_master[df_master['Target Flow'] == flow].sort_values(by='Feed temperature at the inlet (°C)')
        
        # Experimental points with error bars
        ax.errorbar(
            sub['Feed temperature at the inlet (°C)'],
            sub['Permeate mass flow rate (g/min)'],
            yerr=sub['Permeate mass flow rate uncertainty (g/min)'],
            xerr=sub['Feed temperature at the inlet uncertainty (°C)'],
            fmt=style['marker'],
            color=style['color'],
            ecolor=style['color'],
            capsize=3.5,
            markersize=7.0,
            markeredgecolor='black',
            markeredgewidth=0.9,
            linestyle='None',
            label=f'Exp. {style["label"]}'
        )
        
        # Default simulation curve
        ax.plot(
            sub['Feed temperature at the inlet (°C)'],
            sub['Sim Permeate Default (g/min)'],
            color=style['color'],
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
            label='Def. Model' if flow == '800 ml/min' else None
        )
        
        # LM Calibrated simulation curve
        ax.plot(
            sub['Feed temperature at the inlet (°C)'],
            sub['Sim Permeate Flow LM (g/min)'],
            color=style['color'],
            linestyle='-',
            linewidth=2.0,
            label='Calib. Model' if flow == '800 ml/min' else None
        )
        
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='gray', marker='o', linestyle='None', markeredgecolor='black', label='Experimental Data'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Nominal Model (2.00 mm)'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.0, label='LM Calibrated Model')
    ]
    legend1 = ax.legend(handles=[h for h in handles if 'Exp.' in h.get_label()], loc='upper left', frameon=True, facecolor='white', framealpha=0.9, title='Operating Flows')
    legend2 = ax.legend(handles=custom_lines, loc='center left', frameon=True, facecolor='white', framealpha=0.9, title='Model Types')
    ax.add_artist(legend1)
    
    ax.set_xlabel('Real Feed Inlet Temperature [°C]')
    ax.set_ylabel('Permeate Mass Flow Rate [g/min]')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig('figures/fig3_permeate_vs_temperature_flow_curves.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/fig3_permeate_vs_temperature_flow_curves.pdf")


def main():
    print("=== GENERATING PUBLICATION FIGURES (VECTOR PDF, ENGLISH) ===")
    plot_fig1a_parity_permeate_default()
    plot_fig1b_parity_permeate_calibrated()
    plot_fig2_parity_temperatures()
    plot_fig3_permeate_vs_temp()
    print("=== ALL PUBLICATION FIGURES GENERATED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()