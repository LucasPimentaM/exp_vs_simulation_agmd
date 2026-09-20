import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Abstract
cells.append(nbf.v4.new_markdown_cell("""# Vacuum-Enhanced Air Gap Membrane Distillation (V-AGMD)
## Experimental Validation, Parametric Sensitivity Analysis, and Inverse Problem Estimation (Levenberg-Marquardt)

This notebook consolidates the entire analytical workflow:
1. **Master Experimental Dataset**: Loading real measured flow rates, inlet temperatures ($T_1, T_4$), vacuum pressure ($P_2$), and expanded uncertainties ($u_c$).
2. **C Numerical Solver Execution**: Interfacing with the high-performance PETSc C model (`vagmd0Dmodel`) with native permeate mass flow rate output in **g/min**.
3. **Formal Parametric Sensitivity Analysis**: Computing dimensionless relative elasticities ($E_\\theta$), ranking critical parameters, and generating publication Tornado diagrams.
4. **Inverse Problem Resolution via Levenberg-Marquardt**: Quantifying hydrodynamically and thermally induced membrane deflection by estimating the effective air gap ($\\delta_{g,\\mathrm{eff}}$).
5. **Publication Figures (600 DPI, English)**: Generating parity plots, temperature validation, and parametric curves.
"""))

# Dependencies
cells.append(nbf.v4.new_code_cell("""import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from IPython.display import Image, display

# Publication plot styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
"""))

# Section 1: Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 1. Loading Master Experimental Dataset (`Experimento_bancada_geral.csv`)"""))

cells.append(nbf.v4.new_code_cell("""df_master = pd.read_csv('Experimento_bancada_geral.csv')
print(f'Total Experimental Cases: {len(df_master)}')
display(df_master[['Case', 'Real Flow (mL/min)', 'Feed temperature at the inlet (°C)', 'Coolant temperature at the inlet (°C)', 'Vacuum pressure (Pa)', 'Permeate mass flow rate (g/min)']].head(6))
"""))

# Section 2: Sensitivity Analysis
cells.append(nbf.v4.new_markdown_cell("""## 2. Parametric Sensitivity Ranking Table & Elasticity Analysis"""))

cells.append(nbf.v4.new_code_cell("""df_sens = pd.read_csv('results/sensitivity_ranking.csv')
display(df_sens[['Rank', 'Parameter Label', 'Category', 'Nominal Value', 'Unit', 'Elasticity (Dimensionless)']])
"""))

cells.append(nbf.v4.new_code_cell("""# Display Generated Tornado Diagram and Perturbation Curves
display(Image(filename='figures/tornado_sensitivity.png', width=700))
display(Image(filename='figures/parameter_perturbation_curves.png', width=700))
"""))

# Section 3: Inverse Problem & Performance
cells.append(nbf.v4.new_markdown_cell("""## 3. Levenberg-Marquardt Inverse Problem & Model Performance Metrics"""))

cells.append(nbf.v4.new_code_cell("""df_metrics = pd.read_csv('results/model_performance_metrics.csv')
display(df_metrics)
"""))

cells.append(nbf.v4.new_code_cell("""df_flow_gaps = pd.read_csv('results/flow_dependent_air_gaps.csv')
display(df_flow_gaps)
"""))

# Section 4: Publication Figures
cells.append(nbf.v4.new_markdown_cell("""## 4. Master Publication Figures (600 DPI)"""))

cells.append(nbf.v4.new_code_cell("""# 1. Permeate Parity Plot (Default vs Calibrated)
display(Image(filename='figures/fig1_parity_permeate_flow_rate.png', width=800))
"""))

cells.append(nbf.v4.new_code_cell("""# 2. Outlet Temperatures Parity Plot
display(Image(filename='figures/fig2_parity_outlet_temperatures.png', width=550))
"""))

cells.append(nbf.v4.new_code_cell("""# 3. Permeate Rate vs Real Feed Temperature Curves
display(Image(filename='figures/fig3_permeate_vs_temperature_flow_curves.png', width=650))
"""))

cells.append(nbf.v4.new_code_cell("""# 4. Inferred Effective Air Gap Deflection Profile
display(Image(filename='figures/fig4_effective_air_gap_deflection.png', width=650))
"""))

cells.append(nbf.v4.new_code_cell("""# 5. Composite 4-Panel Master Validation Figure
display(Image(filename='figures/fig5_master_validation_panel.png', width=900))
"""))

nb.cells = cells

with open('runs.ipynb', 'w') as f:
    nbf.write(nb, f)

print('Successfully updated runs.ipynb!')
