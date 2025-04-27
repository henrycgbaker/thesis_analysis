import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt

from scripts.a_data_loading_cleaning import (
    df_controlled_cleaned,
)

# --- Ensure that key variables are converted to appropriate types ---
df_controlled_cleaned['batch_size_numeric'] = pd.to_numeric(df_controlled_cleaned['batch_size___fixed_batching'], errors='coerce')
df_controlled_cleaned['num_processes'] = pd.to_numeric(df_controlled_cleaned['num_processes'], errors='coerce')

# Convert precision and quantization to categorical types.
df_controlled_cleaned['fp_precision'] = df_controlled_cleaned['fp_precision'].astype('category')
df_controlled_cleaned['quantization'] = df_controlled_cleaned['quantization'].astype('category')

# --- Choose a grouping variable ---
# For example, if experiment_id is repeated across runs, you can use that; alternatively, use config_name if that makes sense.
# Here, we use 'config_name' as the grouping variable:
group_var = "config_name"  
if group_var not in df_controlled_cleaned.columns:
    # Fallback: use experiment_id if available.
    group_var = "experiment_id"

# --- Define the mixed effects model formula ---
# The model assumes a random intercept for each group.
formula = "energy_per_token_kwh ~ batch_size_numeric + num_processes + C(fp_precision) + C(quantization)"

# Fit the MixedLM model using Maximum Likelihood (set reml=False for ML estimation).
mixed_model = smf.mixedlm(formula, 
                          data=df_controlled_cleaned, 
                          groups=df_controlled_cleaned[group_var])
mixed_model_fit = mixed_model.fit(reml=False)

# Print the summary of the mixed effects model.
print(mixed_model_fit.summary())

# --- Optional: Build a Mixed Model for the Divergence Metric ---
# For example, you can model 'divergence_energy_flops' similarly:
formula_div = "divergence_energy_flops ~ batch_size_numeric + num_processes + C(fp_precision) + C(quantization)"
mixed_model_div = smf.mixedlm(formula_div, 
                              data=df_controlled_cleaned, 
                              groups=df_controlled_cleaned[group_var])
mixed_model_div_fit = mixed_model_div.fit(reml=False)
print(mixed_model_div_fit.summary())

# --- Diagnostic Plot: Fitted Values vs. Residuals for the Mixed Model ---
plt.figure(figsize=(10, 6))
plt.scatter(mixed_model_fit.fittedvalues, mixed_model_fit.resid, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Energy per Token (kWh)')
plt.ylabel('Residuals')
plt.title('Mixed Effects Model: Fitted Values vs. Residuals')
plt.grid(True)
plt.show()
