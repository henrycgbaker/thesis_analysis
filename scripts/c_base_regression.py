import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

from scripts.a_data_loading_cleaning import get_cleaned_df

df = get_cleaned_df()

# --- Convert variables to appropriate types ---
# Ensure batch size and num_processes are numeric
df_controlled_cleaned['batch_size_numeric'] = pd.to_numeric(df_controlled_cleaned['batch_size___fixed_batching'], errors='coerce')
df_controlled_cleaned['num_processes'] = pd.to_numeric(df_controlled_cleaned['num_processes'], errors='coerce')

# Convert precision and quantization to categorical types.
df_controlled_cleaned['fp_precision'] = df_controlled_cleaned['fp_precision'].astype('category')
df_controlled_cleaned['quantization'] = df_controlled_cleaned['quantization'].astype('category')

# --- Define the regression formula ---
# Here the model predicts energy consumption per token based on batch size, number of processes, and the categorical variables.
formula = "energy_per_token_kwh ~ batch_size_numeric + num_processes + C(fp_precision) + C(quantization)"

# Fit the regression model using Ordinary Least Squares (OLS)
model_energy = smf.ols(formula, data=df_controlled_cleaned).fit()

# Print the regression results summary
print(model_energy.summary())

# --- Optionally, build a model on the divergence metric ---
# For instance, divergence_energy_flops as dependent variable.
formula_divergence = "divergence_energy_flops ~ batch_size_numeric + num_processes + C(fp_precision) + C(quantization)"
model_divergence = smf.ols(formula_divergence, data=df_controlled_cleaned).fit()
print(model_divergence.summary())

# --- Regression Diagnostics ---
# Plot fitted values vs. residuals for the energy model for diagnostic purposes.
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(model_energy.fittedvalues, model_energy.resid, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Energy per Token (kWh)')
plt.ylabel('Residuals')
plt.title('Fitted Values vs. Residuals')
plt.grid(True)
plt.show()
