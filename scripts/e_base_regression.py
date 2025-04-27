import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# -----------------------------
# 1. Preprocessing
# -----------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts relevant variables to numeric or categorical types as needed.
    """
    df = df.copy()
    df['batch_size_numeric'] = pd.to_numeric(df['batch_size___fixed_batching'], errors='coerce')
    df['num_processes'] = pd.to_numeric(df['num_processes'], errors='coerce')
    df['fp_precision'] = df['fp_precision'].astype('category')
    df['quantization'] = df['quantization'].astype('category')
    return df


# -----------------------------
# 2. Run OLS regression
# -----------------------------
def run_ols_model(df: pd.DataFrame, formula: str):
    """
    Fits an OLS regression model using the specified formula.
    Returns the fitted model object.
    """
    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())
    return model


# -----------------------------
# 3. Diagnostics plot
# -----------------------------
def plot_residuals(model, x_label='Fitted Values', y_label='Residuals', title='Fitted vs Residuals'):
    """
    Plots residuals vs. fitted values for regression diagnostics.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()


# -----------------------------
# 4. Full regression workflow
# -----------------------------
def run_full_regression_analysis(df: pd.DataFrame, predictors: list = None):
    """
    Run OLS regressions on two dependent variables using given predictors.
    """
    default_predictors = ['batch_size_numeric', 'num_processes', 'C(fp_precision)', 'C(quantization)']
    predictors = predictors or default_predictors
    rhs = ' + '.join(predictors)

    print("\n📊 Running OLS regression for energy_per_token_kwh...\n")
    energy_formula = f"energy_per_token_kwh ~ {rhs}"
    model_energy = run_ols_model(df, energy_formula)

    print("\n📊 Running OLS regression for divergence_energy_flops...\n")
    divergence_formula = f"divergence_energy_flops ~ {rhs}"
    model_divergence = run_ols_model(df, divergence_formula)

    print("\n📈 Plotting residuals for energy model...")
    plot_residuals(
        model_energy,
        x_label='Fitted Energy per Token (kWh)',
        y_label='Residuals',
        title='Energy Model: Fitted Values vs Residuals'
    )


# -----------------------------
# 5. Script entry point
# -----------------------------
if __name__ == "__main__":
    csv_path = "results/controlled_results.csv"
    df = get_cleaned_df(csv_path)
    df = preprocess_data(df)
    run_full_regression_analysis(df)