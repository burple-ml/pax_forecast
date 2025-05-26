""" fitting l2 regression model"""
from preprocessing import preprocessing, IMG_FOLDER
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os

def l2_regression(X, y):
    # Grid search space
    alphas = np.logspace(-3, 3, 20)

    # Outer 5-fold CV
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_results = []

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
        print(f"\n=== Outer Fold {i} ===")

        # still outer split
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])

        # Inner CV with grid search
        param_grid = {'ridge__alpha': alphas}
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(pipeline, param_grid, cv=inner_cv,
                            scoring='neg_mean_squared_error', return_train_score=True)
        grid.fit(X_train, y_train)

        mean_mse = -grid.cv_results_['mean_test_score']
        std_mse = grid.cv_results_['std_test_score']
        best_alpha = grid.best_params_['ridge__alpha']

        # Plot
        plt.figure(figsize=(8, 5))
        plt.semilogx(alphas, mean_mse, marker='o', label='Mean CV MSE')
        plt.fill_between(alphas, mean_mse - std_mse, mean_mse + std_mse, alpha=0.2)
        plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best λ = {best_alpha:.4f}')
        plt.title(f'Inner 5-Fold CV: MSE vs Lambda (Outer Fold {i})')
        plt.xlabel('Lambda (Alpha)')
        plt.ylabel('Mean CV MSE')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_FOLDER, f"hyperparameter-ridgeregression-fold-{i}.png"))
        plt.show()

        print(f"Best lambda for Outer Fold {i}: {best_alpha:.4f}")

        # Train with best lambda
        final_model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=best_alpha))
        ])
        final_model.fit(X_train, y_train)
        
        # Evaluate on unused test
        y_pred = final_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        accuracy_150 = np.mean(np.abs(y_test - y_pred) <= 150)

        print(f"Evaluation on Outer Test Set:")
        print(f"  MSE       = {mse:.2f}")
        print(f"  R² Score  = {r2:.4f}")
        print(f"  Accuracy within ±150 passengers = {accuracy_150*100:.2f}%")

        outer_results.append({
            'fold': i,
            'best_lambda': best_alpha,
            'mse': mse,
            'r2': r2,
            'accuracy_100': accuracy_150
        })

    # unbiased estimates
    mse_vals = np.array([res['mse'] for res in outer_results])
    r2_vals = np.array([res['r2'] for res in outer_results])
    acc_vals = np.array([res['accuracy_100'] for res in outer_results])

    print("\n=== Final Cross-Validated Generalization Metrics ===")
    print(f"Mean MSE       : {mse_vals.mean():.2f}")
    print(f"Mean R² Score  : {r2_vals.mean():.4f}")
    print(f"Mean Accuracy (±150 passengers): {acc_vals.mean() * 100:.2f}%")



if __name__ == '__main__':
    df = preprocessing()
    df.drop('Date', axis=1, inplace=True)
    print(df,"\n" , df.columns)
    print(df.dtypes)
    X = df.drop('PAX', axis=1)
    Y = df['PAX']
    print(X,Y)
    l2_regression(X, Y)