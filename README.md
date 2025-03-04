# House Price Prediction from Scratch & Scikit-Learn

## Overview
This project implements a machine learning model to predict house prices using both **scratch implementations (NumPy, Pandas, Matplotlib, Seaborn)** and **scikit-learn**. The objective is to explore key concepts in regression analysis, learning rate optimization, bias-variance tradeoff, and regularization techniques while comparing handcrafted models with industry-standard implementations.

## Project Workflow
The project follows a step-by-step approach:

1. **Data Preprocessing & Exploratory Data Analysis (EDA)**
   - Cleaned and preprocessed the dataset.
   - Performed exploratory data analysis to understand feature distributions and relationships.
   - Identified `sqft_living` as the major contributor to house prices.

2. **Simple Linear Regression**
   - Implemented using one feature (`sqft_living`).
   - Applied gradient descent for optimization.
   - Visualized the effect of different learning rates on cost (Mean Squared Error - MSE) and \( R^2 \) score.
   - Compared performance with **scikit-learn’s LinearRegression**.

3. **Multiple Linear Regression**
   - Extended the model to multiple features.
   - Implemented gradient descent for optimization.
   - Analyzed the performance improvement compared to simple linear regression.
   - Benchmarked against **scikit-learn’s LinearRegression**.

4. **Polynomial Regression**
   - Explored the impact of different polynomial degrees on model performance.
   - Visualized the tradeoff between underfitting and overfitting.
   - Compared models based on MSE and \( R^2 \) score.
   - Used **scikit-learn’s PolynomialFeatures** for comparison.

5. **Bias-Variance Tradeoff**
   - Analyzed how increasing polynomial degree affects the model.
   - Identified the optimal polynomial degree for best generalization.

6. **Regularization Techniques (L1 & L2)**
   - Implemented Lasso (L1) and Ridge (L2) regularization.
   - Explored the effect of different values of the regularization parameter (lambda) on model performance.
   - Compared custom implementations with **scikit-learn’s Ridge and Lasso**.
   - **Regularization did not improve the model's performance.**

## Key Findings
- `sqft_living` is the major contributor to house prices.
- Learning rate significantly impacts model convergence.
- Higher-degree polynomials tend to overfit, while lower-degree models underfit.
- The best-performing model is a **quadratic polynomial regression (degree = 2)**.
- Regularization did not improve performance in this case.
- **Scikit-learn models achieved similar results but were more computationally efficient.**

### Final Model Performance
- **Train \( R^2 \) for degree 2 (scratch):** 0.7366
- **Test \( R^2 \) for degree 2 (scratch):** 0.7432
- **Scikit-learn Polynomial Regression (degree 2) performs better with** R-squared (Training): 0.8346
R-squared (Test): 0.7980


## Dependencies
Ensure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
Clone the repository and run the Jupyter Notebook or Python script:

```bash
git clone <repository_url>
cd <repository>
python main.py  # If implemented as a script
```

Or open the notebook in Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Structure
```
|-- data/                # Dataset files
|-- notebooks/           # Jupyter Notebooks
|-- scripts/             # Python scripts for models
|-- results/             # Plots and visualizations
|-- README.md            # Project documentation
```

## Future Improvements
- Implement cross-validation to select the best model parameters.
- Extend the dataset with more features for better predictions.
- Optimize custom implementations for computational efficiency.
- Explore additional regularization techniques like ElasticNet.

## Author
[Habibullah](https://github.com/Habibu-Ahmad)


## License
This project is open-source and available under the MIT License.

