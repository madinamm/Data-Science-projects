# Stock Index Predictability Analysis Using Machine Learning

## Project Motivation
The purpose of this project is to identify which stock indices are the most predictable using machine learning (ML) models. By determining the most predictable stock index, we aim to enhance profitability in investment decisions and reduce associated risks. This analysis focuses on predicting weekly closing prices of selected stock indices and evaluating their predictability through various accuracy metrics.

## Libraries Used
This project utilizes the following libraries:
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib` and `seaborn`: For data visualization.
- `statsmodels`: For statistical modeling and time series analysis.
- `scikit-learn`: For machine learning model evaluation and selection.
- `lightgbm` and `xgboost`: For gradient boosting models.

## Repository Structure
- `data/Historical_Data.csv`: Contains historical weekly closing prices for selected stock indices from January 2000 to February 2025.
- `Stock_Indices.ipynb`: Notebook for data analysis, model training, and evaluation.

## Data Source
Data was sourced from [Investing.com](https://www.investing.com/indices/major-indices), which provides live updates for major world indices. The dataset includes 10 major indices, such as S&P 500 and Dow Jones.

## Methodology
1. **Exploratory Data Analysis (EDA)**: Conducted EDA to select 3 indices from a pool of 10.
2. **Data Preparation**: Weekly closing prices from January 2000 to February 2025 were used for training and testing the models.
3. **Modeling**: Implemented various machine learning models including ARIMA, LightGBM, XGBoost, and Random Forest to predict closing prices.
4. **Evaluation**: Employed four accuracy metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Weighted Mean Absolute Percentage Error (WMAPE), and R² score to assess model performance.

## Results
After thorough analysis and model evaluation, the final selected indices were S&P 500, Dow Jones, and FTSE China A50. Among these, FTSE China A50 emerged as the most predictable index based on the accuracy metrics used.

## Predictions
We generated predictions for the upcoming 10 weeks using the LightGBM model, which performed best according to the metrics for FTSE China A50.

## Acknowledgements
Data was sourced from [Investing.com](https://www.investing.com/indices/major-indices), which provides live updates for major world indices. This Analysis was made possible thanks to the available data and a content of Udacity Nanodegree program.

## Conclusion
This project highlights the potential for using machine learning in financial forecasting, particularly in identifying predictable stock indices. The insights gained can assist investors in making informed decisions and optimizing their investment strategies.

## Blog post
Relevant Blog post can be found here https://medium.com/@mmadeyeva/stock-indices-prediction-a-machine-learning-approach-97122565242a
