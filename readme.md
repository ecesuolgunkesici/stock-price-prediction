<p>
  <a href="https://github.com/ecesuolgunkesici/stock-price-prediction/blob/master/README.md">
    <img alt="Documentation" src="https://img.shields.io/static/v1?label=lang&message=en&color=blue" target="_blank" />
  </a>
</p>

# stock-price-prediction
This project implements stock price prediction using *LSTM, **GRU, and **Prophet* models. The best model is selected based on the lowest *RMSE (Root Mean Squared Error)* value. The project automatically tunes hyperparameters using *Optuna* and selects the best-performing feature set.

## 📌 How to Run the Code

### 1️⃣ Install Dependencies
Make sure you have *Python 3.8+* installed. Then install the required dependencies:

bash
pip install -r requirements.txt


### 2️⃣ Run the Script
Simply execute the Python script:

bash
python3 stock_prediction.py


## :star2: Data Preprocessing
### 2.1 Data Collection
  - Data was fetched using yfinance for the stock ticker AAPL (Apple Inc.).
  - The dataset contained Open, High, Low, Close, and Volume values from 2020-01-01 to 2024-01-01.
### 2.2 Handling MultiIndex Issue
  - The downloaded data had a MultiIndex structure, which was flattened to ensure compatibility with our feature engineering pipeline.
### 2.3 Feature Engineering
  - Several technical indicators were calculated to enhance model performance:
    
  | Feature | Description |
  |---------|------------|
  | *RSI (Relative Strength Index)* | Measures momentum and overbought/oversold conditions. |
  | *MACD (Moving Average Convergence Divergence)* | Captures trend strength and direction. |
  | *Signal Line* | Smooths out MACD values to reduce noise. |
  | *Bollinger Bands (Upper & Lower)* | Measures volatility and trend reversal points. |
  | *Momentum* | Captures the rate of price change over 10 days. |
  | *ADX (Average Directional Index)* | Measures trend strength. |
  | *Volume Change* | Tracks the percentage change in trading volume. |
  | *MA Ratio (Short/Long MA)* | Compares short-term and long-term moving averages. |

  
  - Feature Selection Optimization: We used Optuna to select the best features dynamically.


## :clipboard: Model Selection and Hyperparameter Tuning
  ### 3.1 Long Short-Term Memory (LSTM)
  ### Hyperparameters
  - *Sequence Length:* 15  
  - *Dropout:* 0.3  
  - *Optimizer:* Adam  
  - *Loss Function:* MSE  
  - *Epochs:* 20  
  - *Results:* RMSE = 6.01 (not optimal)  
  
  Limitations: Struggled with volatility, slow adaptation to sudden changes.
  <p float="left">
  <img src="https://github.com/ecesuolgunkesici/stock-price-prediction/blob/main/images/lstm/lstm_1.png" alt="alt text" width="600px" height="300px">
  </p>

### Feature Engineering & Hyperparameter Tuning for LSTM
  - After feature selection and hyperparameter tuning, we retrained LSTM with:

  ### Best Hyperparameters:
  - *Sequence Length:* 10  
  - *Dropout:* 0.1
  - *Selected Features*: ['Close', 'RSI', 'MACD', 'Upper_Band', 'ADX']
  - *Optimizer:* Adam  
  - *Loss Function:* MSE  
  - *Epochs:* 20  
  - *Results:* RMSE = 4.64 (significant improvement)
  - *Observations:* Feature engineering helped improve accuracy, but LSTM still lagged in capturing sharp price movements.

  <p float="left">
  <img src="https://github.com/ecesuolgunkesici/stock-price-prediction/blob/main/images/lstm/lstm_2.png" alt="alt text" width="600px" height="300px">
  </p>

  ### 3.2 Gated Recurrent Unit (GRU)
  - We optimized GRU using Optuna for hyperparameter tuning and feature selection:
 
  ### Hyperparameters
  - *Sequence Length:* 20  
  - *Dropout:* 0.1
  - *Selected Features*: ['Close', 'RSI', 'MACD', 'Lower_Band', 'Volume_Change']
  - *Optimizer:* Adam  
  - *Loss Function:* MSE  
  - *Epochs:* 20  
  - *Results:* RMSE = 2.53 (Best model)
  - *Observations:* - GRU provided a better fit compared to LSTM, adapting well to volatility and lower RMSE indicated improved generalization.

    <p float="left">
    <img src="https://github.com/ecesuolgunkesici/stock-price-prediction/blob/main/images/gru/gru.png" alt="alt text" width="600px" height="300px">
    </p>

  ### Training vs. Validation Loss Analysis
  - To further analyze model performance, we plotted the training vs. validation loss over epochs.
  #### The results showed:
  - Stable validation loss, confirming that the model generalizes well.
  - No significant gap between training and validation loss, indicating minimal overfitting.
  - Small fluctuations in validation loss, suggesting a well-regularized model with good predictive power.
  - This analysis validated that our feature selection and hyperparameter tuning process helped in achieving a robust model without excessive overfitting.

  <p float="left">
    <img src="https://github.com/ecesuolgunkesici/stock-price-prediction/blob/main/images/gru/gru_val_loss.png" alt="alt text" width="640px" height="480px">
    </p>

  ### 3.3 Prophet
  - *Results:* RMSE 6.71 (worst performance)
  - *Limitations:* Did not capture short-term price movements well and trend-following nature led to high lag in predictions.

  <p float="left">
  <img src="https://github.com/ecesuolgunkesici/stock-price-prediction/blob/main/images/prophet/prophet.png" alt="alt text" width="600px" height="300px">
  </p>

## :flashlight: 4. Model Evaluation & Insights
- GRU significantly outperformed LSTM and Prophet, achieving the lowest RMSE (2.53).
- Feature selection using Optuna helped in improving model performance.
- Overfitting was controlled by tuning dropout and sequence length.
- Validation loss was stable, confirming the model's generalization ability.
  ### 4.1 Future Improvements
  Exploring Transformer-based architectures (e.g., Temporal Fusion Transformer).
  Increasing dataset size to improve long-term generalization.
  Testing different hyperparameter search techniques like Bayesian Optimization.
    
## 📊 5. Conclusion
This project demonstrated stock price forecasting using deep learning models. The GRU model provided the best trade-off between accuracy and computational efficiency, making it the most suitable choice for stock price prediction among the tested models.

## :clap: 6. References
- yfinance Documentation: https://pypi.org/project/yfinance/
- Optuna Hyperparameter Optimization: https://optuna.org/
- TensorFlow/Keras for Deep Learning: https://www.tensorflow.org/

