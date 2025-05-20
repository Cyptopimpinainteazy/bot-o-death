import xgboost as xgb
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import tensorflow as tf

def train_ml_models(df):
    features = ['price', 'depth', 'volume', 'volatility', 'spread', 'hour']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_model.save_model('models/xgb_model.json')
    
    # LSTM
    time_steps = 5
    X_lstm, y_lstm = [], []
    for chain in df['chain'].unique():
        chain_df = df[df['chain'] == chain].sort_values('timestamp')
        for i in range(len(chain_df) - time_steps):
            X_lstm.append(chain_df[features].iloc[i:i+time_steps].values)
            y_lstm.append(chain_df['target'].iloc[i+time_steps])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2)
    
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, len(features))),
        Dropout(0.2),
        LSTM(50),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=0)
    lstm_model.save('models/lstm_model.h5')
    
    return xgb_model, lstm_model

def predict_profitability(df, xgb_model, lstm_model):
    features = ['price', 'depth', 'volume', 'volatility', 'spread', 'hour']
    X_latest = df[features].iloc[-1:]
    X_lstm_latest = np.array([df[features].iloc[-5:].values])
    
    xgb_prob = xgb_model.predict_proba(X_latest)[0][1]
    lstm_prob = lstm_model.predict(X_lstm_latest, verbose=0)[0][0]
    return (xgb_prob + lstm_prob) / 2 > 0.5
