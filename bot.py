import time
import threading
from flask import Flask, jsonify, render_template
from data import fetch_uniswap_data, enrich_data
from quantum import quantum_weighted_prediction
from ml_model import train_ml_models, predict_profitability
from rl_agent import TradingEnv, train_rl_agent
from trade_execution import execute_trade
from config import CHAINS
from stable_baselines3 import DQN
import logging
import plotly.graph_objs as go

logger = logging.getLogger("BotX3")
app = Flask(__name__)

@app.route('/quantum', methods=['GET'])
def get_quantum_prediction():
    chain_data = fetch_uniswap_data(CHAINS)
    prediction = quantum_weighted_prediction(chain_data)
    return jsonify(prediction)

@app.route('/ml', methods=['GET'])
def get_ml_prediction():
    df = enrich_data(CHAINS)
    xgb_model = xgb.XGBClassifier().load_model('models/xgb_model.json')
    lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
    is_profitable = predict_profitability(df, xgb_model, lstm_model)
    return jsonify({"profitable": is_profitable, "timestamp": time.ctime()})

@app.route('/rl', methods=['GET'])
def get_rl_prediction():
    env = TradingEnv(CHAINS)
    model = DQN.load("models/dqn_buy_sell", env=env)
    obs = env.reset()
    action, _ = model.predict(obs)
    return jsonify({"action": int(action), "timestamp": time.ctime()})

@app.route('/heatmap', methods=['GET'])
def get_heatmap():
    chain_data = fetch_uniswap_data(CHAINS)
    fig = go.Figure(data=go.Heatmap(
        z=[[chain_data[chain]['depth'] for chain in chain_data]],
        x=list(chain_data.keys()),
        y=['Depth'],
        colorscale='Viridis',
        hoverongaps=False))
    fig.update_layout(
        title="Liquidity Heatmap",
        paper_bgcolor="#0a0a15",
        font=dict(color="#00ffcc", family="Orbitron"),
        plot_bgcolor="#1c2526"
    )
    graph_html = fig.to_html(full_html=False)
    return render_template('heatmap.html', graph_html=graph_html)

def run_bot():
    print("Starting Bot X3...")
    df = enrich_data(CHAINS)
    xgb_model, lstm_model = train_ml_models(df)
    rl_model = train_rl_agent(CHAINS)
    env = TradingEnv(CHAINS)
    obs = env.reset()
    
    while True:
        action, _ = rl_model.predict(obs)
        if action == 1:
            buy_chain = "Polygon"
            sell_chain = "Polygon"
            if execute_trade(buy_chain, sell_chain):
                print(f"[{time.ctime()}] AI Buy: {buy_chain}")
        elif action == 2:
            buy_chain = "Polygon"
            sell_chain = "Polygon"
            if execute_trade(buy_chain, sell_chain):
                print(f"[{time.ctime()}] AI Sell: {buy_chain}")
        elif action == 3:
            buy_chain = "Polygon"
            sell_chain = "Polygon"
            if execute_trade(buy_chain, sell_chain):
                print(f"[{time.ctime()}] AI Sandwich: {buy_chain}")
        else:
            print(f"[{time.ctime()}] Holding...")
        
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
        time.sleep(5)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False))
    flask_thread.daemon = True
    flask_thread.start()
    run_bot()
