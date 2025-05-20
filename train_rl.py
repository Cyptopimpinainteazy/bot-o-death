from rl_agent import train_rl_agent, CHAINS
import os

# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

if __name__ == "__main__":
    print("Training RL Model for Bot X3...")
    model = train_rl_agent(CHAINS)
    print("RL Model Trained and Saved to models/dqn_buy_sell.zip!")
