import pandas as pd
from environment import HelpdeskEnvironment
from sentence_transformers import SentenceTransformer
from stable_baselines3 import PPO

print("Loading training data...")
df = pd.read_csv("dataset/training_data.csv")
department_map = {"HR": 0, "IT": 1, "Payroll": 2}
df['correct_action_idx'] = df['correct_department'].map(department_map)
training_data = list(zip(df['query'], df['correct_action_idx']))

print("Setting up training environment...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
env = HelpdeskEnvironment(training_data, embedding_model)

print("🚀 Starting training for the Communication Agent...")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_helpdesk_tensorboard/", device="cpu")
model.learn(total_timesteps=len(training_data) * 100)
model.save("communication_agent_model")

print("\n✅ Training complete. Model saved as 'communication_agent_model.zip'")