#!/usr/bin/env python3
"""
Train DQN on the hierarchical housing environment.
"""
import sys
sys.path.insert(0, '.')

from housing_env import HousingEnv
from stable_baselines3 import DQN
import numpy as np
from gymnasium import spaces
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

def main():
    print("=" * 60)
    print("Training DQN on Hierarchical Housing Environment")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating hierarchical environment...")
    env = HousingEnv(
        hierarchical=True,
        max_active_reports=300,
        num_inspectors=100,
        years=0.1,  # 36 days
    )
    
    # Wrap to flatten observations
    env = FlattenObservation(env)
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    
    # Create DQN model
    print("\n2. Creating DQN model...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        tau=1.0,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
    )
    
    # Train
    print("\n3. Training for 50,000 timesteps...")
    model.learn(total_timesteps=50000, log_interval=10)
    
    # Evaluate
    print("\n4. Evaluating on 5 episodes...")
    total_reward = 0
    violations_fixed = 0
    
    for episode in range(5):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        
        violations_fixed = max(violations_fixed, env.unwrapped.episode_violations_fixed)
        total_reward += ep_reward
        print(f"   Episode {episode+1}: reward={ep_reward:.1f}, violations_fixed={env.unwrapped.episode_violations_fixed}")
    
    print(f"\n   Average reward: {total_reward/5:.1f}")
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()
