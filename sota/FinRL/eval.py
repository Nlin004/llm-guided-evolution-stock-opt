# sota/FinRL/eval.py
import warnings
import os
import sys

# Suppress all gym deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Gym.*')
warnings.filterwarnings('ignore', message='.*gymnasium.*')
warnings.filterwarnings('ignore', message='.*zipline.*')


import argparse
import importlib
import re
# import ast
import pickle
import cloudpickle
import random
import numpy as np
import pandas as pd
from pathlib import Path as p
import warnings
import torch
import multiprocessing as mp


from os.path import join as pj
from pyfolio import timeseries
# from finrl import config
from finrl import config_tickers
# from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# from finrl.meta.data_processor import DataProcessor
# from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
# from finrl.plot import backtest_stats

from preprocess import clean_data, YahooDownloaderLocal
from utils.custom_DRLAgent import CustomDRLAgent
# from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv # for parallelizing the Environments!
import model # import the custom user-defined models.py! This contains the actual architecture we will evolve.
# ---------------------------------------------------------------------------------

def create_save_dir(save_root):
    if not p(save_root).exists():
        p(save_root).mkdir(exist_ok=True, parents=True)
    n = []
    for exp_dir in p(save_root).iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            i = -1
            while exp_name[i].isdigit():
                i -= 1
            i += 1
            if i != 0:
                n.append(int(exp_name[i:]))
    if len(n) == 0:
        save_dir = pj(save_root, "exp1")
    else:
        save_dir = f"{save_root}/exp{sorted(n)[-1] + 1}"
    return save_dir


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="model", help="model file name")
    parser.add_argument("-save_dir", type=str, default="trained", help="directory to save results")
    parser.add_argument("-random_seed", type=int, default=42)
    parser.add_argument("-variant_dir", type=str, default="models", help="LLM-generated model directory")
    return parser.parse_args()



# def vectorized_prediction(model, env_vec):
#     """Prediction on vectorized environment - returns first env results."""
#     obs = env_vec.reset()
#     n_envs = env_vec.num_envs
    
#     # Track results for all environments
#     all_returns = [[] for _ in range(n_envs)]
#     all_actions = [[] for _ in range(n_envs)]
#     dones = [False] * n_envs
    
#     while not all(dones):
#         actions, _ = model.predict(obs, deterministic=True)
#         obs, rewards, new_dones, infos = env_vec.step(actions)
        
#         for i in range(n_envs):
#             if not dones[i]:  # Only append if not done yet
#                 all_returns[i].append(rewards[i])
#                 all_actions[i].append(actions[i])
        
#         dones = new_dones
    
#     # Use first environment's results (they should all be identical for trading)
#     env_idx = 0
    
#     df_daily_return = pd.DataFrame({
#         'daily_return': all_returns[env_idx]
#     })
    
#     df_actions = pd.DataFrame(all_actions[env_idx])
    
#     return df_daily_return, df_actions

# def simple_vectorized_prediction(model, env_vec):
#     """Simple prediction for single vectorized environment."""
#     obs = env_vec.reset()
#     done = False
#     returns = []
#     actions_list = []
    
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env_vec.step(action)
#         returns.append(reward[0])  # Extract from array
#         actions_list.append(action[0])
#         done = done[0]  # Extract boolean from array
    
#     df_daily_return = pd.DataFrame({'daily_return': returns})
#     df_actions = pd.DataFrame(actions_list)
    
#     return df_daily_return, df_actions

# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = p(__file__).parent
    os.chdir(script_dir)  # no resolve() needed

    args = get_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    sys.path.append(args.variant_dir)
    model_module = importlib.import_module(args.model)

    gene_id = args.model.split("model_")[1] if "model_" in args.model else "seed"

    save_dir = p(args.save_dir) / gene_id
    save_dir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*Gym.*')


    # === Load and Clean Preprocessed Data ===
    print("[eval.py] Loading training and trading data...")


    # True (DEFAULT): if you want to use a local pkl file (faster if you already have it), which should be the case if you ran 'python preprocess.py' first.
    # False: if you want to utilize the YFinance Downloader to preprocess the data instead of saving to a local pkl/csv.
    USE_LOCAL_DF = True 
    if USE_LOCAL_DF:
        data_filepath_pkl = "data/full_data_clean.pkl"
        if os.path.exists(data_filepath_pkl):
            print(f"local hard data exists!")
            # df = pd.read_csv(data_filepath)
            # df = pd.read_pickle(data_filepath_pkl)
        else: 
            print("Error: file not found. Run 'python preprocess.py' first to generate the data, or set USE_CSV to False.")
    else: # 
        print(f"Downloading and cleaning data first.")
        df_not_processed = YahooDownloaderLocal(
                    start_date = '2008-01-01',
                    end_date = '2021-09-02',
                    ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
        df = clean_data(df_not_processed)


    # FETCH THE TRAINING AND TRADING ENVIRONMENTS, defined from 'python preprocess.py'
    # with open("data/train_df.pkl", "rb") as f:
    #     train_df = pickle.load(f)
    with open("data/trade_df.pkl", "rb") as f:
        trade_df = pickle.load(f)
    with open("data/env_kwargs.pkl", "rb") as f:
        env_kwargs = pickle.load(f)
    # To parallelize the training environment (faster training):
    # Check if GPU available
    device = "cpu"
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        device = "cuda"
        torch.cuda.init()
    print(f"Using device: {device}")
    # === Define callable env factory functions ===
    def make_train_env(data_path, env_kwargs_local): # we could do the training data outside the function, but this helps for parallelization
        def _init():
            # Each process loads its own copy from disk
            # import pandas as pd
            train_df = pd.read_pickle(data_path)
            return StockPortfolioEnv(df=train_df, **env_kwargs_local)
        return _init

    # parallelize the training environment, 3 seems to work, 4 might? but sometimes cuts out. play around with it
    # mp.set_start_method("spawn", force=True)
    num_envs = min(3, os.cpu_count())  # Up to 8 envs
    env_train_vec = SubprocVecEnv(
        [make_train_env('data/train_df.pkl', env_kwargs) for _ in range(num_envs)],
        start_method='spawn'
    )
    # env_train_vec = DummyVecEnv([make_train_env()])  # single-threaded but still vector API

    # TRAINING THE MODEL OF CHOICE:
    # =========== Default DDPG =================
    # agent = DRLAgent(env=env_train)
    # DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
    # model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS)
    # trained_ddpg = agent.train_model(
    #     model=model_ddpg, tb_log_name="ddpg", total_timesteps=50000
    # )
    ### trained_ddpg.save("sota/FinRL/trained/seed/trained_ddpg.zip")

    # =========== CUSTOM DDPG =================
    agent = CustomDRLAgent(env=env_train_vec) 

    CUSTOM_DDPG_PARAMS = {
        "verbose": 0, # disable logging in eval stage
        "device": device,
        "action_noise": OrnsteinUhlenbeckActionNoise(mean=np.zeros(env_train_vec.action_space.shape[0]), sigma=0.1 * np.ones(env_train_vec.action_space.shape[0])),
        # "action_noise": NormalActionNoise(mean=np.zeros(env_train_vec.action_space.shape[0]), sigma=0.1 * np.ones(env_train_vec.action_space.shape[0])),
    }

    # Instantiate the agent, given the custom model class and parameters
    model_ddpg = agent.get_model(
        model_name="custom_ddpg",
        model_class=model.CustomDDPG,
        model_kwargs=CUSTOM_DDPG_PARAMS)

    # TRAIN MODEL
    trained_ddpg = agent.train_model(
        model=model_ddpg, tb_log_name=None, total_timesteps=50000
    )
    env_train_vec.close() # close the parallel envs after training is done



    # ================ TRADING / Test =================
    print("[eval.py] Trading Environment + Predictions on Trained DRLAgent...")

    e_trade_gym = StockPortfolioEnv(df=trade_df, **env_kwargs)
    df_daily_return, df_actions = agent.DRL_prediction(model=trained_ddpg, environment=e_trade_gym)
    # df_daily_return, df_actions = vectorized_prediction(trained_ddpg, e_trade_gym) # don't need to parallelize one testing run

    # === Backtesting ===
    print("[eval.py] Running performance backtest...")
    DRL_strat = pd.Series(df_daily_return["daily_return"].values, 
                    index=df_daily_return["date"])
    perf_stats = timeseries.perf_stats(
        returns=DRL_strat,
        factor_returns=DRL_strat,
        positions=None,
        transactions=None,
        turnover_denom="AGB"
    )
    print("\n" + "="*80)
    print("PERFORMANCE METRICS:")
    print("="*80)
    print(perf_stats)
    print("="*80 + "\n")

    # Extract key metrics for evolution scoring, choose which ones you'd like
    annual_return = perf_stats['Annual return']
    annual_volatility = perf_stats['Annual volatility']
    sharpe_ratio = perf_stats['Sharpe ratio']
    max_drawdown = perf_stats['Max drawdown']
    calmar_ratio = perf_stats['Calmar ratio']

    # You can also add other metrics:
    sortino_ratio = perf_stats.get('Sortino ratio', 0)
    cumulative_return = (1 + DRL_strat).cumprod().iloc[-1] - 1


    # === Write Results ===
    print("[eval.py] Writing results...")

    try:
        gene_id = args.model.split("model_")[1]
    except:
        gene_id = "seed"


    # Format results as comma-separated values
    # These are the metrics LLM will optimize
    # results_text = f"{annual_return:.6f},{annual_volatility:.6f},{sharpe_ratio:.6f},{max_drawdown:.6f},{calmar_ratio:.6f},{sortino_ratio:.6f},{cumulative_return:.6f}"
    results_text = f"{annual_return:.6f},{annual_volatility:.6f},{sharpe_ratio:.6f}"

    # Write to results file
    results_dir = p("results")
    results_path = results_dir / f"{gene_id}_results.txt"
    with open(results_path, 'w') as f:
        f.write(results_text)

    print(f"[eval.py] Results written to {results_path}")


    # ======================= BASELINES FOR FUN =========================
    # Add simple baselines to compare
    print("=" * 80)
    print("BASELINE COMPARISONS")
    print("=" * 80)

    # Baseline 1: Buy and hold equal weight
    equal_weight_returns = trade_df.groupby('date')['close'].mean().pct_change()
    ew_sharpe = (equal_weight_returns.mean() * 252) / (equal_weight_returns.std() * np.sqrt(252))
    ew_annual_return = (1 + equal_weight_returns).prod() ** (252 / len(equal_weight_returns)) - 1

    print(f"Equal Weight Portfolio:")
    print(f"  Annual Return: {ew_annual_return:.2%}")
    print(f"  Sharpe Ratio: {ew_sharpe:.2f}")

    # Baseline 2: Best single stock
    best_stock_returns = trade_df.groupby('date').apply(
        lambda x: x.nlargest(1, 'close')['close']
    ).pct_change()

    # Your model
    model_annual_return = perf_stats['Annual return']
    model_sharpe = perf_stats['Sharpe ratio']

    print(f"\nYour DDPG Model:")
    print(f"  Annual Return: {model_annual_return:.2%}")
    print(f"  Sharpe Ratio: {model_sharpe:.2f}")

    if model_sharpe > ew_sharpe * 2:
        print("WARNING: Model significantly outperforms baseline - check for overfitting!")

    print("=" * 80 + "\n")

    # ========================== GET WEIGHTS OF EACH STOCK OVER TIME: =========================
    # def extract_weights(drl_actions_list):
    #     model_weight_df = {'date':[], 'weights':[]}
    #     for i in range(len(drl_actions_list)):
    #         date = drl_actions_list.index[i]
    #         tic_list = list(drl_actions_list.columns)
    #         weights_list = drl_actions_list.reset_index()[list(drl_actions_list.columns)].iloc[i].values
    #         weight_dict = {'tic':[], 'weight':[]}
    #         for j in range(len(tic_list)):
    #             weight_dict['tic'] += [tic_list[j]]
    #             weight_dict['weight'] += [weights_list[j]]

    #         model_weight_df['date'] += [date]
    #         model_weight_df['weights'] += [pd.DataFrame(weight_dict)]

    #     model_weights = pd.DataFrame(model_weight_df)
    #     return model_weights
    
    # ddpg_weights = extract_weights(df_actions)
    # print("============================ TRADING WEIGHTS FROM DDPG ============================")
    # ddpg_weights.to_csv(results_dir / "ddpg_weights.csv", index=False)
    # print(ddpg_weights.head())

    



