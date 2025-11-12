import os
import pickle
from pathlib import Path as p
import pandas as pd
import numpy as np
import yfinance as yf
from finrl import config
from finrl import config_tickers

# from finrl.meta.preprocessor.yahoodownloader import YahooDownloader --- IGNORE ---
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from finrl.meta.data_processor import DataProcessor
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor


class YahooDownloaderLocal:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API, based on the FinRL adaptation of YahooDownloader.

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None, auto_adjust=False) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic,
                start=self.start_date,
                end=self.end_date,
                proxy=proxy,
                auto_adjust=auto_adjust,
            )
            if temp_df.columns.nlevels != 1:
                temp_df.columns = temp_df.columns.droplevel(1)
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                # data_df = data_df.append(temp_df)
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.rename(
                columns={
                    "Date": "date",
                    "Adj Close": "adjcp",
                    "Close": "close",
                    "High": "high",
                    "Low": "low",
                    "Volume": "volume",
                    "Open": "open",
                    "tic": "tic",
                },
                inplace=True,
            )

            if not auto_adjust:
                data_df = self._adjust_prices(data_df)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def _adjust_prices(self, data_df: pd.DataFrame) -> pd.DataFrame:
        # use adjusted close price instead of close price
        data_df["adj"] = data_df["adjcp"] / data_df["close"]
        for col in ["open", "high", "low", "close"]:
            data_df[col] *= data_df["adj"]

        # drop the adjusted close price column
        return data_df.drop(["adjcp", "adj"], axis=1)

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df

def clean_data(df_in) -> pd.DataFrame:
    """
    Preprocess data from Yahoo Finance to be compatible with FinRL.
    This includes renaming columns, adding technical indicators, and computing covariance matrices.
    """
    dp = df_in
    fe = FeatureEngineer(
        use_technical_indicator=True,
        use_turbulence=False,
        user_defined_feature=False
    )

    df = fe.preprocess_data(df_in) # adds technical indicators
    # df = compute_technical_indicators(df) # OR you can do this one, for feature engineering manually.

    # add covariance matrix as states
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    return_list = []

    # look back is one year
    lookback=252
    for i in range(lookback,len(df.index.unique())):
        data_lookback = df.loc[i-lookback:i,:]
        price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        covs = return_lookback.cov().values 
        cov_list.append(covs)

    
    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)
    return df


# def build_envs(train_df, trade_df, env_kwargs):
#     e_train = StockPortfolioEnv(df=train_df, **env_kwargs)
#     env_train, _ = e_train.get_sb_env()

#     e_trade = StockPortfolioEnv(df=trade_df, **env_kwargs)

#     with open("data/env_train.pkl", "wb") as f:
#         pickle.dump(env_train, f)
#     with open("data/env_trade.pkl", "wb") as f:
#         pickle.dump(e_trade, f)
#     print("Stock Environments saved to disk.")


def create_env_kwargs_pkl(env_kwargs, train_df):
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    tech_indicator_list = ['macd', 'rsi_30', 'cci_30', 'dx_30']
    feature_dimension = len(tech_indicator_list)
    print(f"Feature Dimension: {feature_dimension}")

    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": tech_indicator_list, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-1
    }

    with open("data/env_kwargs.pkl", "wb") as f:
        pickle.dump(env_kwargs, f)



if __name__ == "__main__":
    # Create all relevant directories in sota/FinRL/
    # PATHS FOR FINRL'S ARTIFACTS (where they save trained models, tensorboard logs, results, data saved from preprocessors, etc.)
    script_dir = p(__file__).parent
    finrl_dir = script_dir
    dirs = {
        "data": finrl_dir / "data",
        "trained": finrl_dir / "trained",
        # "tensorboard": finrl_dir / "tensorboard_log",
        "results": finrl_dir / "results"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Check if you've already downloaded the data
    file_path = "data/full_data_clean.pkl"
    # file_path = "data/train_df.pkl"

    if os.path.exists(file_path):
        print(f"local pkl df exists, reading from there.")
        df = pd.read_pickle(file_path)
    else:
        print(f"\n\n ====== Local pkl df does not exist, downloading now. ====== \n\n")
        myDOW_30_TICKER = [
            "AAPL", "AMGN", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS",
            "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO",
            "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV",
            "UNH", "V", "VZ", "WBA", "WMT"
        ]
        df_not_processed = YahooDownloaderLocal(
            start_date = '2001-01-01',
            end_date = '2018-10-02',
            ticker_list = myDOW_30_TICKER).fetch_data()

        df = clean_data(df_not_processed)
        df.to_pickle("data/full_data_clean.pkl")
        print("=== Saved full_data_clean.pkl ===")

    train = data_split(df, '2001-01-01','2013-12-30')
    trade = data_split(df, '2014-01-02','2018-10-02')
    train.to_pickle("data/train_df.pkl")
    trade.to_pickle("data/trade_df.pkl")
    print("=== Created train_df.pkl and trade_df.pkl data splits. ===")

    # check if envKwargs has a pkl. If so, read from there.
    if not os.path.exists("data/env_kwargs.pkl"):
        create_env_kwargs_pkl(env_kwargs=None, train_df=train)
        print("=== Created and saved env_kwargs.pkl ===")




