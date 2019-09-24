from utils import FsStorage
import pandas as pd


TRAIN_TEST_SPLIT = 0.1
VALID_TEST_SPLIT = 0.2
pd.options.mode.chained_assignment = None 

    
def split_train_test(df, test_ratio, v=1):
    # Just splits according to given order
    cut_index = (int(1-test_ratio*df.shape[0]))
    df_train, df_test = df.iloc[:cut_index], df.iloc[cut_index:]
    return df_train, df_test


def split_by_time(df, split_ratio=.2):
    df = df.dropna(subset=['first_seen'])
    df['timestamp'] = pd.to_datetime(df['first_seen'])
    split_date = df['timestamp'].quantile(1 - split_ratio).date()
    df['timestamp'] = df['timestamp'].apply(lambda x: x.date())
    df_train = df[df['timestamp'] <= split_date]
    df_test = df[df['timestamp'] > split_date]
    return df_train, df_test


def get_datasets(files_df):
        df = pd.read_csv(files_df)
        df_ben = df[df['verdict'] == 0].sample(frac=1, random_state=42)
        df_mal = df[df['verdict'] == 1]
        df_ben_train, df_ben_test = split_train_test(df_ben, TRAIN_TEST_SPLIT)
        df_ben_test, df_ben_valid = split_train_test(df_ben_test, VALID_TEST_SPLIT)
        df_mal_train, df_mal_test = split_by_time(df_mal, TRAIN_TEST_SPLIT)
        df_mal_valid, df_mal_test = split_by_time(df_mal_test, 1-VALID_TEST_SPLIT)  # Valid before Test
        df_train = pd.concat((df_mal_train, df_ben_train), sort=False).sort_values('hash')
        df_test = pd.concat((df_mal_test, df_ben_test), sort=False).sort_values('hash')
        df_valid = pd.concat((df_mal_valid, df_ben_valid), sort=False).sort_values('hash')
        return df_train, df_valid, df_test
        
