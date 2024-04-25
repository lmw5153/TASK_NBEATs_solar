#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

STORAGE=os.getenv('STORAGE')
DATASETS_PATH=os.path.join(STORAGE, 'datasets')
EXPERIMENTS_PATH=os.path.join(STORAGE, 'experiments')
TESTS_STORAGE_PATH=os.path.join(STORAGE, 'test')


import logging
import os
import pathlib
import sys
from urllib import request

def download(url: str, file_path: str) -> None:
    """
    Download a file to the given path.

    :param url: URL to download
    :param file_path: Where to download the content.
    """

    def progress(count, block_size, total_size):
        progress_pct = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write('\rDownloading {} to {} {:.1f}%'.format(url, file_path, progress_pct))
        sys.stdout.flush()

    if not os.path.isfile(file_path):
        opener = request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        request.install_opener(opener)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        f, _ = request.urlretrieve(url, file_path, progress)
        sys.stdout.write('\n')
        sys.stdout.flush()
        file_info = os.stat(f)
        logging.info(f'Successfully downloaded {os.path.basename(file_path)} {file_info.st_size} bytes.')
    else:
        file_info = os.stat(file_path)
        logging.info(f'File already exists: {file_path} {file_info.st_size} bytes.')


def url_file_name(url: str) -> str:
    """
    Extract file name from url.

    :param url: URL to extract file name from.
    :return: File name.
    """
    return url.split('/')[-1] if len(url) > 0 else ''


# In[2]:


"""
M4 Dataset
"""
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
#import patoolib
from tqdm import tqdm

#from common.http_utils import download, url_file_name
#from common.settings import DATASETS_PATH

FREQUENCIES = ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
URL_TEMPLATE = 'https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/{}/{}-{}.csv'

TRAINING_DATASET_URLS = [URL_TEMPLATE.format("Train", freq, "train") for freq in FREQUENCIES]
TEST_DATASET_URLS = [URL_TEMPLATE.format("Test", freq, "test") for freq in FREQUENCIES]
INFO_URL = 'https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/M4-info.csv'
NAIVE2_FORECAST_URL = 'https://github.com/M4Competition/M4-methods/raw/master/Point%20Forecasts/submission-Naive2.rar'

DATASET_PATH = os.path.join(DATASETS_PATH, 'm4')

TRAINING_DATASET_FILE_PATHS = [os.path.join(DATASET_PATH, url_file_name(url)) for url in TRAINING_DATASET_URLS]
TEST_DATASET_FILE_PATHS = [os.path.join(DATASET_PATH, url_file_name(url)) for url in TEST_DATASET_URLS]
INFO_FILE_PATH = os.path.join(DATASET_PATH, url_file_name(INFO_URL))
NAIVE2_FORECAST_FILE_PATH = os.path.join(DATASET_PATH, 'submission-Naive2.csv')


TRAINING_DATASET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'training.npz')
TEST_DATASET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'test.npz')


@dataclass()
class M4Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True) -> 'M4Dataset':
        """
        Load cached dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        m4_info = pd.read_csv(INFO_FILE_PATH)
        return M4Dataset(ids=m4_info.M4id.values,
                         groups=m4_info.SP.values,
                         frequencies=m4_info.Frequency.values,
                         horizons=m4_info.Horizon.values,
                         values=np.load(
                             TRAINING_DATASET_CACHE_FILE_PATH if training else TEST_DATASET_CACHE_FILE_PATH,
                             allow_pickle=True))

    @staticmethod
    def download() -> None:
        """
        Download M4 dataset if doesn't exist.
        """
        if os.path.isdir(DATASET_PATH):
            logging.info(f'skip: {DATASET_PATH} directory already exists.')
            return

        download(INFO_URL, INFO_FILE_PATH)
        m4_ids = pd.read_csv(INFO_FILE_PATH).M4id.values

        def build_cache(files: str, cache_path: str) -> None:
            timeseries_dict = OrderedDict(list(zip(m4_ids, [[]] * len(m4_ids))))
            logging.info(f'Caching {files}')
            for train_csv in tqdm(glob(os.path.join(DATASET_PATH, files))):
                dataset = pd.read_csv(train_csv)
                dataset.set_index(dataset.columns[0], inplace=True)
                for m4id, row in dataset.iterrows():
                    values = row.values
                    timeseries_dict[m4id] = values[~np.isnan(values)]
            np.array(list(timeseries_dict.values())).dump(cache_path)

        for url, path in zip(TRAINING_DATASET_URLS, TRAINING_DATASET_FILE_PATHS):
            download(url, path)
        build_cache('*-train.csv', TRAINING_DATASET_CACHE_FILE_PATH)

        for url, path in zip(TEST_DATASET_URLS, TEST_DATASET_FILE_PATHS):
            download(url, path)
        build_cache('*-test.csv', TEST_DATASET_CACHE_FILE_PATH)

        naive2_archive = os.path.join(DATASET_PATH, url_file_name(NAIVE2_FORECAST_URL))
        download(NAIVE2_FORECAST_URL, naive2_archive)
        patoolib.extract_archive(naive2_archive, outdir=DATASET_PATH)


@dataclass()
class M4Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }

def load_m4_info() -> pd.DataFrame:
    """
    Load M4Info file.

    :return: Pandas DataFrame of M4Info.
    """
    return pd.read_csv(INFO_FILE_PATH)


# In[3]:


def group_values(values: np.ndarray, groups: np.ndarray, group_name: str) -> np.ndarray:
    """
    Filter values array by group indices and clean it from NaNs.

    :param values: Values to filter.
    :param groups: Timeseries groups.
    :param group_name: Group name to filter by.
    :return: Filtered and cleaned timeseries.
    """
    return [v[~np.isnan(v)] for v in values[groups == group_name]]


# In[4]:


train = M4Dataset.load(training=True)
test = M4Dataset.load(training=False)

# 계절성분별로 m4데이터 리스트로 분할
group_train= [group_values(train.values, train.groups, M4Meta.seasonal_patterns[i]) for i in range(len(M4Meta.seasonal_patterns))]
group_test= [group_values(test.values, test.groups, M4Meta.seasonal_patterns[i]) for i in range(len(M4Meta.seasonal_patterns))]


# In[5]:


"""
Timeseries sampler
"""
import numpy as np

#import gin

#@gin.configurable
class TimeseriesSampler:
    def __init__(self,
                 timeseries: np.ndarray,
                 insample_size: int,
                 outsample_size: int,
                 window_sampling_limit: int,
                 batch_size: int = 1024):
        """
        Timeseries sampler.

        :param timeseries: Timeseries data to sample from. Shape: timeseries, timesteps
        :param insample_size: Insample window size. If timeseries is shorter then it will be 0-padded and masked.
        :param outsample_size: Outsample window size. If timeseries is shorter then it will be 0-padded and masked.
        :param window_sampling_limit: Size of history the sampler should use.
        :param batch_size: Number of sampled windows.
        """
        self.timeseries = [ts for ts in timeseries]
        self.window_sampling_limit = window_sampling_limit
        self.batch_size = batch_size
        self.insample_size = insample_size
        self.outsample_size = outsample_size

    def __iter__(self):
        """
        Batches of sampled windows.

        :return: Batches of:
         Insample: "batch size, insample size"
         Insample mask: "batch size, insample size"
         Outsample: "batch size, outsample size"
         Outsample mask: "batch size, outsample size"
        """
        while True:
            insample = np.zeros((self.batch_size, self.insample_size))
            insample_mask = np.zeros((self.batch_size, self.insample_size))
            outsample = np.zeros((self.batch_size, self.outsample_size))
            outsample_mask = np.zeros((self.batch_size, self.outsample_size))
            sampled_ts_indices = np.random.randint(len(self.timeseries), size=self.batch_size)
            for i, sampled_index in enumerate(sampled_ts_indices):
                sampled_timeseries = self.timeseries[sampled_index]
                cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                              high=len(sampled_timeseries),
                                              size=1)[0]

                insample_window = sampled_timeseries[max(0, cut_point - self.insample_size):cut_point]
                insample[i, -len(insample_window):] = insample_window
                insample_mask[i, -len(insample_window):] = 1.0
                outsample_window = sampled_timeseries[
                                   cut_point:min(len(sampled_timeseries), cut_point + self.outsample_size)]
                outsample[i, :len(outsample_window)] = outsample_window
                outsample_mask[i, :len(outsample_window)] = 1.0
            yield insample, insample_mask, outsample, outsample_mask

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.insample_size))
        insample_mask = np.zeros((len(self.timeseries), self.insample_size))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.insample_size:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


# In[6]:


# 100000 data 통합 
dct = {'group'+str(i): TimeseriesSampler(group_train[i],12,6,1) for i in range(len(group_train))}
dct_name = [k for k,v in dct.items()]
inputsize = np.array(M4Meta.horizons)*2

dct2 = {'group'+str(i): TimeseriesSampler(group_test[i],12,6,1) for i in range(len(group_test))}
dct_name2 = [k for k,v in dct2.items()]


# In[7]:


dataframes = []
# input
for group  in range(len(group_train)):
    lst = pd.DataFrame([np.array([dct[j].timeseries[i][-inputsize[group]:] for i in range(len(dct[j].timeseries))] )for j in dct_name ][ group ])
    dataframes.append(lst)
#total_X = pd.concat(dataframes )


# output
dataframes2= []
for group  in range(len(group_test)):
    lst2 = pd.DataFrame([np.array([dct2[j].timeseries[i][:] for i in range(len(dct2[j].timeseries))] )for j in dct_name2 ][ group ])
    dataframes2.append(lst2)


# In[8]:


datashape = [dataframes[i].shape for i in range(len(dataframes))]

base_domain= [pd.concat([dataframes[i],dataframes2[i]],axis=1) for i in range(len(dataframes))]


# In[9]:


def base_data():
    return base_domain


# In[10]:


for df in base_domain:
    df.columns = range(df.shape[1])

# 새로운 컬럼 이름으로 데이터프레임 합치기
result = pd.concat(base_domain, axis=0)


# In[14]:


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
minmax2 = MinMaxScaler()


# In[15]:


X = result.reset_index(drop=True)
X= minmax.fit_transform(X)


# In[16]:


inputsize = np.array(M4Meta.horizons)*2

zt1= X[:23000,:inputsize[0]]
zt2=X[23000:23000+24000,:inputsize[1]]
zt3=X[23000+24000:23000+24000+48000,:inputsize[2]]
zt4=X[95000:95000+359,:inputsize[3]]
zt5=X[95000+359:95000+359+4227,:inputsize[4]]
zt6=X[95000+359+4227:95000+359+4227+414,:inputsize[5]]

zt11= X[:23000,inputsize[0]:(inputsize[0]+M4Meta.horizons[0])]
zt22=X[23000:23000+24000,inputsize[1]:(inputsize[1]+M4Meta.horizons[1])]
zt33=X[23000+24000:23000+24000+48000,inputsize[2]:(inputsize[2]+M4Meta.horizons[2])]
zt44=X[95000:95000+359,inputsize[3]:(inputsize[3]+M4Meta.horizons[3])]
zt55=X[95000+359:95000+359+4227,inputsize[4]:(inputsize[4]+M4Meta.horizons[4])]
zt66=X[95000+359+4227:95000+359+4227+414,inputsize[5]:]


# In[17]:


zt_in= [zt1,zt2,zt3,zt4,zt5,zt6] # 민맥스 전체 데이터 적용
zt_out= [zt11,zt22,zt33,zt44,zt55,zt66]

def train_data():
    return zt_in

def output_data():
    return zt_out