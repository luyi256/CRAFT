r"""
"""
import ast
from datetime import datetime
import gzip
import requests
import os
import pandas as pd
import numpy as np
import torch
import zipfile
from tgb_seq.LinkPred.negsampler import NegativeSampler
import argparse
import tarfile
src_url = {"ML-20M": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
           "Yelp": "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Yelp/yelp-full.zip",
           "GoogleLocal": "http://jmcauley.ucsd.edu/data/googlelocal/googlelocal.tar.gz",
           "Flickr": "https://nrvis.com/download/data/dynamic/soc-flickr-growth.zip",
           "YouTube": "https://nrvis.com/download/data/dynamic/soc-youtube-growth.zip",
           "Patent": "https://www.dropbox.com/scl/fo/5w0a14icfv4o7t0azrqda/ALzmXYPzoVbuvYEFPNWAzyM/data/patent/patent_edges.json?rlkey=qhx7csgahlcbuppjx4ewa3l0o&e=1&dl=1",
           "WikiLink": "https://nrvis.com/download/data/web/web-wikipedia-growth.zip"
           }
bipartite_dict = {"ML-20M": True, "Taobao": True, "Yelp": True, "GoogleLocal": True,
                  "Flickr": False, "YouTube": False, "Patent": False, "WikiLink": False}


class TGBSeqPreprocessor(object):
    r"""
    TGBSeqPreprocessor class provides the complete pipeline to preprocess the raw data and generate the TGB-Seq processed dataset.
    """

    def __init__(self, dataset_name, root, split_ratio=[0.15, 0.15], threshold=3, neg_num=100):
        r"""
        Initialize the TGBSeqPreprocessor object.
        :param dataset_name: str, dataset name
        :param root: str, root directory to save the processed dataset
        :param split_ratio: list, split ratio for train, validation, and test set
        :param threshold: int, threshold of the degree of nodes for filtering
        """
        self.dataset_name = dataset_name
        self.root = root
        self.file_dir = f"{root}/{dataset_name}/"
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)
        self.split_ratio = split_ratio
        self.threshold = threshold
        self.neg_num = neg_num
        print(f"Start processing dataset {dataset_name}...")
        self._download()
        self._preprocess()
        # self._generate_negatives()

    def _download(self):
        r"""
        Download the raw dataset from the source URL.
        Note that you have to download Taobao dataset from https://tianchi.aliyun.com/dataset/649 and put it into the root directory manually.
        This is because the dataset requires a login to download. To respect the requirement, we do not provide the download script.
        If you meet any problems, we provide a collection of the source datasets in .
        """
        if self.dataset_name in src_url:
            url = src_url[self.dataset_name]
            # if downloaded in self.root/self.dataset_name, skip downloading
            if self.dataset_name == 'Patent' and os.path.exists(f"{self.file_dir}/patent_edges.json"):
                return
            elif self.dataset_name == 'GoogleLocal' and os.path.exists(f"{self.file_dir}/googlelocal.tar.gz"):
                return
            elif os.path.exists(f"{self.file_dir}/{self.dataset_name}.zip"):
                return
            try:
                response = requests.get(url)
                if self.dataset_name == 'Patent':
                    with open(f"{self.file_dir}/patent_edges.json", "wb") as f:
                        f.write(response.content)
                elif self.dataset_name == 'GoogleLocal':
                    with open(f"{self.file_dir}/googlelocal.tar.gz", "wb") as f:
                        f.write(response.content)
                else:
                    with open(f"{self.file_dir}/{self.dataset_name}.zip", "wb") as f:
                        f.write(response.content)
            except Exception as e:
                print(f"Error: {e}")
        else:
            if self.dataset_name == "Taobao":
                # We will read the Taobao dataset from the root directory
                pass
            else:
                raise ValueError(
                    f"Dataset {self.dataset_name} not supported by TGB-Seq.")

    def _preprocess(self):
        r"""
        Preprocess the raw dataset and save the processed dataset to the root directory.
        """
        # load datasets
        if os.path.exists(f"{self.file_dir}/{self.dataset_name}.zip"):
            with zipfile.ZipFile(f"{self.file_dir}/{self.dataset_name}.zip", 'r') as zip_ref:
                zip_ref.extractall(f"{self.file_dir}/")
        if self.dataset_name == "Taobao":
            self.Taobao_load()
        elif self.dataset_name == "YouTube":
            self.YouTube_load()
        elif self.dataset_name == "Flickr":
            self.Flickr_load()
        elif self.dataset_name == "Patent":
            self.Patent_load()
        elif self.dataset_name == "GoogleLocal":
            self.GoogleLocal_load()
        elif self.dataset_name == "ML-20M":
            self.ML_20M_load()
        elif self.dataset_name == "Yelp":
            self.Yelp_load()
        elif self.dataset_name == "WikiLink":
            self.WikiLink_load()
        else:
            raise ValueError(
                f"Dataset {self.dataset_name} not supported by TGB-Seq.")
        if bipartite_dict[self.dataset_name]:
            self.bipartite_reindex()
        else:
            self.unipartite_reindex()
        # preprocess
        if self.dataset_name != "Patent":
            self.filter()  # Patent has been filetered in Patent_load()
        if bipartite_dict[self.dataset_name]:
            self.bipartite_reindex()
        else:
            self.unipartite_reindex()
        # save the processed dataset
        self.df.to_csv(f"{self.file_dir}/{self.dataset_name}_named.csv", index=False)
        print(f"Dataset {self.dataset_name} is processed and saved to {
              self.file_dir}/{self.dataset_name}.csv.")

    def _generate_negatives(self):
        neg_sampler = NegativeSampler(self.dataset_name, self.df.dst.min(
        ), self.df.dst.max(), num_neg_e=100, strategy="rnd")
        val_store_file = f"{self.file_dir}/val_ns.npy"
        test_store_file = f"{self.file_dir}/test_ns.npy"
        val_negs = neg_sampler.sample_neg(self.df[self.df['split'] == 1]['src'].values, self.df[self.df['split'] == 1]['dst'].values, file_name=val_store_file, store=True, collision_check=True)
        test_negs = neg_sampler.sample_neg(self.df[self.df['split'] == 2]['src'].values, self.df[self.df['split'] == 2]['dst'].values, file_name=test_store_file, store=True, collision_check=True)
        print(f"Negative samples for validation set are saved to {
              val_store_file}.")
        print(f"Negative samples for test set are saved to {test_store_file}.")

    def Taobao_load(self):
        zip_file = self.file_dir+"/UserBehavior.csv.zip"
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.file_dir)
        df = pd.read_csv(self.file_dir+"/UserBehavior.csv", header=None)
        df.columns = ["src", "dst", "dst_cat", "behavior", "time"]
        df = df[df["behavior"] == "pv"]
        df.drop(columns=["behavior", "dst_cat"], inplace=True)
        self.df = df

    def GoogleLocal_load(self):
        with tarfile.open(f"{self.file_dir}/googlelocal.tar.gz", 'r:gz') as tar:
            tar.extractall(f"{self.file_dir}/")
        # Step 1: Initialize lists to store the data
        data = []
        empty_line = []
        # Step 2: Read and process each line of the input file
        with gzip.open(self.file_dir+'googlelocal/reviews.clean.json.gz', 'rt', encoding='utf-8') as file:
            for i, line in enumerate(file):
                try:
                    review = ast.literal_eval(line.strip())
                    user_id = review['gPlusUserId']
                    item_id = review['gPlusPlaceId']
                    timestamp = float(review['unixReviewTime'])
                    data.append([i,user_id, item_id, timestamp])
                except:
                    empty_line.append(i)
                    continue
        df = pd.DataFrame(data, columns=['ori_ori_idx','src', 'dst', 'time'])
        df = df.sort_values(by='time').reset_index(drop=True)
        df['ori_dst']=df['dst']
        df['ori_src']=df['src']
        # user_mapping = {old_id: new_id for new_id,
        #                 old_id in enumerate(df['src'].unique())}
        # item_mapping = {old_id: new_id for new_id,
        #                 old_id in enumerate(df['dst'].unique())}
        # df['src'] = df['src'].map(user_mapping)
        # df['dst'] = df['dst'].map(item_mapping)
        self.df = df

    def ML_20M_load(self):
        df = pd.read_csv(self.file_dir+"ml-20m/ratings.csv", skiprows=1, header=None, names=[
                         'src', 'dst', 'rating', 'time'], dtype={"src": "int64", "dst": "int64", "rating": "float", "time": "int64"})
        df.drop(columns=["rating"], inplace=True)
        self.df = df

    def Yelp_load(self):
        df = pd.read_csv(self.file_dir+"/yelp-full/yelp-full.inter", skiprows=1, header=None, sep='\t', names=['src', 'dst', 'rating', 'time', 'useful', 'funny', 'cool', 'review'], dtype={
                         'src': 'str', 'dst': 'str', 'rating': 'float', 'time': 'int64', 'useful': 'int', 'funny': 'int', 'cool': 'int', 'review': 'str'})
        df.drop(columns=['rating', 'useful', 'funny',
                'cool', 'review'], inplace=True)
        df = df.sort_values(by='time').reset_index(drop=True)
        # user_mapping = {old_id: new_id for new_id,
        #                 old_id in enumerate(df['src'].unique())}
        # item_mapping = {old_id: new_id for new_id,
        #                 old_id in enumerate(df['dst'].unique())}
        # df['src'] = df['src'].map(user_mapping)
        # df['dst'] = df['dst'].map(item_mapping)
        self.df = df

    def YouTube_load(self):
        df = pd.read_csv(self.file_dir+'soc-youtube-growth.edges', skiprows=2, sep=' ', header=None, names=[
                         'src', 'dst', 'w', 'time'], dtype={'src': 'int64', 'dst': 'int64', 'w': 'int', 'time': 'int64'})
        df['idx'] = df.index
        df.drop(columns=['w'], inplace=True)
        self.df = df
        self.filter_Who_To_Follow()

    def Flickr_load(self):
        df = pd.read_csv(self.file_dir+"/soc-flickr-growth.edges", header=None, delim_whitespace=True, skiprows=1,
                         names=['src', 'dst', 'w', 'time'], dtype={'src': 'int64', 'dst': 'int64', 'w': 'int', 'time': 'int64'})
        df['idx'] = df.index
        self.df = df
        self.filter_Who_To_Follow()

    def filter_Who_To_Follow(self):
        df = self.df
        # 1. There are some duplicate edges in the dataset, where one of the edges has a timestamp of t0. Note that the num of duplication is always two. We remove the edges with the latest timestamps and only keep the edge with timestamp t0.
        # There is no such edges in Flickr.
        grouped_counts = df.groupby(
            ['src', 'dst']).size().reset_index(name='count')
        duplicate_combinations = grouped_counts[grouped_counts['count'] > 1]
        duplicate_rows = df.merge(
            duplicate_combinations[['src', 'dst']], on=['src', 'dst'])
        new_rows = duplicate_rows.sort_values(
            'time').groupby(['src', 'dst']).last()
        df_cleaned = df[~df.index.isin(new_rows.idx)]
        df_cleaned.reset_index(drop=True, inplace=True)
        df_cleaned = df_cleaned.loc[df_cleaned.index]
        df_cleaned['idx'] = df_cleaned.index
        df_cleaned.sort_values(['time', 'idx'], inplace=True)
        df = df_cleaned
        # 2. Some nodes have more than a half of their edges at time t0 (the first timestamp in the dataset). We remove these nodes and all their edges.
        t0 = df['time'].min()
        u_t0_counts = df[df['time'] == t0]['src'].value_counts()
        u_other_counts = df[df['time'] != t0]['src'].value_counts()
        nodes_to_remove = u_t0_counts[u_t0_counts > u_other_counts.reindex(
            u_t0_counts.index, fill_value=0)].index
        df = df[~((df['src'].isin(nodes_to_remove)) |
                  (df['dst'].isin(nodes_to_remove)))]
        self.df = df

    def Patent_load(self):
        r"""
        Load the patent dataset. 
        For the patent dataset, all the source nodes (patent) make their citations at the same time. So, we choose all the source nodes in the validation and test set and split the citations of them into three parts: serving as historical neighbors (not in training set), validation set, and test set.
        """
        file = self.file_dir+"/patent_edges.json"
        data = []
        with open(file) as f:
            for l in f:
                u, i, t, _, _ = eval(l)
                dt_obj = datetime.strptime(str(t), '%Y%m%d')
                unix_timestamp = int(dt_obj.timestamp())
                data.append([int(u), int(i), int(unix_timestamp)])
        df = pd.DataFrame(data, columns=['src', 'dst', 'time'])
        print("original shape:", df.shape)
        self.df = df
        self.filter()
        print("filtered shape:", self.df.shape)
        self.df['idx'] = self.df.index
        df = self.df
        src = torch.tensor(df['src'].values, device='cuda')
        dst = torch.tensor(df['dst'].values, device='cuda')
        time = torch.tensor(df['time'].values, device='cuda')
        idx = torch.tensor(df['idx'].values, device='cuda')
        split = torch.tensor(df['split'].values, device='cuda')

        # choose the validation and test set
        mask = (split == 1) | (split == 2)
        src_vt = src[mask]
        dst_vt = dst[mask]
        time_vt = time[mask]
        idx_vt = idx[mask]
        split_vt = split[mask]

        # sort, group by source nodes
        sorted_indices = torch.argsort(src_vt * 1e17 + time_vt * 1e8 + idx_vt)
        src_vt_sorted = src_vt[sorted_indices]
        dst_vt_sorted = dst_vt[sorted_indices]
        time_vt_sorted = time_vt[sorted_indices]
        idx_vt_sorted = idx_vt[sorted_indices]
        split_vt_sorted = split_vt[sorted_indices]
        group_changes = torch.cat(
            (torch.tensor([True], device='cuda'), src_vt_sorted[1:] != src_vt_sorted[:-1]))
        # group_starts: the start index of each group; group_ends: the end index of each group
        group_starts = torch.nonzero(group_changes).flatten()
        group_ends = torch.cat(
            (group_starts[1:], torch.tensor([len(src_vt_sorted)], device='cuda')))

        results_src = []
        results_time = []
        results_idx = []
        results_split = []
        results_dst = []
        for start, end in zip(group_starts.tolist(), group_ends.tolist()):
            group_src = src_vt_sorted[start:end]
            group_time = time_vt_sorted[start:end]
            group_idx = idx_vt_sorted[start:end]
            group_split = split_vt_sorted[start:end]
            group_dst = dst_vt_sorted[start:end]

            n = end - start
            mid_idx = int(np.ceil(n/2))
            quarter_idx = mid_idx + int(np.ceil(n/4))

            # half of the citations in a group (i.e., for a patent) are used as historical neighbors. The `split` is set to -1, indicating that they are not in the training/validation/test set. For simplicity, we set the `time` to be 2 time steps before the first citation in the validation set, so that models can use these historical neighbors to predict when validating.
            group_split[:mid_idx] = -1
            group_time[:mid_idx] -= 2

            # the next quarter of the citations are used as the validation set. The `split` is set to 1. For simplicity, we set the `time` to be 1 time step before the first citation in the test set.
            group_split[mid_idx:quarter_idx] = 1
            group_time[mid_idx:quarter_idx] -= 1

            # the rest of the citations are used as the test set. The `split` is set to 2.
            group_split[quarter_idx:] = 2

            results_src.append(group_src)
            results_dst.append(group_dst)
            results_time.append(group_time)
            results_idx.append(group_idx)
            results_split.append(group_split)

        results_src = torch.cat(results_src)
        results_dst = torch.cat(results_dst)
        results_time = torch.cat(results_time)
        results_idx = torch.cat(results_idx)
        results_split = torch.cat(results_split)

        # 转换为 pandas DataFrame
        results_df = pd.DataFrame({
            'src': results_src.cpu().numpy(),
            'dst': results_dst.cpu().numpy(),
            'time': results_time.cpu().numpy(),
            'idx': results_idx.cpu().numpy(),
            'split': results_split.cpu().numpy()
        })
        results_df.drop(columns=['idx'], inplace=True)
        self.df = pd.concat(
            [df[df['split'] == 0], results_df], ignore_index=True)
        self.df.reset_index(drop=True, inplace=True)

    def WikiLink_load(self):
        self.df = pd.read_csv(self.file_dir+'/web-wikipedia-growth.edges', skiprows=1, delim_whitespace=True, names=['src', 'dst', 'w', 'time'],
                              dtype={'src': 'int64', 'dst': 'int64', 'w': 'int', 'time': 'int64'})
        self.df.drop(columns=['w'], inplace=True)

    def filter(self):
        r"""
        Filter the dataset based on the threshold of the degree of nodes and split the dataset into training (split=0), validation (split=1), and test set (split=2).
        Only the nodes with the degree greater than the threshold will be kept in the training set.
        Only the nodes in the training set will be kept in the validation and test set.
        For Taobao, only the last 1/4 of the time will be kept in the validation and test set due to the large size of the original dataset.
        Before filtering, the dataset should contain columns ['src', 'dst', 'time', 'idx'].
        """
        if 'idx' not in self.df.columns:
            self.df['idx'] = self.df.index
        self.df.sort_values(by=['time', 'idx'], inplace=True)
        if self.dataset_name == "Taobao":
            time_threshold = self.df['time'].quantile(0.75)
            self.df = self.df[self.df['time'] >= time_threshold]
        val_time, test_time = list(np.quantile(self.df.time, [(
            1 - self.split_ratio[0] - self.split_ratio[1]), (1 - self.split_ratio[1])]))
        train_df = self.df[self.df.time <= val_time]
        print(f"Original train: {train_df.shape}")
        while True:
            u_total_counts = train_df['src'].value_counts()
            i_total_counts = train_df['dst'].value_counts()
            # if not bipartite_dict[self.dataset_name]:
            total_counts = u_total_counts.add(
                i_total_counts, fill_value=0).astype(int)
            nodes_to_remove_count = total_counts[total_counts <
                                                 self.threshold].index
            if nodes_to_remove_count.empty:
                break
            train_df = train_df[~((train_df['src'].isin(nodes_to_remove_count)) | (
                train_df['dst'].isin(nodes_to_remove_count)))]
            # else:
            #     u_nodes_to_remove = u_total_counts[u_total_counts <
            #                                   self.threshold].index
            #     i_nodes_to_remove = i_total_counts[i_total_counts <
            #                                   self.threshold].index
            #     if u_nodes_to_remove.empty and i_nodes_to_remove.empty:
            #         break
            #     train_df = train_df[~((train_df['src'].isin(u_nodes_to_remove)) | (
            #         train_df['dst'].isin(i_nodes_to_remove)))]
        print(f"New train: {train_df.shape}")
        not_train_df = self.df[self.df.time > val_time].copy()
        # if not bipartite_dict[self.dataset_name]:
        u_counts = train_df['src'].value_counts()
        i_counts = train_df['dst'].value_counts()
        tot_counts = u_counts.add(i_counts, fill_value=0).astype(int)
        not_train_df['cnt_i'] = not_train_df['dst'].map(tot_counts)
        not_train_df['cnt_u'] = not_train_df['src'].map(tot_counts)
        not_train_df.fillna(0, inplace=True)
        not_train_df = not_train_df[~((not_train_df['cnt_i'] == 1.0) | (not_train_df['cnt_i'] == 2.0) | (
            not_train_df['cnt_i'] == 0.0))]
        # only the destination nodes (i.e. the cited patents) are filtered since many new patents will appear after training time
        if self.dataset_name != 'Patent':
            not_train_df = not_train_df[~((not_train_df['cnt_u'] == 1.0) | (not_train_df['cnt_u'] == 2.0) | (
                not_train_df['cnt_u'] == 0.0))]  # actually, only 0.0 is enough
        # else:
        #     not_train_df = not_train_df[((not_train_df['src'].isin(set(train_df['src']))) & (
        #         not_train_df['dst'].isin(set(train_df['dst']))))]
        new_df = pd.concat([train_df, not_train_df], ignore_index=True)
        train_mask = (new_df.time <= val_time)
        val_mask = (new_df.time > val_time) & (new_df.time <= test_time)
        test_mask = (new_df.time > test_time)
        new_df.loc[train_mask, 'split'] = 0
        new_df.loc[val_mask, 'split'] = 1
        new_df.loc[test_mask, 'split'] = 2
        self.check_filter(new_df)
        if self.dataset_name in ['GoogleLocal']:
            self.df = new_df[['src', 'dst', 'time', 'split', 'ori_ori_idx','ori_src','ori_dst']]
            self.df=self.df.astype({'src': 'int64', 'dst': 'int64',
                       'time': 'int64', 'split': 'int64', 'ori_ori_idx': 'str', 'ori_src': 'str', 'ori_dst': 'str'})
        else:
            self.df = new_df[['src', 'dst', 'time', 'split']]
            self.df=self.df.astype({'src': 'int64', 'dst': 'int64',
                       'time': 'int64', 'split': 'int64'})

    def check_filter(self, new_df):
        r"""
        Check if the filtering process is successful.
        1) The training set should contain nodes with the degree greater than the threshold.
        2) The validation and test set should contain nodes that are only in the training set.
        """
        train_df = new_df[new_df['split'] == 0]
        u_counts = train_df['src'].value_counts()
        i_counts = train_df['dst'].value_counts()
        # if bipartite_dict[self.dataset_name]:
        #     assert u_counts.min() >= self.threshold and i_counts.min(
        #     ) >= self.threshold, "training set filtering failed!"
        # else:
        tot_counts = u_counts.add(i_counts, fill_value=0).astype(int)
        assert tot_counts.min() >= self.threshold, "training set filtering failed!"
        not_train_df = new_df[new_df['split'] != 0]
        # if bipartite_dict[self.dataset_name]:
        #     assert set(not_train_df['src']).issubset(
        #         set(train_df['src'].unique())), "Val/test set filtering failed!"
        #     assert set(not_train_df['dst']).issubset(
        #         set(train_df['dst'].unique())), "Val/test set filtering failed!"
        # else:
        seen_nodes = set(train_df['src'].unique()).union(
            set(train_df['dst'].unique()))
        if self.dataset_name == 'Patent':  # only test the destination nodes for Patent
            valtest_nodes = set(not_train_df['dst'].unique())
        else:
            valtest_nodes = set(not_train_df['src'].unique()).union(
                set(not_train_df['dst'].unique()))
        assert valtest_nodes.issubset(
            seen_nodes), "Val/test set filtering failed!"

    def bipartite_reindex(self):
        r"""
        Reindex the nodes in the bipartite graph, starting from 1.
        """
        self.df['dst'] = self.df['dst'].rank(
            method='dense', ascending=True).astype(int) 
        self.df['src'] = self.df['src'].rank(
            method='dense', ascending=True).astype(int) 
        self.df['src'] += self.df['dst'].max()

    def unipartite_reindex(self):
        r"""
        Reindex the nodes in the unipartite graph, starting from 1.
        """
        # To keep the index of destination nodes continuous, we first map the destination nodes to a continuous index starting from 0.
        node_mapping = {node: idx for idx,
                        node in enumerate(sorted(set(self.df.dst)))}
        src_set = set(self.df.src)
        for node in src_set:
            if node not in node_mapping:
                node_mapping[node] = len(node_mapping)
        self.df['src'] = self.df['src'].map(node_mapping)
        self.df['dst'] = self.df['dst'].map(node_mapping)
        self.df.reset_index(drop=True, inplace=True)
        self.df.src = self.df.src+1
        self.df.dst = self.df.dst+1


if __name__ == "__main__":
    # "ML-20M", "Taobao", "Yelp", "GoogleLocal",
    # "Flickr", "YouTube", "Patent", "WikiLink"
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='ML-20M', help='dataset name')
    parser.add_argument('--root', type=str, default='./data',
                        help='root directory to save the processed dataset')
    parser.add_argument('--all', action='store_true',
                        help='process all datasets', default=False)
    all_datasets = ["ML-20M", "Taobao", "Yelp", "GoogleLocal",
                    "Flickr", "YouTube", "Patent", "WikiLink"]
    args = parser.parse_args()
    root = args.root
    if args.all:
        for dataset in all_datasets:
            preprocessor = TGBSeqPreprocessor(dataset, root)
            print(f"Dataset {dataset} is processed and saved to {
                  root}/{dataset}.")
    else:
        preprocessor = TGBSeqPreprocessor(args.dataset, root)
        print(f"Dataset {args.dataset} is processed and saved to {
              root}/{args.dataset}.")
