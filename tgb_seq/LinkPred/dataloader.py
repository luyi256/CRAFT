import os
import os.path as osp
from tgb_seq.datasets.datasets_info import DATA_VERSION, BColors, DATA_NAME
import zipfile
import subprocess
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download


class TGBSeqLoader(object):
    def __init__(self, name: str, root: str):
        self._name = name
        self._root = root
        self._file_dir = os.path.join(self._root, self._name)
        if self._name in DATA_VERSION and DATA_VERSION[self._name]!="0":
            self._version = DATA_VERSION[self._name]
            self._downloaded_name=f"{self._name}_v{self._version}"
            self._edgelist_path = os.path.join( 
                self._root, self._name, f'{self._name}_v{self._version}.csv')
            self._test_ns_path = os.path.join(self._root, self._name, f'{self._name}_test_ns_v{self._version}.npy')
            self._edge_feat_path = os.path.join(self._root, self._name, f'{self._name}_edge_feat_v{self._version}.npy')
            self._node_feat_path = os.path.join(self._root, self._name, f'{self._name}_node_feat_v{self._version}.npy')
        else:
            self._version = ''
            self._downloaded_name=f"{self._name}"
            self._edgelist_path = os.path.join(
                self._root, self._name, f'{self._name}.csv')
            self._test_ns_path = os.path.join(
                self._root, self._name, f'{self._name}_test_ns.npy')
            self._edge_feat_path = os.path.join(
                self._root, self._name, f'{self._name}_edge_feat.npy')
            self._node_feat_path = os.path.join(
                self._root, self._name, f'{self._name}_node_feat.npy')
        self._load_file()

    def _download(self):
        if not self._name in DATA_NAME:
            raise ValueError(f'Dataset {self._name} not supported by TGB-Seq.')
        print(f"{BColors.WARNING}Download started, this might take a while . . . {BColors.ENDC}")
        if not osp.isdir(self._root):
            os.makedirs(self._root)
        print(f"Dataset {self._name} will be downloaded in ", self._root)
        try:
            filename=hf_hub_download(repo_id=f"TGB-Seq/{self._name}",filename=f"{self._downloaded_name}.csv",local_dir=self._file_dir,repo_type="dataset")
            test_ns_filename=hf_hub_download(repo_id=f"TGB-Seq/{self._name}",filename=f"{self._downloaded_name}_test_ns.npy",local_dir=self._file_dir,repo_type="dataset")
        except Exception as e:
            print(f"Error: {e}")
            exit()
        try:
            self._check_version()
        except VersionNotMatchError as e:
            print(f'Downloaded dataset {self._name} not match with TGB-Seq version requirements. Please report the issue to TGB-Seq developers')

    def _load_file(self):
        if os.path.exists(self._file_dir):
            try:
                self._check_version()
            except Warning as e:
                print(e)
            except Exception as e:
                print(e)
                self._download()
        else:
            os.makedirs(self._file_dir)
            self._download()
        self._edgelist_df = pd.read_csv(self._edgelist_path)
        self._edge_feat, self._node_feat, self._test_ns = None, None, None
        if os.path.exists(self._test_ns_path):
            self._test_ns = np.load(self._test_ns_path)
        if os.path.exists(self._edge_feat_path):
            self._edge_feat = np.load(self._edge_feat_path)
        if os.path.exists(self._node_feat_path):
            self._node_feat = np.load(self._node_feat_path)
        self._src_node_ids = self._edgelist_df['src'].values.astype(np.longlong)
        self._dst_node_ids = self._edgelist_df['dst'].values.astype(np.longlong)
        self._time = self._edgelist_df['time'].values.astype(np.float64)
        self._train_mask = self._edgelist_df['split'] == 0
        self._val_mask = self._edgelist_df['split'] == 1
        self._test_mask = self._edgelist_df['split'] == 2
        self._split = self._edgelist_df['split'].values.astype(np.int32)

    def _check_version(self):
        if not os.path.exists(self._edgelist_path):
            raise FileNotFoundError(f'Local dataset file {self._edgelist_path} not found')
        if not os.path.exists(self._test_ns_path):
            raise Warning(f'Local test negative samples file {self._test_ns_path} not found. We will generate it when testing.')

    @property
    def train_mask(self) -> np.ndarray:
        r"""
        Returns the train mask of the dataset
        """
        if self._train_mask is None:
            raise ValueError("training split hasn't been loaded")
        return self._train_mask

    @property
    def val_mask(self) -> np.ndarray:
        r"""
        Returns the train mask of the dataset
        """
        if self._val_mask is None:
            raise ValueError("training split hasn't been loaded")
        return self._val_mask

    @property
    def test_mask(self) -> np.ndarray:
        r"""
        Returns the train mask of the dataset
        """
        if self._test_mask is None:
            raise ValueError("training split hasn't been loaded")
        return self._test_mask

    @property
    def src_node_ids(self) -> np.ndarray:
        r"""
        Returns the source node ids of the dataset
        """
        if self._src_node_ids is None:
            raise ValueError("source node ids hasn't been loaded")
        return self._src_node_ids

    @property
    def dst_node_ids(self) -> np.ndarray:
        r"""
        Returns the source node ids of the dataset
        """
        if self._dst_node_ids is None:
            raise ValueError("destination node ids hasn't been loaded")
        return self._dst_node_ids

    @property
    def node_interact_times(self) -> np.ndarray:
        r"""
        Returns the node interaction times of the dataset
        """
        if self._time is None:
            raise ValueError("node interaction times hasn't been loaded")
        return self._time

    @property
    def edge_features(self) -> np.ndarray:
        r"""
        Returns the edge features of the dataset
        """
        if self._edge_feat is None:
            return None
        return self._edge_feat

    @property
    def node_features(self) -> np.ndarray:
        r"""
        Returns the node features of the dataset
        """
        if self._node_feat is None:
            return None
        return self._node_feat

    @property
    def edgelist(self) -> np.ndarray:
        r"""
        Returns the dataframe of edge list
        """
        if self._edgelist_df is None:
            return None
        return self._edgelist_df

    @property
    def negative_samples(self) -> np.ndarray:
        r"""
        Returns the dataframe of edge list
        """
        if self._test_ns is None:
            return None
        return self._test_ns
    
    @property
    def split(self) -> np.ndarray:
        r"""
        Returns the split of the dataset
        """
        if self._split is None:
            return None
        return self._split
         
         
class VersionNotMatchError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def main():
    name = "GoogleLocal"
    root = "./data/"
    loader = TGBSeqLoader(name, root)
    print(loader._edgelist_path)


if __name__ == "__main__":
    main()
