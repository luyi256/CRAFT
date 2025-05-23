import numpy as np
class NegativeSampler(object):
  def __init__(
        self,
        dataset_name: str,
        first_dst_id: int,
        last_dst_id: int, # one must keep the index of destination nodes continuous
        num_neg_e: int = 100,  # number of negative destinations sampled per positive sample
        strategy: str = "rnd", 
        rnd_seed: int = 0
    ) -> None:
    self.rnd_seed = rnd_seed
    np.random.seed(self.rnd_seed)
    self.dataset_name = dataset_name

    self.first_dst_id = first_dst_id
    self.last_dst_id = last_dst_id
    self.num_neg_e = num_neg_e
    assert strategy in [
        "rnd",
    ], "Only `rnd` is supported"
    self.strategy = strategy

  def sample_neg(self, src_ids: np.ndarray, dst_ids: np.ndarray, file_name=None, store=False, collision_check=False) -> np.ndarray:
    """
    Sample negative edges
    :param src_ids: np.ndarray, source node ids
    :param dst_ids: np.ndarray, destination node ids
    :return: np.ndarray, negative edges
    """
    neg_dst_ids = np.random.randint(
        low=self.first_dst_id, high=self.last_dst_id, size=(len(src_ids), self.num_neg_e)
    ).astype(np.int64)
    if collision_check:
      pos_dst_ids = np.tile(dst_ids, (self.num_neg_e, 1)).T
      mask=pos_dst_ids==neg_dst_ids
      while np.any(mask):
        mask_rows=np.where(mask)[0]
        num_mask_rows=len(mask_rows)
        neg_dst_ids[mask_rows]=np.random.randint(
            low=self.first_dst_id, high=self.last_dst_id, size=(num_mask_rows, self.num_neg_e)
        )
        mask=pos_dst_ids==neg_dst_ids
    if store:
      np.save(file_name, neg_dst_ids)
    return neg_dst_ids