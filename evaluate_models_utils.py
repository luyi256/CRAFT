import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json
from utils.metrics import get_link_prediction_metrics
from models.EdgeBank import edge_bank_link_prediction
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler, TIME_SLOT_DICT
from tgb_seq.LinkPred.evaluator import Evaluator 
from utils.DataLoader import Data

def evaluate_model_link_prediction_multi_negs(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module, device: str = 'cpu',
                                   num_neighbors: int = 20, time_gap: int = 2000, mode='test', num_negs=100, loss_type = 'BCE', full_data: Data = None, dataset_name: str = None, collision_check: bool = False, analyze_results: bool = False):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()
    if model_name not in ['CRAFT']:
        model[0].set_neighbor_sampler(neighbor_sampler)
    
    model.eval()

    evaluator=Evaluator()

    if evaluate_data.neg_samples is not None:
        num_negs = evaluate_data.num_neg_samples

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        neg_samples_idx = 0
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]
            if mode in ['test']:
                to_test_mask=evaluate_data.split[evaluate_data_indices]==2
            else:
                to_test_mask=evaluate_data.split[evaluate_data_indices]==1
            test_neg_sample_idx = np.arange(neg_samples_idx, neg_samples_idx + to_test_mask.sum())
            neg_samples_idx += to_test_mask.sum()
            repeated_batch_src_node_ids = np.repeat(batch_src_node_ids, repeats=num_negs, axis=0)
            repeated_batch_dst_node_ids = np.repeat(batch_dst_node_ids, repeats=num_negs, axis=0)
            repeated_batch_dst_node_ids_reshape = repeated_batch_dst_node_ids.reshape(-1, num_negs)
            original_batch_size = batch_src_node_ids.shape[0]
            if evaluate_data.neg_samples is not None:
                # since tgb-seq neg samples are only provided for test sample, 
                test_neg_dst_node_ids = evaluate_data.neg_samples[test_neg_sample_idx]
                batch_neg_dst_node_ids = np.zeros((original_batch_size, num_negs), dtype=np.int32)
                batch_neg_dst_node_ids[to_test_mask] = test_neg_dst_node_ids
                if (~to_test_mask).sum() > 0:
                    if collision_check:
                        not_test_neg_dst_node_ids = evaluate_neg_edge_sampler.sample_with_time_collision_check(num_negs, batch_src_node_ids[~to_test_mask], batch_node_interact_times[~to_test_mask], neighbor_sampler)
                    else:
                        _, not_test_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(num_negs*len(batch_src_node_ids[~to_test_mask]))
                        not_test_neg_dst_node_ids = not_test_neg_dst_node_ids.reshape(-1, num_negs)
                    batch_neg_dst_node_ids[~to_test_mask] = not_test_neg_dst_node_ids
            else:
                if collision_check:
                    batch_neg_dst_node_ids=evaluate_neg_edge_sampler.sample_with_time_collision_check(num_negs, batch_src_node_ids, batch_node_interact_times, neighbor_sampler)
                else:
                    _, batch_neg_dst_node_ids=evaluate_neg_edge_sampler.sample(num_negs*len(batch_src_node_ids))
                    batch_neg_dst_node_ids=batch_neg_dst_node_ids.reshape(-1, num_negs)
            batch_neg_dst_node_ids=batch_neg_dst_node_ids.flatten()
            repeated_batch_node_interact_times = np.repeat(batch_node_interact_times, repeats=num_negs, axis=0)

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=repeated_batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_node_interact_times=np.concatenate([batch_node_interact_times, batch_node_interact_times, repeated_batch_node_interact_times], axis=0)
                batch_src_node_embeddings, batch_dst_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        neg_dst_node_ids=batch_neg_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        edge_ids=batch_edge_ids,
                                                                        edges_are_positive=False,
                                                                        num_neighbors=num_neighbors)
                batch_neg_src_node_embeddings=torch.repeat_interleave(batch_src_node_embeddings, repeats=num_negs, dim=0)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                # for i in range(len(repeated_batch_src_node_ids)):
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=repeated_batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                src_node_ids = np.concatenate([batch_src_node_ids, repeated_batch_src_node_ids], axis=0)
                dst_node_ids = np.concatenate([batch_dst_node_ids, batch_neg_dst_node_ids], axis=0)
                node_interact_times = np.concatenate([batch_node_interact_times, repeated_batch_node_interact_times], axis=0)
                src_node_embeddings, dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times)
                batch_src_node_embeddings, batch_dst_node_embeddings = src_node_embeddings[:len(batch_src_node_ids)], dst_node_embeddings[:len(batch_dst_node_ids)]
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = src_node_embeddings[len(batch_src_node_ids):], dst_node_embeddings[len(batch_dst_node_ids):]
            elif model_name in ['SASRec', 'SGNNHN']:
                neighbor_node_ids, _, _=neighbor_sampler.get_historical_neighbors_left(node_ids=batch_src_node_ids, node_interact_times=batch_node_interact_times, num_neighbors=num_neighbors)
                neighbor_num=(neighbor_node_ids!=0).sum(axis=1)
                batch_neg_dst_node_ids = batch_neg_dst_node_ids.reshape(original_batch_size,-1)
                pos_item = torch.from_numpy(batch_dst_node_ids)
                neg_item = torch.from_numpy(batch_neg_dst_node_ids)
                items = torch.cat([pos_item.unsqueeze(1), neg_item], dim=1)
                batch_data=[torch.from_numpy(neighbor_node_ids), torch.from_numpy(neighbor_num), items]
                positive_probabilities, negative_probabilities = model[0].predict(batch_data)
                negative_probabilities = negative_probabilities.flatten().cpu().numpy()
                positive_probabilities = positive_probabilities.flatten().cpu().numpy()
            elif model_name in ['CRAFT']:
                src_neighb_seq, _, src_neighb_interact_times=neighbor_sampler.get_historical_neighbors_left(node_ids=batch_src_node_ids, node_interact_times=batch_node_interact_times, num_neighbors=num_neighbors)
                neighbor_num=(src_neighb_seq!=0).sum(axis=1)
                batch_neg_dst_node_ids = batch_neg_dst_node_ids.reshape(original_batch_size,-1)
                pos_item = torch.from_numpy(batch_dst_node_ids)
                neg_item = torch.from_numpy(batch_neg_dst_node_ids)
                test_dst = torch.cat([pos_item.unsqueeze(1), neg_item], dim=1)
                dst_last_neighbor, _, dst_last_update_time = neighbor_sampler.get_historical_neighbors_left(node_ids=test_dst.flatten(), node_interact_times=np.broadcast_to(batch_node_interact_times[:,np.newaxis], (len(batch_node_interact_times), test_dst.shape[1])).flatten(), num_neighbors=1)
                dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
                dst_last_update_time[dst_last_neighbor.reshape(len(test_dst),-1)==0]=-100000
                dst_last_update_time = torch.from_numpy(dst_last_update_time)
                positive_probabilities, negative_probabilities = model.predict(src_neighb_seq=torch.from_numpy(src_neighb_seq), 
                                                                src_neighb_seq_len=torch.from_numpy(neighbor_num), 
                                                                src_neighb_interact_times=torch.from_numpy(src_neighb_interact_times), 
                                                                cur_pred_times=torch.from_numpy(batch_node_interact_times), 
                                                                test_dst=test_dst,
                                                                dst_last_update_times=dst_last_update_time)
                negative_probabilities = negative_probabilities.flatten().cpu().numpy()
                positive_probabilities = positive_probabilities.flatten().cpu().numpy()
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            if model_name not in ['CRAFT', 'SASRec', 'SGNNHN']:
                if 'BCE' in loss_type:
                # get positive and negative probabilities, shape (batch_size, )
                    positive_probabilities = model[1](
                        input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid().cpu().numpy()
                    # get negative probabilities, Tensor, shape (batch_size * num_negative_samples_per_node, )
                    negative_probabilities = model[1](
                        input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid().cpu().numpy()
                elif loss_type == 'BPR':
                    positive_probabilities = model[1](
                        input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).cpu().numpy()
                    negative_probabilities = model[1](
                        input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).cpu().numpy()
            batch_neg_dst_node_ids = batch_neg_dst_node_ids.reshape(original_batch_size,-1)
            if to_test_mask.sum() == 0:
                continue
            positive_probabilities = np.atleast_1d(positive_probabilities)[to_test_mask]
            negative_probabilities = negative_probabilities.reshape(-1,num_negs)[to_test_mask]
            mrr_list = evaluator.eval(y_pred_pos=positive_probabilities, y_pred_neg=negative_probabilities)
            evaluate_metrics.extend(mrr_list)
    return {'mrr': np.mean(evaluate_metrics)}


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module, device: str = 'cpu',
                                   num_neighbors: int = 20, time_gap: int = 2000,mode:str='val', loss_type = 'BCE', full_data: Data = None, collision_check: bool = False,dataset_name:str=''):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()
    if model_name not in ['CRAFT']:
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(
            evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]
            if mode in ['test']:
                to_test_mask=evaluate_data.split[evaluate_data_indices]==2
            else:
                to_test_mask=evaluate_data.split[evaluate_data_indices]==1
            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                if collision_check:
                    batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample_with_time_collision_check(num_negs=1, batch_src_node_ids=batch_src_node_ids, batch_node_interact_times=batch_node_interact_times, neighbor_sampler=neighbor_sampler).flatten()
                    batch_neg_src_node_ids = batch_src_node_ids
                else:
                    batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(
                        size=len(batch_src_node_ids))

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_node_interact_times=np.concatenate([batch_node_interact_times,batch_node_interact_times,batch_node_interact_times],axis=0)
                batch_src_node_embeddings, batch_dst_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          neg_dst_node_ids=batch_neg_dst_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=num_neighbors)
                batch_neg_src_node_embeddings=batch_src_node_embeddings
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                src_node_ids = np.concatenate([batch_src_node_ids, batch_neg_src_node_ids], axis=0)
                dst_node_ids = np.concatenate([batch_dst_node_ids, batch_neg_dst_node_ids], axis=0)
                node_interact_times = np.concatenate([batch_node_interact_times, batch_node_interact_times], axis=0)
                src_node_embeddings, dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times)
                batch_src_node_embeddings, batch_neg_src_node_embeddings = src_node_embeddings[:len(batch_src_node_ids)], src_node_embeddings[len(batch_src_node_ids):]
                batch_dst_node_embeddings, batch_neg_dst_node_embeddings = dst_node_embeddings[:len(batch_dst_node_ids)], dst_node_embeddings[len(batch_dst_node_ids):]
            elif model_name in ['SASRec', 'SGNNHN']:
                neighbor_node_ids, _, _=neighbor_sampler.get_historical_neighbors_left(node_ids=batch_src_node_ids, node_interact_times=batch_node_interact_times, num_neighbors=num_neighbors)
                neighbor_num=(neighbor_node_ids!=0).sum(axis=1)
                pos_item = torch.from_numpy(batch_dst_node_ids)
                neg_item = torch.from_numpy(batch_neg_dst_node_ids)
                items = torch.cat([pos_item.unsqueeze(1), neg_item.unsqueeze(1)], dim=1)
                batch_data=[torch.from_numpy(neighbor_node_ids), torch.from_numpy(neighbor_num), items]
                positive_probabilities, negative_probabilities = model[0].predict(batch_data)
                negative_probabilities = negative_probabilities.flatten()
            elif model_name in ['CRAFT']:
                src_neighb_seq, _, src_neighb_interact_times=neighbor_sampler.get_historical_neighbors_left(node_ids=batch_src_node_ids, node_interact_times=batch_node_interact_times, num_neighbors=num_neighbors)
                neighbor_num=(src_neighb_seq!=0).sum(axis=1)
                pos_item = torch.from_numpy(batch_dst_node_ids)
                neg_item = torch.from_numpy(batch_neg_dst_node_ids)
                test_dst = torch.cat([pos_item.unsqueeze(1), neg_item.unsqueeze(1)], dim=1)
                dst_last_neighbor, _, dst_last_update_time = neighbor_sampler.get_historical_neighbors_left(node_ids=test_dst.flatten(), node_interact_times=np.broadcast_to(batch_node_interact_times[:,np.newaxis], (len(batch_node_interact_times), test_dst.shape[1])).flatten(), num_neighbors=1)
                dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
                dst_last_update_time[dst_last_neighbor.reshape(len(test_dst),-1)==0]=-100000
                dst_last_update_time = torch.from_numpy(dst_last_update_time)
                positive_probabilities, negative_probabilities = model.predict(
                                        src_neighb_seq=torch.from_numpy(src_neighb_seq), 
                                        src_neighb_seq_len=torch.from_numpy(neighbor_num), 
                                        src_neighb_interact_times=torch.from_numpy(src_neighb_interact_times), 
                                        cur_pred_times=torch.from_numpy(batch_node_interact_times), 
                                        test_dst=test_dst, 
                                        dst_last_update_times=dst_last_update_time)
                negative_probabilities = negative_probabilities.flatten()
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            if to_test_mask.sum() == 0:
                continue
            if model_name not in ['CRAFT', 'SASRec', 'SGNNHN']:
                if 'BCE' in loss_type:
                # get positive and negative probabilities, shape (batch_size, )
                    positive_probabilities = model[1](
                        input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                    negative_probabilities = model[1](
                        input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                elif loss_type == 'BPR':
                    positive_probabilities = model[1](
                        input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1)
                    negative_probabilities = model[1](
                        input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1)
            positive_probabilities = positive_probabilities[to_test_mask]
            negative_probabilities = negative_probabilities[to_test_mask]
            predicts = torch.cat(
                [positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(
                positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
            if loss_func is not None:
                evaluate_losses.append(loss_func(positive_probabilities, negative_probabilities).item())
            evaluate_metrics.append(get_link_prediction_metrics(
                predicts=predicts, labels=labels))
    
    return_metrics={}
    for metric_name in evaluate_metrics[0].keys():
        average_test_metric = np.mean(
            [test_metric[metric_name] for test_metric in evaluate_metrics])
        return_metrics[metric_name] = average_test_metric
    if len(evaluate_losses)>0:
        return_metrics["val_loss"] = np.mean(evaluate_losses)
    return evaluate_losses, return_metrics
