# CRAFT

CRAFT is a simple yet effective architecture for future link prediction, without the commonly adopted memory or aggregation modules in prior temporal graph learning models. This repository provides the code of CRAFT and baselines used in the paper. The code is build upon [DyGLib](https://github.com/yule-BUAA/DyGLib) and we also adopt several modifications from [TGB-Seq](https://github.com/TGB-Seq/TGB-Seq). For simplicity, we have included the source code of the newest version of the TGB and TGB-Seq benchmark in this repository.

## Get Started
We provide a simple example to train CRAFT on the mooc dataset. The mooc datasets is already provided in `./data/`. We also include the log of this experiment in `./logs/` for reference.

```shell
python train_link_prediction.py --dataset_name mooc --model_name CRAFT --batch_size 200 --gpu 1 --num_neighbors 30 --embedding_size 64 --loss BPR --hidden_dropout 0.1 --attn_dropout_prob 0.1 --emb_dropout_prob 0.1 --shuffle --output_cat_time_intervals --num_output_layer 2 --num_layers 2 --output_cat_repeat_times --use_pos &
```

The specific arguments of CRAFT are as follows:
- --num_neighbors: the number of neighbors used for each source
- --embedding_size: embedding size of the node embeddings, 64 for small datasets (wikipedia, reddit, mooc, lastfm, uci, Flights) and 128 for large datasets (all other datasets)
- --output_cat_repeat_times: using repeat time encoding
- --output_cat_time_intervals: using elapsed time encoding
- --use_pos: using positional encoding
- --num_output_layer: number of output layers, we use 2 layers for all experiments
- --num_layers: number of layers of cross-attention
- --hidden_dropout: the dropout rate for FFNs and MLPs
- --attn_dropout_prob: the dropout rate for attention scores
- --emb_dropout_prob: the dropout rate for node embeddings

## General Commands to Reproduce the Experiments Results
The general command to train CRAFT is as follows. Please refer to Table 6 in the appendix for the batch size, the embedding size and optimal hyperparameters for each dataset. To train CRAFT-R for seen-dominant datasets, please add `--output_cat_repeat_times` to the command.
```shell
python train_link_prediction.py --dataset_name $dataset_name --model_name CRAFT --batch_size $batch_size --gpu $gpu --num_neighbors $num_neighbors --embedding_size $embedding_size --loss BPR --hidden_dropout $hidden_dropout --attn_dropout_prob $attn_dropout_prob --emb_dropout_prob $emb_dropout_prob --shuffle --output_cat_time_intervals --num_output_layer 2 --num_layers $num_layers --use_pos (--output_cat_repeat_times) &
```

## Ablation Study
To compare the effect of positional encoding and time encoding, please replace `--use_pos` with `--input_cat_time_intervals` in the command. 

To compare the effect of BPR loss and BCE loss, please replace `--loss BPR` with `--loss BCE` in the command.