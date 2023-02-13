import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_days', type=int, default=12)
    parser.add_argument('--eval_transfer', type=bool, default=True)
    parser.add_argument('--seeds', nargs='+', default=[1111])
    args = parser.parse_args()
    train_days = args.train_days
    eval_transfer = args.eval_transfer
    seeds = args.seeds

    model = 'LSTM'
    dataset_name = 'HPDMobile'
    lr = 0.05
    transfer_lr_list = [0.05]
    batch_size = 512
    hidden_size = 32
    epoch = 25
    transfer_epoch = 10
#    seeds = [1111, 22, 333, 4444, 5555, 66666, 77777, 888888, 9999999]
#     for seed in seeds:
#         cl_builder_source = f'python -u ./main.py --model {model} --dataset {dataset_name} \
#                                         --lr {lr} --batch_size {batch_size} --hidden_size {hidden_size} \
#                                         --epoch {epoch} --val_days 0.1 --save_model False --visualize False \
#                                         --buildings "Household 02" "Household 04" "Household 05" --seed {seed}'
#
#
#         os.system(cl_builder_source)
#    for num_frozen_layers, transfer_lr in zip([2], transfer_lr_list):
#        train_days_list = [3, 6, 9, 12, 15]
#        for train_days in train_days_list:
#        for seed in seeds:
#            print(f'Train days: {train_days}')
#            cl_builder_transfer = f'python -u ./main.py --model {model} --dataset {dataset_name} \
 #                                           --lr {lr} --batch_size {batch_size} --hidden_size {hidden_size} \
#                                            --epoch {epoch} --train_days {train_days} --val_days 0.1 --visualize True --transfer True \
#                                  --transfer_path ../model_checkpoints/{dataset_name}_{model}_2023-01-31_seed_888888_traindays_None_9189.pt \
#                                   --buildings "Household 01" "Household 04" "Household 06" --seed {seed} --num_frozen_layers 0'

 #           os.system(cl_builder_transfer)

    if eval_transfer:
        for num_frozen_layers, transfer_lr in zip([0], transfer_lr_list):
            # train_days_list = [2, 4, 6, 8, 10]
            # for train_days in train_days_list:
            for seed in seeds:
                print(f'Train days: {train_days}')
                cl_builder_transfer = f'python -u ./main.py --model {model} --dataset {dataset_name} \
                                                --lr {transfer_lr} --batch_size {batch_size} --hidden_size {hidden_size} \
                                                --epoch {transfer_epoch} --train_days {train_days} --val_days 5 --visualize True --transfer True \
                                      --transfer_path ../model_checkpoints/HPDMobile_LSTM_2023-01-31_seed_888888_traindays_None_9189.pt \
                                       --buildings "Household 01" "Household 04" "Household 06" --seed {seed} --num_frozen_layers {num_frozen_layers}'

                os.system(cl_builder_transfer)
