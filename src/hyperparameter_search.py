import os


def hpo():
    lr_list = [0.01, 0.005, 0.001, 0.0005]
    batch_size_list = [256, 128, 64]
    epoch_list = [30]
    hidden_size_list = [16, 32, 64]
    model_list = ['LSTM', 'GRU', 'ConvNet']
    dataset_name = 'ROBOD'

    for lr in lr_list:
        for batch_size in batch_size_list:
            for hidden_size in hidden_size_list:
                for model in model_list:
                    for epoch in epoch_list:
                        cl_builder = f'python -u ./main.py --model {model} --dataset {dataset_name} \
                                --lr {lr} --batch_size {batch_size} --hidden_size {hidden_size} \
                                --epoch {epoch} --train_days 90 --val_days 10'

                        os.system(cl_builder)


if __name__ == '__main__':
    hpo()
