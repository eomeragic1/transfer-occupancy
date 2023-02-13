import os


def hpo():
    lr_list = [0.05, 0.01, 0.005]
    batch_size_list = [256]
    epoch_list = [30, 45]
    hidden_size_list = [64, 128]
    model_list = ['ConvNet', 'LSTM', 'GRU']
    dataset_name = 'ECO'

    for lr in lr_list:
        for batch_size in batch_size_list:
            for hidden_size in hidden_size_list:
                for model in model_list:
                    for epoch in epoch_list:
                        cl_builder = f'python -u ./main.py --model {model} --dataset {dataset_name} \
                                --lr {lr} --batch_size {batch_size} --hidden_size {hidden_size} \
                                --epoch {epoch} --train_days 190 --val_days 15'

                        os.system(cl_builder)


if __name__ == '__main__':
    hpo()
