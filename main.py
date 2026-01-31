from model.base import BaseTrainer
from shared_util.evaluation_metrics import *
import argparse
from shared_util.file_handler import FileHandler
from shared_util.seed import *
import matplotlib

if __name__ == '__main__':
    matplotlib.set_loglevel("warning")
    seed_everything()
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()

    window_size = 10
    data_base_path = '/home/yzz/RCL/crossSysFC-23FC-bk2'
    parser.add_argument("--dataset_path_SN", default=f'{data_base_path}/../SN_Dataset/dataset/merge/window_size_{window_size}.pkl', type=str)
    parser.add_argument("--dataset_path_TT", default=f'{data_base_path}/../TT_Dataset/dataset/merge/window_size_{window_size}.pkl', type=str)
    parser.add_argument("--model_path", default=f'{FileHandler.set_folder(data_base_path + "/model/test")}/150.pt', type=str)
    parser.add_argument("--prototype_path", default=f'{FileHandler.set_folder(data_base_path + "/model/test/proto")}/prototypes.pt', type=str)
    parser.add_argument("--llm_embedding", default=1024, type=int)
    parser.add_argument("--window_size", default=window_size, type=int)

    parser.add_argument("--gpu", default=2)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--service_accuracy_th", default=0.2, type=float)

    parser.add_argument("--orl_te_heads", default=4, type=int)
    parser.add_argument("--orl_te_layers", default=2, type=int)
    parser.add_argument("--orl_te_in_channels", default=128, type=int)

    parser.add_argument("--efi_in_dim", default=128, type=int)
    parser.add_argument("--efi_te_heads", default=2, type=int)
    parser.add_argument("--efi_te_layers", default=1, type=int)
    parser.add_argument("--efi_out_dim", default=64 * 4, type=int)

    parser.add_argument("--eff_in_dim", default=64 * 4, type=int)
    parser.add_argument("--eff_out_dim", default=128, type=int)

    parser.add_argument("--ec_fault_types", default=4, type=int)

    parser.add_argument("--train", action='store_true', default=False)

    params = vars(parser.parse_args())

    rca_data_trainer = BaseTrainer(params)

    if params['train']:
        rca_data_trainer.train()
    else:
        rca_data_trainer.evaluate_rca_d3()


