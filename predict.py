import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse

from augment_utils import augment as _augment_base


#network parameters: (input size, layer size, number of layers)
MODEL_PARAMETERS = {
    "C1orf132": (273, 512, 6),
    "FHL2": (531, 512, 6),
    "ELOVL2_6": (531, 512, 6),
    "CCDC102B": (25, 256, 6),
    "ELOVL2_6_C1orf132": (531 + 273, 1024, 6),
    "ELOVL2_6_C1orf132_FHL2": (531 + 273 + 531, 1024, 6),
    "ELOVL2_6_C1orf132_CCDC102B": (531 + 273 + 25, 1024, 6),
    "ELOVL2_6_C1orf132_FHL2_CCDC102B": (531 + 273 + 531 + 25, 1024, 6)
}


class CustomDataset(Dataset):
    def __init__(self, df):
        """
        :param df: the data frame that include all of the 3 input-types, subsampled, and tag of the original sample
        """
        self.all_data = df
        self.all_data.dropna(inplace=True)
        self.tags = list(self.all_data['tag'].values)
        self.all_data.drop(columns=['tag'], inplace=True)
        self.data = torch.tensor(self.all_data.values, dtype=torch.float32)

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        return self.data[idx], self.tags[idx]


class Predictor(torch.nn.Module):
    def __init__(self, input_size, layer_size, layers_num=3):
        super().__init__()
        self.hidden = torch.nn.ModuleList()
        self.fc_in = torch.nn.Linear(input_size, layer_size)
        for i in range(layers_num):
            self.hidden.append(torch.nn.Linear(layer_size, layer_size))
        self.fc_out = torch.nn.Linear(layer_size, 1)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.fc_out(x)
        return x


def predict(args, df):
    marker = args.marker
    device = torch.device('cpu')
    input_size, layer_size, num_layers = MODEL_PARAMETERS[marker]
    predictor = Predictor(input_size, layer_size, num_layers)
    predictor.load_state_dict(torch.load("models/new_predictor_" + marker + '_' + str(input_size)))
    predictor.to(device)
    test_data = CustomDataset(df)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=4)
    with torch.no_grad():
        predictor.eval()
        batch_test, tags = next(iter(test_dataloader))    # predict all the samples at once
        out_test = predictor(batch_test)

        predictions_by_tag = dict()     # here all 128 subsamples are collected, by tag, to average them later
        for l in range(len(out_test)):
            tag = tags[l]
            original_tag = tag
            if original_tag in predictions_by_tag:
                predictions_by_tag[original_tag].append(out_test[l].item())
            else:
                predictions_by_tag[original_tag] = [out_test[l].item()]
        sorted_prediction_by_tag = sorted(list(predictions_by_tag.keys()))
        print('marker', 'tag', 'prediction', 'standard_deviation', '25', '50', '75', sep='\t')
        df_rows = []
        for tag in sorted_prediction_by_tag:
            predictions = predictions_by_tag[tag]
            mean_prediction = np.mean(predictions)    # mean of the 128 predictions
            std = np.round(np.std(predictions), decimals=2)
            percentile_25 = np.round(np.percentile(predictions, 25), 3)
            percentile_50 = np.round(np.percentile(predictions, 50), 3)
            percentile_75 = np.round(np.percentile(predictions, 75), 3)
            print(marker, tag, "%.1f" % mean_prediction, std, percentile_25, percentile_50, percentile_75,
                sep='\t')
            df_rows.append([tag, np.round(mean_prediction, 2), std, percentile_25, percentile_50, percentile_75])
        print('\n')
        if args.save_to_csv:
            df = pd.DataFrame(df_rows, columns=['tag', 'prediction', 'standard_deviation', '25', '50', '75'])
            df = md.merge(df, on='tag', how='left')
            df.sort_values(by='tag', inplace=True)
            df.to_csv(args.output_dir + "/results_" + marker + ".csv")


def augment(data_path):
    return _augment_base(data_path, include_age=False)


if __name__ == "__main__":
    print('start test script')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--marker', type=str)
    parser.add_argument('-dp', '--data_path', type=str)
    parser.add_argument('-s', '--save_to_csv', type=bool, default=False)
    parser.add_argument('-o', '--output_dir', type=str, default=".")
    args = parser.parse_args()
    df = augment(args.data_path)
    predict(args, df)
