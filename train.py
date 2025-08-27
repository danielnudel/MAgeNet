import argparse
import numpy as np
import pandas as pd
import sys
from random import randrange, sample
import copy

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F


np.seterr(divide='ignore')
criterion = torch.nn.L1Loss()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loss = []
train_loss = []
validation_loss = []
median_errors = []
errors_per_epochs = []

batch_list = []
labels = []

CONCATINATED_SETS_NAMES = {
    "ELOVL2_6_C1orf132": 512 + 256,
    "ELOVL2_6_C1orf132_FHL2": 512 + 256 + 512,
    "ELOVL2_6_C1orf132_CCDC102B": 512 + 256 + 16,
    "ELOVL2_6_C1orf132_FHL2_CCDC102B": 512 + 256 + 512 + 16
}


locus_parameters = {
    "ELOVL2_6": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "C1orf132": [2, 3, 4, 5, 6, 7, 8, 9],
    "FHL2": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "CCDC102B": [0, 1, 2, 3]
}


class Predictor(torch.nn.Module):
    def __init__(self, input_size, layer_size, layers_num=6, dropout=0.01):
        super().__init__()
        self.hidden = torch.nn.ModuleList()
        self.fc_in = torch.nn.Linear(input_size, layer_size)
        for i in range(layers_num):
            self.hidden.append(torch.nn.Linear(layer_size, layer_size))
        self.fc_out = torch.nn.Linear(layer_size, 1)
        self.drop = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.drop(self.fc_in(x)))
        for layer in self.hidden:
            x = F.relu(self.drop(layer(x)))
        x = self.fc_out(x)
        return x


class EarlyStopping():
    def __init__(self, conseq_loss_steps, name="early_stopped", min_loss=100000, trunc_val=0.001):
        self.conseq_loss_steps = conseq_loss_steps
        self.min_loss = min_loss
        self.last_loss = 0
        self.checkpoint_loss = 0
        self.counter = 0
        self.model = None
        self.model_name = name
        self.avg = 0
        self.trunc_val = trunc_val
        self.predictor = None

    def __call__(self, loss_val, loss_train, predictor=None):
        loss = loss_val - loss_val % self.trunc_val
        if loss >= self.min_loss:
            self.counter += 1
        else:
            self.counter = 0
            self.predictor = copy.deepcopy(predictor)
        if self.counter > self.conseq_loss_steps:
            self.checkpoint_loss = loss
            torch.save(self.predictor.state_dict(), "./models/new_predictor_" + self.model_name)
            return True
        if loss < self.min_loss:
            self.min_loss = loss
        self.last_loss = loss
        return False

    def get_loss(self):
        return self.min_loss


class CustomDataset(Dataset):
    def __init__(self, df):
        """
        :param df: the data frame that include all of the 3 input-types, subsampled, and tag of the original sample
        """
        self.all_data = df.copy()
        self.all_data.dropna(inplace=True)
        self.tags = list(self.all_data['tag'].values)
        self.labels = list(self.all_data['age'].values)
        self.all_data.drop(columns=['tag', 'age'], inplace=True)
        self.data = torch.tensor(self.all_data.values, dtype=torch.float32)

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.tags[idx]


def augment(data_path):
    READS_PER_AUGMENTED_SAMPLE = 8192
    NUM_SUB_SAMPLES = 128
    data = pd.read_csv(data_path)
    ages = []
    original_total_reads = []
    tags = []
    boot_rows = np.zeros((len(data.index) * NUM_SUB_SAMPLES, len(data.columns) - 3))
    row_counter = 0
    for i, row in data.iterrows():
        age = row['age']
        total_reads = row['total_reads_origin']
        tag = row['tag']
        probabilities = row.values[:-3] / total_reads
        for i in range(NUM_SUB_SAMPLES):
            out = np.random.multinomial(READS_PER_AUGMENTED_SAMPLE, probabilities)
            boot_rows[row_counter, :] = out
            ages.append(age)
            original_total_reads.append(total_reads)
            tags.append(tag)
            row_counter += 1
    df = pd.DataFrame(boot_rows, columns=data.columns[:-3])
    fixed_columns = df.columns
    df['age'] = ages
    df['total_reads_origin'] = original_total_reads
    df['tag'] = tags

    num_sites = len(data.columns[0])
    for i in range(num_sites + 1):
        columns_of_interest = [c for c in fixed_columns if c.count('C') == i]
        df["C_count_" + str(i)] = df[columns_of_interest].sum(axis=1)
    for site in range(num_sites):
        columns_of_interest = [c for c in fixed_columns if c[site] == 'C']
        df["site_" + str(site + 1)] = df[columns_of_interest].sum(axis=1)
    columns_order = [c for c in df.columns if "site_" in c] + \
                    [c for c in df.columns if "C_count_" in c] + \
                    sorted(list(data.columns[:-3]))
    df = df[columns_order + ['tag', 'age']]
    return df


def test(args, df):
    print("Running test")
    predictor = Predictor(args.input_size, args.layer_size, args.num_layers, dropout=args.dropout)
    predictor.load_state_dict(torch.load("./models/new_predictor_" + args.marker + '_' + str(args.input_size)))
    predictor.to(device)
    test_data = CustomDataset(df)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True, num_workers=4)
    with torch.no_grad():
        predictor.eval()
        batch_test, labels_test, tags = next(iter(test_dataloader))
        out_test = predictor(batch_test)
        loss_test = criterion(out_test, labels_test[:, None])
        predictions_by_tag = dict()
        tag_to_age_dict = {}
        for l in range(len(out_test)):
            tag = tags[l].item()
            tag_to_age_dict[tag] = labels_test[l].item()
            original_tag = tag
            if original_tag in predictions_by_tag:
                predictions_by_tag[original_tag].append(out_test[l].item())
            else:
                predictions_by_tag[original_tag] = [out_test[l].item()]
        errors = []
        sorted_prediction_by_tag = sorted(list(predictions_by_tag.keys()))
        print('sample', 'age', 'prediction', 'standard_deviation', '25', '50', '75', sep='\t')
        df_rows = []
        for tag in sorted_prediction_by_tag:
            predictions = predictions_by_tag[tag]
            predictions.sort()
            mean_prediction = np.mean(predictions)
            std = np.around(np.std(predictions), decimals=2)
            percentile_25 = np.percentile(predictions, 25)
            percentile_50 = np.percentile(predictions, 50)
            percentile_75 = np.percentile(predictions, 75)
            real_age = tag_to_age_dict[tag]
            print(tag, np.round(real_age, 2), "%.1f" % mean_prediction, std, percentile_25, percentile_50, percentile_75,
                sep='\t')
            df_rows.append([tag, np.round(real_age, 2), np.round(mean_prediction, 2), std, percentile_25, percentile_50, percentile_75])
            errors.append(float("{:.2f}".format(abs(mean_prediction - (tag_to_age_dict[tag])))))
        print("Test loss: " + str(loss_test.item()), file=sys.stderr)
        median_error_epoch = np.median(errors)
        print("Median ", median_error_epoch, file=sys.stderr)
        rmse = np.sqrt(np.mean(np.power(errors, 2)))
        print("RMSE ", rmse, file=sys.stderr)


def train(args, df_train, df_val, df_test):
    training_data = CustomDataset(df_train)
    if len(training_data) == 0:
        print("Traning data set is empty", file=sys.stderr)
        exit()
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_data = CustomDataset(df_val)
    if len(val_data) == 0:
        print("Validation data set is empty", file=sys.stderr)
        exit()
    val_dataloader = DataLoader(val_data, batch_size=len(val_data), shuffle=True, num_workers=4)
    predictor = Predictor(args.input_size, args.layer_size, args.num_layers, dropout=args.dropout)
    predictor.to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=1)
    es = EarlyStopping(5, args.marker + '_' + str(args.input_size))
    print("Expected batches: ", len(training_data) / args.batch_size)
    for i in range(int(args.epochs)):
        predictor.train()
        batch_counter = 0
        for _, data in enumerate(train_dataloader):
            batch = data[0]
            labels = data[1]
            batch_counter += 1
            optimizer.zero_grad()
            out = predictor(batch)
            loss = criterion(out, labels[:, None])
            loss.backward()
            optimizer.step()

        # run validation accuracy estimations
        with torch.no_grad():
            print("train loss: ", loss.item())
            predictor.eval()
            batch_validation, labels_validation, tags = next(iter(val_dataloader))
            out_validation = predictor(batch_validation)
            loss_test = criterion(out_validation, labels_validation[:, None])
            print("validation loss: ", loss_test.item())
            predictions_by_tag = dict()
            age_by_tag = dict()
            for l in range(len(out_validation)):
                tag = tags[l]
                age_by_tag[tag] = labels_validation[l].item()
                if tag in predictions_by_tag:
                    predictions_by_tag[tag].append(out_validation[l].item())
                else:
                    predictions_by_tag[tag] = [out_validation[l].item()]
            errors = []
            for tag in predictions_by_tag:
                predictions = predictions_by_tag[tag]
                mean_prediction = np.mean(predictions)
                errors.append(abs(mean_prediction - age_by_tag[tag]))
            scheduler.step(loss_test.item())
            median_error_epoch = np.median(errors)
            errors_per_epochs.append(median_error_epoch)
            print("Median Abs. Error", median_error_epoch)
            print("---------------------------------------")

            # when "early stop" is triggered the training stops and the model is saved in the "models" directory
            if es(loss_test.item(), loss.item(), predictor=predictor):
                print("Training stopped")
                test(args, df_test)
                return True
    return False



if __name__ == "__main__":
    print('start train script')
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type=int)
    parser.add_argument('-lr', '--learning_rate', type=float)
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-d', '--dropout', type=float)
    parser.add_argument('-ls', '--layer_size', type=int)
    parser.add_argument('-nl', '--num_layers', type=int)
    parser.add_argument('-is', '--input_size', type=int)
    parser.add_argument('-dp', '--data_path_train', type=str)
    parser.add_argument('-dpt', '--data_path_test', type=str)
    parser.add_argument('-dpv', '--data_path_val', type=str)
    parser.add_argument('-m', '--marker', type=str)
    parser.add_argument('-vmet', '--validation_median_error_threshold', default=2)
    args = parser.parse_args()
    if not args.learning_rate:
        args.learning_rate = 0.00003
    if not args.input_size:
        args.input_size = 25
    if not args.batch_size:
        args.batch_size = 128
    if not args.dropout:
        args.dropout = 0.01
    if not args.layer_size:
        args.layer_size = 512
    if not args.num_layers:
        args.num_layers = 6
    if not args.epochs:
        args.epochs = 1000
    device = torch.device('cpu')

    print("Running DL for marker ", args.marker)
    try:
        num_of_sites = len(locus_parameters[args.marker])
    except:
        print("Wrong marker name", file=sys.stderr)
        exit(-1)
    if num_of_sites > 4:
        args.layer_size = 512
    else:
        args.layer_size = 256
    args.input_size = 2**num_of_sites + 2*num_of_sites + 1    # total size of the input
    df_train = augment(args.data_path_train)
    df_val = augment(args.data_path_val)
    df_test = augment(args.data_path_test)
    train(args, df_train, df_val, df_test)