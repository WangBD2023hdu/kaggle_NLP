# Essential import
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
seed = 42
torch.manual_seed(seed=seed)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")

def save_metrics(model_name, **metrics):

    '''
        Dump train/val loss and f1-score lists to a json file.
    '''
    with open(f'./metrics/{model_name}.json', 'w') as fh:
        json.dump(metrics, fh)


def load_metrics(model_name):
    '''
    Load train/val loss and f1-score lists from a json file.
    '''
    try:
        with open(f'./metrics/{model_name}.json', 'r') as fh:
            return json.load(fh)
    except:
        raise RuntimeError(f'The metrics json file not found. Was the model `{model_name}` trained?')


def load_model(model):
    '''
    Try to find and load our fine-tuned model.
    '''
    try:
        model.model.load_state_dict(torch.load(f'./models/{model.name}.pt'))
        model.model = model.model.to(device)
        model.is_trained = True
        print(f'Found fine-tuned model `{model.name}.pt`. Ready for inferring/testing.')

    except:

        print(f'Fine-tuned model `{model.name}` not found. Train/eval the model first.')


def plot_metrics(model_name: str, epochs):
    '''
    Plot model's loss and f1-score metrics obtained during train/val over the number of `epochs`.
    '''
    metrics = load_metrics(model_name)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), metrics['train_losses'], marker='s', markersize=5, color='C0', label='Train')
    plt.plot(range(1, epochs+1), metrics['val_losses'], marker='s', markersize=5, color='C1', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(1, epochs+1))
    plt.legend()
    plt.grid(lw=0.25, color='xkcd:cement')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), metrics['train_f1s'], marker='s', markersize=5, color='C0', label='Train')
    plt.plot(range(1, epochs+1), metrics['val_f1s'], marker='s', markersize=5 , color='C1', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.xticks(range(1, epochs+1))
    plt.legend()
    plt.grid(lw=0.25, color='xkcd:cement')
    plt.suptitle(f'Train and validation datasets metrics: {model_name}', fontsize=11)
    plt.tight_layout()


train = pd.read_csv('./text/train.csv', dtype={'target': np.int64})
test = pd.read_csv('./text/test.csv', dtype={'target': np.int64})
target = 'target'
train, val = train_test_split(train, test_size=0.05, stratify=train[target], random_state=seed)


def prepare_data(model, dataset):
    '''
    Tokenize all sentences in `dataset` Pandas DataFrame and prepare data for DataLoader.
    The first column must me sentences, and the second column must be labels.
    Tokenizer is used from `model`.
    Source: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    '''
    sentences = dataset.iloc[:, 0].values
    labels = dataset.iloc[:, 1].values
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = model.tokenizer.encode_plus(
            sent,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]' special tokens
            max_length=model.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    prepared = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)

    return prepared


train['doc'] = train['text'] + ' keyword_' + train['keyword'].astype(str) + ' location_' + train['location'].astype(str).apply(lambda x: ' location_'.join(x.split()))
val['doc'] = val['text'] + ' keyword_' + val['keyword'].astype(str) + ' location_' + val['location'].astype(str).apply(lambda x: ' location_'.join(x.split()))
test['doc'] = test['text'] + ' keyword_' + test['keyword'].astype(str) + ' location_' + test['location'].astype(str).apply(lambda x: ' location_'.join(x.split()))


class TweetClassifier:

    def __init__(self, base_model: str, max_len=64, verbose=False):
        # 1. Create an existing pre-trained model
        if base_model not in ('twitter-roberta-base-sentiment-latest'):
            raise ValueError("base_model should be 'roberta'")
        # https://huggingface.co/FacebookAI/roberta-base
        name = 'roberta'
        model_str = 'twitter-roberta-base-sentiment-latest'
        tokenizer_str = 'twitter-roberta-base-sentiment-latest'

        self.name = name
        self.model = RobertaForSequenceClassification.from_pretrained(model_str, num_labels=2, output_attentions=False,
                                                                      output_hidden_states=False, ignore_mismatched_sizes=True)
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_str)
        self.max_len = max_len
        self.is_trained = False
        self.verbose = verbose
        self.model = self.model.to(device)

    def train(self, train_subset, val_subset, optimizer, epochs: int, batch_size: int):
        # 1. Prepare data
        prepared_train = prepare_data(self, train_subset)
        prepared_val = prepare_data(self, val_subset)
        # 2. Create data loaders
        self.train_loader = torch.utils.data.DataLoader(prepared_train, batch_size=batch_size, shuffle=True,
                                                        num_workers=4)
        self.val_loader = torch.utils.data.DataLoader(prepared_val, batch_size=batch_size, shuffle=True, num_workers=4)
        # 3. Store optimizer
        self.optimizer = optimizer
        # 4. Do train and eval loop for the number of epochs
        self.train_losses, self.val_losses, self.train_f1s, self.val_f1s = [], [], [], []
        for epoch_i, _ in enumerate(range(epochs), start=1):
            print(f'-------------\nEpoch: {epoch_i:>2}/{epochs}')
            # Train one epoch
            self._train()
            # Eval one epoch
            self._validate()

    def _train(self):
        '''
        Train the classifier on all batches of the `train_loader` for one epoch.
        '''
        # 1. Switch to train mode
        self.model.train()
        # 2. Do train loop for all batches for one epoch
        print('\nTraining...')
        losses, f1s = [], []
        n_batches_to_show = np.ceil(len(self.train_loader) / 10)
        for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            # `data` has 3 tensors, as returned by `prepare_data()`
            input_ids = data[0].to(device)
            input_mask = data[1].to(device)
            labels = data[2].to(device)
            self.optimizer.zero_grad()
            output = self.model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
            loss = output.loss
            logits = output.logits
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            f1 = f1_score(labels.cpu().numpy(), logits.argmax(-1).cpu().numpy())
            f1s.append(f1)
            if self.verbose:
                if i % n_batches_to_show == 0:
                    print(f'Batch: {i:>4}, Train loss: {loss:.5f}, Train f1-score: {f1:.5f}')
        # 3. Keep and report train metrics for one epoch
        epoch_mean_loss = np.mean(losses)
        self.train_losses.append(epoch_mean_loss)
        epoch_mean_f1 = np.mean(f1s)
        self.train_f1s.append(np.mean(epoch_mean_f1))
        print(f'Epoch avg: Train loss: {epoch_mean_loss:.5f}, Train f1-score {epoch_mean_f1:.5f}')

    def _validate(self):
        '''
        Validate the classifier on all batches of the `val_loader` for one epoch.
        '''
        # 1. Switch to eval mode
        self.model.eval()
        # 2. Do eval loop
        print('\nEvaluating...')
        losses, f1s = [], []
        n_batches_to_show = np.ceil(len(self.val_loader) / 10)
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                # `data` has 3 tensors, as returned by `prepare_data()`
                input_ids = data[0].to(device)
                input_mask = data[1].to(device)
                labels = data[2].to(device)
                output = self.model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
                loss = output.loss
                logits = output.logits
                losses.append(loss.item())
                f1 = f1_score(labels.cpu().numpy(), logits.argmax(-1).cpu().numpy())
                f1s.append(f1)
                if self.verbose:
                    if i % n_batches_to_show == 0:
                        print(f'Batch: {i:>3}, Val loss: {loss:.5f}, Val f1-score: {f1:.5f}')
        # 3. Keep and report val metrics for one epoch
        epoch_mean_loss = np.mean(losses)
        self.val_losses.append(epoch_mean_loss)
        epoch_mean_f1 = np.mean(f1s)
        self.val_f1s.append(np.mean(epoch_mean_f1))
        print(f'Epoch avg: Val loss: {epoch_mean_loss:.5f}, Val f1-score: {epoch_mean_f1:.5f}')


def train_model(model, train_subset, val_subset, epochs, batch_size):
    '''
    Set up training parameters and invoke model training.
    '''
    if not model.is_trained:
        print(f'Training model `{model.name}`.')
        lr = 1e-6 # 1e-5
        optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)
        model.train(train_subset, val_subset, optimizer, epochs, batch_size)
        model.is_trained = True
        # Save model's state and train/val metrics to disk
        torch.save(model.model.state_dict(), f'./models/{model.name}.pt')
        save_metrics(model.name, train_losses=model.train_losses, val_losses=model.val_losses, train_f1s=model.train_f1s, val_f1s=model.val_f1s)
        print('Model successfully trained and saved to disk.')
    else:
        print('Model is already trained.')

try:
    Path('./models').mkdir()
    print('`./models` directory successfully created.')
except Exception as e:
    print(str(e))
try:
    Path('./metrics').mkdir()
    print('`./metrics` directory successfully created.')
except Exception as e:
    print(str(e))

model_roberta_fine = TweetClassifier(base_model='twitter-roberta-base-sentiment-latest')
load_model(model_roberta_fine)

train_model(model_roberta_fine, train[['doc', target]], val[['doc', target]], epochs=6, batch_size=64)

plot_metrics(model_roberta_fine.name, epochs=6)

labels_pred = []
for doc in tqdm(test['doc']):
    labels_pred.append(model_roberta_fine.model(**model_roberta_fine.tokenizer(doc, return_tensors='pt').to(device)).logits.argmax(-1).item())
df_pred = pd.DataFrame({'id': test['id'], 'target': labels_pred})

df_pred.to_csv('./submission.csv', index=False)
















