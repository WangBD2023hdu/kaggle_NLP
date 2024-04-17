import argparse
from tqdm import tqdm
import os
from utils.logging.tf_logger import Logger
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch
from torch.nn import CrossEntropyLoss
import numpy as np

from utils.data_utils import construct_edge_image
from utils.dataset import BaseSet
from utils.compute_scores import get_metrics, get_four_metrics
from utils.data_utils import PadCollate_without_know
import json
import re
from utils.data_utils import seed_everything
from model.Classifer import GCNClassifer

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--mode', type=str, default='train',
                    help="mode, {'" + "train" + "',     '" + "eval" + "'}")
parser.add_argument('-p', '--path', type=str, default='saved_model path',
                    help="path, relative path to save model}")
parser.add_argument('-s', '--save', type=str, default='saved model',
                    help="path, path to saved model}")
parser.add_argument('-o', '--para', type=str, default='parameter.json',
                    help="path, path to json file keeping parameter}")
args = parser.parse_args()
with open(args.para) as f:
    parameter = json.load(f)

annotation_files = parameter['annotation_files']

model = GCNClassifer(txt_input_dim=parameter["txt_input_dim"], txt_out_size=parameter["txt_out_size"],
                                txt_gat_layer=parameter["txt_gat_layer"], txt_gat_drop=parameter["txt_gat_drop"],
                                txt_gat_head=parameter["txt_gat_head"],
                                txt_self_loops=parameter["txt_self_loops"])

model.to(device=device)
# 0.05
optimizer = optim.Adam(params=model.parameters(), lr=parameter["lr"], betas=(0.9, 0.999), eps=1e-8,
                       weight_decay=parameter["weight_decay"],
                       amsgrad=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=parameter["patience"], verbose=True)
# optimizer = optim.Adam(params=model.parameters(), lr=parameter["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True)

cross_entropy_loss = CrossEntropyLoss()
# cross_entropy_loss = CrossEntropyLoss(weight=torch.tensor([1,1.1]).cuda())
# args.path must be relative path
logger = Logger(model_name=parameter["model_name"], data_name='twitter',
                log_path=os.path.join(parameter["TARGET_DIR"], args.path,
                                      'tf_logs', parameter["model_name"]))

def train_model(epoch, train_loader):
    """
        Performs one training epoch and updates the weight of the current model

        Args:
            train_loader:
            optimizer:
            epoch(int): Current epoch number

        Returns:
            None
    """
    train_loss = 0.0
    total = 0.0
    model.train()
    predict = []
    real_label = []

    for batch_idx, (encoded_cap, word_spans, mask_batch1, edge_cap1, labels) in enumerate(tqdm(train_loader)):
        embed_batch1 = {k: v.to(device) for k, v in embed_batch1.items()}
        with torch.set_grad_enabled(True):
            y = model(texts=encoded_cap, mask_batch=mask_batch1.cuda(),
                        t1_word_seq=word_spans.cuda(), txt_edge_index=edge_cap1.cuda())

            loss = cross_entropy_loss(y, labels.cuda())
            loss.backward()
            train_loss += float(loss.detach().item())
            optimizer.step()
            optimizer.zero_grad()  # clear gradients for this training step
        predict = predict + get_metrics(y.cpu())
        real_label = real_label + labels.cpu().numpy().tolist()
        total += len(encoded_cap)
        torch.cuda.empty_cache()
        del embed_batch1
    # Calculate loss and accuracy for current epoch
    logger.log(mode="train", scalar_value=train_loss / len(train_loader), epoch=epoch, scalar_name='loss')
    acc, recall, precision, f1 = get_four_metrics(real_label, predict)
    logger.log(mode="train", scalar_value=acc, epoch=epoch, scalar_name='accuracy')

    print(' Train Epoch: {} Loss: {:.4f} Acc: {:.4f} Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(epoch, train_loss / len(
        train_loader), acc, recall,
                                                                                                precision, f1))


def eval_validation_loss(val_loader):
    """
        Computes validation loss on the saved model, useful to resume training for an already saved model
    """
    val_loss = 0.
    predict = []
    real_label = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (encoded_cap, word_spans, mask_batch1, edge_cap1, labels) in enumerate(tqdm(val_loader)):
            embed_batch1 = {k: v.to(device) for k, v in embed_batch1.items()}
            y = model(texts=encoded_cap, mask_batch=mask_batch1.cuda(),
                    t1_word_seq=word_spans.cuda(), txt_edge_index=edge_cap1.cuda())

            loss = cross_entropy_loss(y, labels.cuda())
            val_loss += float(loss.clone().detach().item())
            predict = predict + get_metrics(y.cpu())
            real_label = real_label + labels.cpu().numpy().tolist()
            torch.cuda.empty_cache()
            del embed_batch1

        acc, recall, precision, f1 = get_four_metrics(real_label, predict)
        print(' Val Avg loss: {:.4f} Acc: {:.4f} Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(val_loss / len(val_loader),
                                                                                            acc, recall,
                                                                                            precision, f1))
    return val_loss


def evaluate_model(epoch, val_loader):
    """
        Performs one validation epoch and computes loss and accuracy on the validation set

        Args:
            model:
            epoch (int): Current epoch number

        Returns:
            val_loss (float): Average loss on the validation set
    """
    val_loss = 0.
    predict = []
    real_label = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (encoded_cap, word_spans, mask_batch1, edge_cap1, labels) in enumerate(tqdm(val_loader)):
            embed_batch1 = {k: v.to(device) for k, v in embed_batch1.items()}
            y = model(texts=encoded_cap, mask_batch=mask_batch1.cuda(),
                    t1_word_seq=word_spans.cuda(), txt_edge_index=edge_cap1.cuda())

            loss = cross_entropy_loss(y, labels.cuda())
            val_loss += float(loss.clone().detach().item())
            predict = predict + get_metrics(y.cpu())
            real_label = real_label + labels.cpu().numpy().tolist()
            torch.cuda.empty_cache()
            del embed_batch1

        acc, recall, precision, f1 = get_four_metrics(real_label, predict)

        logger.log(mode="val", scalar_value=val_loss / len(val_loader), epoch=epoch, scalar_name='loss')
        logger.log(mode="val", scalar_value=acc, epoch=epoch, scalar_name='accuracy')
        print(' Val Epoch: {} Avg loss: {:.4f} Acc: {:.4f}  Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(epoch,
                                                                                                       val_loss / len(
                                                                                                           val_loader),
                                                                                                       acc, recall,
                                                                                                       precision, f1))
    return val_loss


def evaluate_model_test(epoch, test_loader):
    """
        Performs one validation epoch and computes loss and accuracy on the validation set

        Args:
            epoch (int): Current epoch number
            test_loader:

        Returns:
            val_loss (float): Average loss on the validation set
    """
    test_loss = 0.
    predict = []
    real_label = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (encoded_cap, word_spans, mask_batch1, edge_cap1, labels) in enumerate(tqdm(test_loader)):
            embed_batch1 = {k: v.to(device) for k, v in embed_batch1.items()}
            y = model(texts=encoded_cap, mask_batch=mask_batch1.cuda(),
                    t1_word_seq=word_spans.cuda(), txt_edge_index=edge_cap1.cuda())

            loss = cross_entropy_loss(y, labels.cuda())
            test_loss += float(loss.clone().detach().item())
        predict = predict + get_metrics(y.cpu())
        real_label = real_label + labels.cpu().numpy().tolist()
        torch.cuda.empty_cache()
        del embed_batch1

    acc, recall, precision, f1 = get_four_metrics(real_label, predict)

    logger.log(mode="test", scalar_value=test_loss / len(test_loader), epoch=epoch, scalar_name='loss')
    logger.log(mode="test", scalar_value=acc, epoch=epoch, scalar_name='accuracy')
    print(' Test Epoch: {} Avg loss: {:.4f} Acc: {:.4f}  Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(epoch,
                                                                                                    test_loss / len(
                                                                                                        test_loader),
                                                                                                    acc, recall,
                                                                                                    precision, f1))
    return test_loss


def test_match_accuracy(val_loader):
    """
    Args:
        Once the model is trained, it is used to evaluate the how accurately the captions align with the objects in the image
    """
    try:
        print("Loading Saved Model")
        checkpoint = torch.load(args.save)
        model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
        val_loss = 0.
        predict = []
        real_label = []
        pv_list = []
        pv_know_list = []
        a_list = []
        a_know_list = []
        model.eval()

        with torch.no_grad():
            for batch_idx, (encoded_cap, word_spans, mask_batch1, edge_cap1, labels) in enumerate(tqdm(val_loader)):
                embed_batch1 = {k: v.to(device) for k, v in embed_batch1.items()}
                with torch.no_grad():
                    y = model(texts=encoded_cap, mask_batch=mask_batch1.cuda(),
                            t1_word_seq=word_spans.cuda(), txt_edge_index=edge_cap1.cuda())
                    loss = cross_entropy_loss(y, labels.cuda())
                    val_loss += float(loss.clone().detach().item())
                predict = predict + get_metrics(y.cpu())
                real_label = real_label + labels.cpu().numpy().tolist()
                pv_list.append(pv.cpu().clone().detach())
                a_list.append(a.cpu().clone().detach())
                torch.cuda.empty_cache()
                del img_batch, embed_batch1
            acc, recall, precision, f1 = get_four_metrics(real_label, predict)
            save_result = {"real_label": real_label, 'predict_label': predict, "pv_list": pv_list,
                            " a_list": a_list}
            torch.save(save_result, "with_out_knowledge")
        print(
            "Avg loss: {:.4f} Acc: {:.4f}  Rec: {:.4f} Pre: {:.4f} F1: {:.4f}".format(val_loss, acc, recall, precision,
                                                                                      f1))
    except Exception as e:
        print(e)
        exit()


def main():
    if args.mode == 'train':
        print("flag")
        # annotation_train = os.path.join(annotation_files, "trainknow.json")
        # annotation_val = os.path.join(annotation_files, "valknow.json")
        # annotation_test = os.path.join(annotation_files, "testknow.json")
        
        annotation_train = os.path.join(annotation_files, "traindep.json")
        annotation_val = os.path.join(annotation_files, "valdep.json")
        annotation_test = os.path.join(annotation_files, "testdep.json")

        train_dataset = BaseSet(type="train", max_length=parameter["max_length"], text_path=annotation_train)
        val_dataset = BaseSet(type="val", max_length=parameter["max_length"], text_path=annotation_val)
        test_dataset = BaseSet(type="test", max_length=parameter["max_length"], text_path=annotation_test)

        train_loader = DataLoader(dataset=train_dataset, batch_size=parameter["batch_size"], num_workers=4,
                                    shuffle=True,
                                    collate_fn=PadCollate_without_know())
        print("training dataset has been loaded successful!")
        val_loader = DataLoader(dataset=val_dataset, batch_size=parameter["batch_size"], num_workers=4,
                                shuffle=True,
                                collate_fn=PadCollate_without_know())
        print("validation dataset has been loaded successful!")
        test_loader = DataLoader(dataset=test_dataset, batch_size=parameter["batch_size"], num_workers=4,
                                    shuffle=True,
                                    collate_fn=PadCollate_without_know())
        print("test dataset has been loaded successful!")

        start_epoch = 0
        patience = 8

        if args.path is not None and not os.path.exists(args.path):
            os.mkdir(args.path)
        try:
            print("Loading Saved Model")
            checkpoint = torch.load(args.save)
            model.load_state_dict(checkpoint)
            start_epoch = int(re.search("-\d+", args.save).group()[1:]) + 1
            print("Saved Model successfully loaded")
            # Only effect special layers like dropout layer
            model.eval()
            best_loss = eval_validation_loss(val_loader=val_loader)
        except:
            print("Failed, No Saved Model")
            best_loss = np.Inf
        early_stop = False
        counter = 0
        for epoch in range(start_epoch + 1, parameter["epochs"] + 1):
            # Training epoch
            train_model(epoch=epoch, train_loader=train_loader)
            # Validation epoch
            avg_val_loss = evaluate_model(epoch, val_loader=val_loader)
            avg_test_loss = evaluate_model_test(epoch, test_loader=test_loader)

            scheduler.step(avg_val_loss)
            if avg_val_loss <= best_loss:
                counter = 0
                best_loss = avg_val_loss
                # torch.save(model.state_dict(), os.path.join(args.path, parameter["model_name"] + '-' + str(epoch) + '.pt'))
                print("Best model saved/updated..")
                torch.cuda.empty_cache()
            else:
                counter += 1
                if counter >= patience:
                    early_stop = True
            # If early stopping flag is true, then stop the training
            torch.save(model.state_dict(), os.path.join(args.path, parameter["model_name"] + '-' + str(epoch) + '.pt'))
            if early_stop:
                print("Early stopping")
                break

    elif args.mode == 'eval':
        # args.save
        annotation_test = os.path.join(annotation_files, "testdep.json")
        test_dataset = BaseSet(type="test", max_length=parameter["max_length"], text_path=annotation_test)
        test_loader = DataLoader(dataset=test_dataset, batch_size=parameter["batch_size"], shuffle=False,
                                 collate_fn=PadCollate_without_know())

        print("validation dataset has been loaded successful!")
        test_match_accuracy(val_loader=test_loader)

    else:
        print("Mode of SSGN is error!")


if __name__ == "__main__":
    print("12")
    main()
    # seed_everything(42)
