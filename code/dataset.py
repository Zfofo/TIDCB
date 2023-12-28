import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from sklearn.model_selection import train_test_split
import functools
from PIL import Image
import torchvision.transforms as transforms
from imageio import imread
import numpy as np
import argparse
import pickle

def split(json_path):
    with open(json_path, "rb") as f:
        caption_all = json.load(f)
    # group all the record (each element of `caption_all`) by id
    group_by_id = dict()
    for record in caption_all:
        # check if image file doesn't exist
        if not os.path.exists(os.path.join("imgs", record["file_path"])):
            continue
        # if record["file_path"].split("/")[0] not in ["test_query", "train_query"]:
        #     continue
        if record["id"] not in group_by_id.keys():
            group_by_id[record["id"]] = []
        for caption in record["captions"]:
            group_by_id[record["id"]].append({
                "id": record["id"],
                "file_path": record["file_path"],
                "caption": caption
            })

    train, val = train_test_split([group_by_id[key] for key in group_by_id.keys()], test_size=0.2, random_state=0, shuffle=True)

    return train, val

class TIPCB_data(Dataset):
    def __init__(self, data, tokenizer, args, train=True):
        self.tokenizer = tokenizer
        self.data = functools.reduce(lambda a, b: a + b, data) # flatten the list https://stackoverflow.com/questions/952914
        self.transform_train = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.Pad(10),
            transforms.RandomCrop((args.height, args.width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train = train
    
    def __getitem__(self, index):
       
        item = self.data[index] # dict of {id, file_path, caption}

        # read image and transform
        img = Image.open(os.path.join("imgs", item["file_path"]))
        if self.train:
            img = self.transform_train(img)
        else:
            img = self.transform_val(img)

        # tokenize the caption
        tok = self.tokenizer(item["caption"], truncation=True, padding='max_length')
        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]
        caption = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # prepare the label
        
        label = torch.tensor(int(item["id"]), dtype=torch.long)

        return img, caption, label, attention_mask

    def __len__(self):
        return len(self.data)

class NPZ_data(Dataset):
    def __init__(self, data, args, train=True):
        self.image_root_path = args.image_root_path
        self.max_length = args.max_length
        self.data = data
        # data is the object read from BERT_id_<split>_64_new.npz
        self.transform_train = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.Pad(10),
            transforms.RandomCrop((args.height, args.width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train = train

    def __getitem__(self, index):
        caption = self.data["caption_id"][index]
        attention_mask = self.data["attention_mask"][index]
        image_path = self.data["images_path"][index]
        label = self.data["labels"][index]

        # if self.train:
        #     label -= 1
        # else:
        #     label -= 12004 # original repo do this for validation data

        img = imread(os.path.join(self.image_root_path, image_path))
        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        img = Image.fromarray(img)

        if self.train:
            img = self.transform_train(img)
        else:
            img = self.transform_val(img)

        caption = np.array(caption)
        attention_mask = np.array(attention_mask)
        if len(caption) >= self.max_length:
            caption = caption[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        else:
            pad = np.zeros((self.max_length - len(caption), 1), dtype=np.int64)
            caption = np.append(caption, pad)
            attention_mask = np.append(attention_mask, pad)
        caption = torch.tensor(caption).long()
        attention_mask = torch.tensor(attention_mask).long()
        return img, caption, label, attention_mask

    def __len__(self):
        return len(self.data["labels"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_dir', type=str,
                        default="./checkpoint",
                        help='directory to store checkpoint')
    parser.add_argument('--log_dir', type=str,
                        default="./log",
                        help='directory to store log')

    #word_embedding
    parser.add_argument('--max_length', type=int, default=64)

    #image setting
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--feature_size', type=int, default=2048)

    #experiment setting
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epoches', type=int, default=80)

    #loss function setting
    parser.add_argument('--epsilon', type=float, default=1e-8)

    # the root of the data folder
    parser.add_argument("--image_root_path", type=str, default="imgs")

    args = parser.parse_args()

    with open("/home/zhengfangfang/tipcb thai/TIPCB/BERT_id_train_64_new.npz", "rb") as f:
        train = pickle.load(f)

    train_dataset = NPZ_data(train, args)
    train_dl = DataLoader(train_dataset, batch_size=77, num_workers=4, shuffle=True)

    sample=next(iter(train_dl))
    img, caption, label, mask=sample

    print(label)
    print(label[-1])
    print(caption[-1])
    print(mask[-1])
    print(caption[-1].shape)
    print(mask[-1].shape)
    print(img[-1])
