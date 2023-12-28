from model import TIPCB
import argparse
import torch
from transformers import AutoTokenizer
from dataset import split, TIPCB_data, NPZ_data
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pickle
import pickle
import logging
import tensorflow as tf
import torch.optim as optim
from torch import optim
import matplotlib.pyplot as plt  # 添加这行导入语句

          
     
            
if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()

    # Your main code here

parser = argparse.ArgumentParser()#定义了 argparse 模块中的一个参数解析器，可以用来读取和解析命令行参数

parser.add_argument('--checkpoint_dir', type=str,#--checkpoint_dir：表示保存训练过程中的模型参数文件的目录，默认为当前目录下的 checkpoint 文件夹
                    default="./checkpoint",
                    help='directory to store checkpoint')
parser.add_argument('--log_dir', type=str,#--log_dir：表示保存 TensorBoard 日志文件的目录，默认为当前目录下的 log 文件夹
                    default="./log",
                    help='directory to store log')

#word_embedding
parser.add_argument('--max_length', type=int, default=64)#解析命令行参数

#image setting
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--height', type=int, default=384)
parser.add_argument('--feature_size', type=int, default=2048)

#experiment setting
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=100)

#loss function setting
parser.add_argument('--epsilon', type=float, default=1e-8)

# the root of the data folder
parser.add_argument("--image_root_path", type=str, default="imgs")

parser.add_argument('--adam_lr', type=float, default=0.003, help='the learning rate of adam')
parser.add_argument('--wd', type=float, default=0.00004)
parser.add_argument('--lr_decay_type', type=str, default='MultiStepLR',
                        help='One of "MultiStepLR" or "StepLR" or "ReduceLROnPlateau"')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
parser.add_argument('--epoches_decay', type=str, default='40', help='#epoches when learning rate decays')
parser.add_argument('--warm_epoch', default=10, type=int, help='the first K epoch that needs warm up')

parser.add_argument("--language", default="en", type=str, help="the language to train for")

args = parser.parse_args()



# ------------------------ test dataset ------------------------
# with open("/aicity/TIPCB/data/BERT_en_original/BERT_id_train_64_new.npz", "rb") as f:
#     train = pickle.load(f)

# with open("/aicity/TIPCB/data/BERT_en_original/BERT_id_val_64_new.npz", "rb") as f:
#     val = pickle.load(f)

# train_dataset = NPZ_data(train, args)
# val_dataset = NPZ_data(val, args, train=False)
# train_dl = DataLoader(train_dataset, batch_size=2)
# val_dl = DataLoader(val_dataset, batch_size=2)

# for batch in val_dl:
#     print(batch)
#     break

# -------------split---------------------------------------------------




early_stopping = EarlyStopping('val_rank1', mode='max', patience=10)#EarlyStopping的作用是在验证集上监控指定的指标（这里是val_rank1，也就是验证集top-1准确率），当连续patience次（这里是10次）验证集指标没有提高时就停止训练
checkpoint_callback = ModelCheckpoint(
    dirpath=args.checkpoint_dir,
    filename='{epoch}-{val_rank1:.4f}-{val_loss:.4f}',
    monitor='val_rank1',
    mode='max'
)

pl.seed_everything(0)
tb_logger = pl_loggers.TensorBoardLogger(args.log_dir)

if args.language == "en":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", model_max_length=args.max_length)
    train_list, val_list = split('caption_all.json')
    train_dataset = TIPCB_data(train_list, tokenizer, args)
    val_dataset = TIPCB_data(val_list, tokenizer, args, train=False)
elif args.language == "th":
    # Add code for Thai language here
    pass
else:
    raise ValueError("Invalid language")

# with open("/aicity/TIPCB/data/BERT_en_original/BERT_id_train_64_new.npz", "rb") as f:
#     train = pickle.load(f)

# with open("/aicity/TIPCB/data/BERT_en_original/BERT_id_test_64_new.npz", "rb") as f:
#     val = pickle.load(f)

# train_dataset = NPZ_data(train, args)
# val_dataset = NPZ_data(val, args, train=False)
train_dl = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)


model = TIPCB(args, val_len=len(val_dl))
#model = TIPCB(args, train_len = len(train_dl),val_len=len(val_dl))






    
class LossPlotCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # 记录训练和验证损失
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())

        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, trainer.current_epoch + 2), self.train_losses, label='Train Loss')
        plt.plot(range(1, trainer.current_epoch + 2), self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('loss_plot2.png')  # 保存图像为 PNG 文件
        plt.close()
        
        
# 在创建 Trainer 前创建 LossPlotCallback 对象
loss_plot_callback = LossPlotCallback()

trainer = pl.Trainer(amp_level='O1', amp_backend="apex",
                        max_epochs=args.num_epochs,
                        callbacks=[checkpoint_callback, loss_plot_callback],  # 添加 loss_plot_callback 到回调列表中
                        gpus=1,  # 代表gpu是否使用，0没有
                        accumulate_grad_batches=1,
                        logger=tb_logger)

trainer.fit(model, train_dl, val_dl)

    
# class LossPlotCallback(pl.Callback):
#     def __init__(self):
#         super().__init__()
#         self.train_losses = []
#         self.val_losses = []

#     def on_train_epoch_end(self, trainer, pl_module):
#         # 记录训练和验证损失
#         self.train_losses.append(trainer.callback_metrics["train_loss"].item())
#         self.val_losses.append(trainer.callback_metrics["val_loss"].item())

#         # 绘制损失曲线
#         plt.figure(figsize=(10, 5))
#         plt.plot(range(1, trainer.current_epoch + 2), self.train_losses, label='Train Loss')
#         plt.plot(range(1, trainer.current_epoch + 2), self.val_losses, label='Val Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Training and Validation Loss')
#         plt.legend()
#         plt.savefig('loss_plot.png')  # 保存图像为 PNG 文件
#         plt.close()

# # 在模型训练之前创建回调函数对象
# loss_plot_callback = LossPlotCallback()

# # 在 Trainer 中添加回调函数
# trainer = pl.Trainer(callbacks=[loss_plot_callback])

# # 训练你的模型
# trainer.fit(model, train_dl, val_dl)

#logging.basicConfig(filename='training.log', level=logging.INFO)
#保存当前训练的模型参数
#optimizer = optim.AdamW(model.parameters(), lr=0.001)
#checkpoint = {'model_state_dict': model.state_dict(),
 #             'optimizer_state_dict': optimizer.state_dict(),
  #            'epoch': 80}
#torch.save(checkpoint, '/home/zhengfangfang/tipcb thai/result7.6.ckpt')



# ----------------------------------------------------------------
