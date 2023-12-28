from model import TIPCB
import argparse
import torch

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
parser.add_argument("--image_root_path", type=str, default="/aicity/data/CUHK-PEDES/imgs")

parser.add_argument('--adam_lr', type=float, default=0.003, help='the learning rate of adam')
parser.add_argument('--wd', type=float, default=0.00004)
parser.add_argument('--lr_decay_type', type=str, default='MultiStepLR',
                        help='One of "MultiStepLR" or "StepLR" or "ReduceLROnPlateau"')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
parser.add_argument('--epoches_decay', type=str, default='40', help='#epoches when learning rate decays')
parser.add_argument('--warm_epoch', default=10, type=int, help='the first K epoch that needs warm up')

parser.add_argument("--language", default="th", type=str, help="the language to train for")

parser.add_argument('--best_checkpoint_path', required=True, type=str, help='path to the best checkpoint after TIPCB training')
parser.add_argument('--output_name', required=True, type=str, help='output file name')

args = parser.parse_args()

# https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
checkpoint = torch.load(args.best_checkpoint_path, map_location=lambda storage, loc: storage)
best_model = TIPCB(args)
best_model.load_state_dict(checkpoint['state_dict'], strict=False)
# best_model.half()
best_model.eval()

# https://github.com/onnx/onnx/issues/654#issuecomment-521233285
# https://pytorch.org/docs/stable/onnx.html#functions
inputs = ['images', 'txt', 'attention_mask']
outputs = ['img_f4', 'txt_f4']
dynamic_axes = {'images': {0: 'batch'}, 'txt': {0: 'batch'}, 'attention_mask': {0: 'batch'}, 'img_f4': {0: 'batch'}, 'txt_f4': {0: 'batch'}}

dummy_input = (torch.rand((1,3,args.height,args.width), dtype=torch.float32),
                                    torch.randint(0, 25000, (1,64), dtype=torch.int64),
                                    torch.randint(0, 2, (1,64), dtype=torch.int64))
                                    
best_model.to_onnx(
    args.output_name,
    dummy_input,
    input_names=inputs,
    output_names=outputs,
    dynamic_axes=dynamic_axes,
    export_params=True,
    opset_version=11)