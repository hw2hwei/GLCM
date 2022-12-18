import argparse
def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', type=str, default='rsicd',
                            choices=['sydney', 'ucm', 'rsicd'])    
    parser.add_argument('-img_arch', type=str, default='googlenet',
                            choices=['bninception', 'inceptionresnetv2', 'inceptionv3', 
                                    'inceptionv4', 'alexnet', 'resnet18', 'resnet34', 
                                    'resnet50', 'resnet101', 'resnet152', 'vgg16', 
                                    'googlenet'])
    parser.add_argument('-sen_arch', type=str, default='global_local')
    parser.add_argument('--model_path', type=str, 
                            default='./checkpoints/dataset_img_arch_sen_arch.pkl',
                            help='path for trained encoder')
    parser.add_argument('--vocab_path', type=str, default='./data/dataset/vocab.pkl',
                            help='path for vocabulary wrapper')
    parser.add_argument('--train_dir', type=str, default='./data/dataset/train',  
                            help='directory for resized images')
    parser.add_argument('--val_dir', type=str, default='./data/dataset/val',  
                            help='directory for resized images')
    parser.add_argument('--train_caption_path', type=str,
                            default='./data/dataset/annotations/train_dataset.json',
                            help='path for train annotation json file')
    parser.add_argument('--val_caption_path', type=str,
                            default='./data/dataset/annotations/val_dataset.json',
                            help='path for train annotation json file')
    parser.add_argument('--is_visualize', type=bool, default=False,
                            help='whether to visualize attention map')

    parser.add_argument('--max_seq_length', type=int , default=25,
                            help='max length of sequence')
    parser.add_argument('--embed_dim', type=int , default=512,
                            help='embedding dim of features')

    # learning setting
    parser.add_argument('--start_epoch', type=int, default=0,
                            help='start epoch for training')
    parser.add_argument('--end_epoch', type=int, default=50,
                            help='end epoch for training')
    # rsicd: 3e-4, ucm: 1e-4,
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--step_size', type=float, default=50)
    parser.add_argument('--image_size', type=int, default=224)
 
    args = parser.parse_args()
    return args
 
