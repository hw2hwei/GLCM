import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import pickle
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import build_model
from utils import *
from data_loader import build_datasets
from build_vocab import Vocabulary
from collections import OrderedDict 
from pycocotools.coco import COCO

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice

from PIL import Image
import nltk
import pdb
import cv2

import args_parser
args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt
import numpy as np


def similar_matrix(img_path, caption, attn_w2w):
    print ('length: ', len(caption))
    tick_marks = np.array(range(len(caption))) + 0.5

    np.set_printoptions(precision=2)
    plt.figure(figsize=(12, 8), dpi=250)

    ind_array = np.arange(len(caption))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = attn_w2w[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='green', fontsize=10, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    # plt.gcf().subplots_adjust(bottom=0.15)

    plt.imshow(attn_w2w, interpolation='nearest', cmap=plt.get_cmap('YlOrBr'))
    plt.title('Word Relevance')
    plt.colorbar()
    xlocations = np.array(range(len(caption)))
    plt.xticks(xlocations, caption, rotation=90)
    # plt.yticks(xlocations, caption)
    # plt.xlabel('Words')

    # show confusion matrix
    plt.savefig('./attn_samples/{}.png'.format(img_path), format='png')
    # plt.show()


def max_min_scaling(attn):
    max = np.max(attn)
    min = np.min(attn)
    return (attn - min) / (max - min + 1e-8)

def attn_visualization(img_path, caption, attn_img, attn_w2w, attn_w2i):
    caption = caption[0].strip('\'').strip(' .').split(' ')

    img = cv2.imread(img_path)
    height, width = img.shape[0:2]
    length = len(caption)
    attn_img = cv2.resize(attn_img, (width, height))
    attn_w2w = attn_w2w[0]
    attn_w2i = attn_w2i[0]
    print ('image_path: ', img_path)
    print ('caption: ', caption)    
    print ('attn_w2w: ', attn_w2w.shape)    
    print ('attn_w2i: ', attn_w2i.shape)    

    attn_w2w = similar_matrix(img_path.split('/')[-1].strip('.jpg'), caption, attn_w2w[:-2, :-2])
    for i in range(length):
        idx = attn_w2i[i]
        print (i, idx, caption[i])
        attn_img_i = attn_img[:, :, idx][:, :, np.newaxis]
        attn_img_i = max_min_scaling(attn_img_i)
        save_path_i = './attn_samples/' + img_path.split('/')[-1].replace('.', '_{}_{}.'.format(i, caption[i]))
        img_i = (img*attn_img_i*1.0 + img*0.0).astype(np.uint8)
        cv2.imwrite(save_path_i, img_i)
    print ()


def main():
    best_bleu = 0

    with open(args.vocab_path.replace('dataset', args.dataset), 'rb') as f:
        vocab = pickle.load(f)

    # Build the models
    model = build_model(img_arch=args.img_arch,
                        sen_arch=args.sen_arch,
                        embed_dim=args.embed_dim,
                        vocab=vocab,
                        vocab_size=len(vocab),
                        max_seq_length=args.max_seq_length).cuda()

    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('img_arch', args.img_arch) \
                                .replace('sen_arch', args.sen_arch)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=True)
        print ('Load the chekpoint of {}'.format(model_path))


    # Custom dataloader
    print ('Validation Bofore Training: ')
    model.eval()

    bleu_scorer = Bleu(n=4)
    rouge_scorer = Rouge()
    cider_scorer = Cider()
    meteor_scorer = Meteor()

    batch_time = AverageMeter()
    losses = AverageMeter()

    val_dir = args.val_dir.replace('dataset', args.dataset)
    val_coco = COCO(args.val_caption_path.replace('dataset', args.dataset))
    val_ids = list(val_coco.anns.keys())
    transform = transforms.Compose([ 
        transforms.Resize(args.image_size),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    imgs_num = int(len(val_ids)/5) 
    captions = {} 
    hypothese = {}  
    for i in range(imgs_num):
        img_id = val_coco.anns[val_ids[i*5]]['image_id']
        img_path = val_coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(val_dir, img_path)).convert('RGB')
        image = transform(image)
        image = image.cuda()
        image = image.view(1, image.size(0), image.size(1), image.size(2))
        
        # output = model.sample(image)
        output, attn_img, attn_w2w, attn_w2i = model.calc_attn(image)
        hypothese_i = batch_ids2words(output, vocab)
        hypothese[str(img_id)] = hypothese_i

        captions_i = []
        for ann_id in range(img_id*5, (img_id+1)*5):
            caption = val_coco.anns[ann_id]['caption']   
            captions_i.append(caption)
        captions[str(img_id)] = captions_i

        if 'center_85' in img_path or 'industrial_76' in img_path or 'parking_9.jpg' in img_path or 'storagetanks_76' in img_path:
            img_path = os.path.join(val_dir, img_path)
            attn_img = attn_img[0].view(7, 7, -1).cpu().detach().numpy()
            attn_w2w = attn_w2w.cpu().detach().numpy()
            attn_w2i = attn_w2i.cpu().detach().numpy()
            attn_visualization(img_path, hypothese_i, attn_img, attn_w2w, attn_w2i)

    # print ("hypothese: ", hypothese)
    # print ("captions: ", captions)

    (bleu1, bleu2, bleu3, bleu4), _ = bleu_scorer.compute_score(captions, hypothese)
    cider, _ = cider_scorer.compute_score(captions, hypothese)
    rouge, _ = rouge_scorer.compute_score(captions, hypothese)
    meteor, _ = meteor_scorer.compute_score(captions, hypothese)
    score_avg = (bleu1 + bleu2 + bleu3 + bleu4 + cider/3.0 + rouge + meteor) / 7
    
    print ('bleu1: {:.2f}'.format(bleu1*100))
    print ('bleu2: {:.2f}'.format(bleu2*100))
    print ('bleu3: {:.2f}'.format(bleu3*100))
    print ('bleu4: {:.2f}'.format(bleu4*100))
    print ('meteor: {:.2f}'.format(meteor*100))
    print ('rouge: {:.2f}'.format(rouge*100))
    print ('cider: {:.2f}'.format(cider*100))
    print ('score_avg: {:.2f}'.format(score_avg*100))

if __name__ == '__main__':
    main()
