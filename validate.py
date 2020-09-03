import torchvision.transforms as transforms
from torch import nn
from utils import *
from build_vocab import Vocabulary

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice

from pycocotools.coco import COCO
from PIL import Image
import nltk
import cv2
import pdb


def validate(args, vocab, model, is_visualize=False):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    score_avg = 0.0
    bleu_scorer = Bleu(n=4)
    rouge_scorer = Rouge()
    cider_scorer = Cider()
    meteor_scorer = Meteor()

    val_dir = args.val_dir.replace('dataset', args.dataset)
    val_coco = COCO(args.val_caption_path.replace('dataset', args.dataset))
    val_ids = list(val_coco.anns.keys())
    transform = transforms.Compose([ 
        transforms.Resize(args.image_size),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    captions = {} 
    hypothese = {}   
    imgs_num = int(len(val_ids)/5)    
    for i in range(imgs_num):
        img_id = val_coco.anns[val_ids[i*5]]['image_id']
        img_path = val_coco.loadImgs(img_id)[0]['file_name']
        # print ('image_path: ', img_path)
        image = Image.open(os.path.join(val_dir, img_path)).convert('RGB')
        image = transform(image)
        image = image.cuda()
        image = image.view(1, image.size(0), image.size(1), image.size(2))
        
        output = model.sample(image)
        hypothese_i = batch_ids2words(output, vocab)
        hypothese[str(img_id)] = hypothese_i

        captions_i = []
        for ann_id in range(img_id*5, (img_id+1)*5):
            caption = val_coco.anns[ann_id]['caption']   
            captions_i.append(caption)
        captions[str(img_id)] = captions_i

    (bleu1, bleu2, bleu3, bleu4), _ = bleu_scorer.compute_score(captions, hypothese)
    cider, _ = cider_scorer.compute_score(captions, hypothese)
    rouge, _ = rouge_scorer.compute_score(captions, hypothese)
    meteor, _ = meteor_scorer.compute_score(captions, hypothese)
    score_avg = (bleu1 + bleu2 + bleu3 + bleu4 + cider/3.0 + rouge + meteor) / 7

    print ('bleu1: {:.4f}'.format(bleu1))
    print ('bleu2: {:.4f}'.format(bleu2))
    print ('bleu3: {:.4f}'.format(bleu3))
    print ('bleu4: {:.4f}'.format(bleu4))
    print ('meteor: {:.4f}'.format(meteor))
    print ('rouge: {:.4f}'.format(rouge))
    print ('cider: {:.4f}'.format(cider))
    print ('score: {:.4f}'.format(score_avg))

    return score_avg