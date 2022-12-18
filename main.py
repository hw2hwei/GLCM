import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import pickle
from torch import nn
from model import build_model
from utils import *
from data_loader import build_datasets
from validate import validate
from train import train
import pdb
import args_parser
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from build_vocab import Vocabulary



def main():
    args = args_parser.args_parser()
    print (args)
    best_score = 0
    # build vocabulary
    vocab = pickle.load(open(args.vocab_path.replace('dataset', args.dataset), 'rb'))

    # build model
    model = build_model(img_arch=args.img_arch,
                        sen_arch=args.sen_arch,
                        embed_dim=args.embed_dim,
                        vocab=vocab,
                        vocab_size=len(vocab),
                        max_seq_length=args.max_seq_length).cuda()
    # optimizer = torch.optim.SGD(model.get_parameters(), lr=args.lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.get_parameters(), lr=args.lr)

    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('img_arch', args.img_arch) \
                                .replace('sen_arch', args.sen_arch)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))
    # pdb.set_trace()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab('<pad>')).cuda()

    # Custom dataloader
    train_loader = build_datasets(args, vocab)

    print ('Validation Bofore Training: ')
    best_score = validate(args=args,
                          vocab=vocab, 
                          model=model,
                          is_visualize=args.is_visualize)
    print ('')

    # Epochs
    print ('Start Training: ')
    for epoch in range(args.start_epoch, args.end_epoch):
        # One epoch's traininginceptionv3
        print ('Train_Epoch_{}: '.format(epoch))
        train_loss = train(args=args,
                           vocab=vocab,
                           train_loader=train_loader,
                           model=model,
                           criterion=criterion, 
                           optimizer=optimizer,
                           epoch=epoch)

        # One epoch's validation
        print ('Val_Epoch_{}: '.format(epoch))
        recent_score = validate(args=args,
                                vocab=vocab, 
                                model=model,
                                is_visualize=args.is_visualize)

        # # save model
        is_best = recent_score > best_score
        best_score = max(recent_score, best_score)
        # is_best = True
        if is_best:
            torch.save(model.state_dict(), model_path)
            print ('Saved!')
            print ('')

        txt_file = args.dataset + "_" + args.img_arch + "_" + args.sen_arch + '.txt'
        with open(txt_file, 'a') as f:
            f.write(str(epoch) + ' ' + str(train_loss) + ' ' + str(recent_score) + '\n')

if __name__ == '__main__':
    main()
