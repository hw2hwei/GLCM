import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import pickle
from torch import nn
from model import build_model
from utils import *
from data_loader import build_datasets
from build_vocab import Vocabulary
from validate import validate
from train import train
import pdb
import args_parser
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print (args)


def vocab_distr(vocab, model):
    words = {}
    centroids = torch.zeros(len(vocab)).long().cuda()
    for id in range(len(vocab)):
        # print ('id: ', id, vocab.idx2word[id])
        words[id] = vocab.idx2word[id]
        centroids[id] = id

    centroids = model.decoder.embedding(centroids)

    # for i in range(centroids.size(0)):
    #     similar_i = similar_matrix[i]
    #     _, indices = torch.sort(similar_i, descending=True)
    #     indices = indices[:8].cpu().detach().numpy()
    #     print ('{},'.format(i), words[i], ': ', words[indices[0]], words[indices[1]], 
    #                                             words[indices[2]], words[indices[3]], 
    #                                             words[indices[4]])
    #     print ()

    centroids = centroids.cpu().detach().numpy()
    print ('centroids: ', centroids.shape)
    # print ('words: ', words)
    centroids = TSNE(n_components=2, learning_rate=500).fit_transform(centroids)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax=200,xmin=-200)
    plt.ylim(ymax=200,ymin=-200)
    #画两条（0-9）的坐标轴并设置轴标签x，y
     
    colors = '#00CED1' #点的颜色
    area = np.pi * 1.1**2  # 点面积 
    # 画散点图
    plt.scatter(centroids[:,0], centroids[:,1], linewidths=0.01, marker='d', s=area, c=colors)
    for i in range(0, len(centroids)):
        plt.text(centroids[i,0], centroids[i,1], words[i], fontsize=1)
    plt.savefig(r'embedding_dstribution.png', dpi=800)



def main():
    best_score = 0

    with open(args.vocab_path.replace('dataset', args.dataset), 'rb') as f:
        vocab = pickle.load(f)

    # Build the models
    model = build_model(img_arch=args.img_arch,
                        sen_arch=args.sen_arch,
                        embed_dim=args.embed_dim,
                        vocab=vocab,
                        vocab_size=len(vocab),
                        max_seq_length=args.max_seq_length).cuda()
    # optimizer = torch.optim.SGD(model.get_parameters(), lr=args.lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.get_parameters(), lr=args.lr)
    sche = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)

    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('img_arch', args.img_arch) \
                                .replace('sen_arch', args.sen_arch)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))
    # vocab_distr(vocab, model)
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
        train(args=args,
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
        sche.step()

if __name__ == '__main__':
    main()
