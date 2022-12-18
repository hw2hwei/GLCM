import torch
from torch import nn
from utils import to_var, batch_ids2words
from build_vocab import Vocabulary
from PIL import Image
from loss import caption_loss

def train(args, 
          vocab, 
          train_loader, 
          model, 
          criterion, 
		  optimizer, 
          epoch):

    model.train()
    loss_total = 0.0
    total_step = len(train_loader)
    for i, (images, captions) in enumerate(train_loader):
        # Set mini-batch dataset
        images = to_var(images).detach()
        captions = to_var(captions).detach()
        batch_size = images.size(0)

        # Forward, Backward and Optimize
        optimizer.zero_grad()
        outputs = model(images, captions[:, :-1])
        loss = caption_loss(outputs, captions[:, 1:], criterion)
        # print (captions[0], torch.max(outputs, dim=2)[1][0])

        loss.backward()
        optimizer.step()
        loss_total += loss.data.detach().cpu().numpy()

        # calculate the metric scores
        # references = batch_ids2words(captions.view(1, -1), vocab)
        # candidates = batch_ids2words(outputs.view(1, -1), vocab)
        # print ('ref: ', references)
        # print ('out: ', candidates)
        # classes = batch_ids2words(output_classes.view(1, -1), vocab)


        # Print log info
        if (i%20==0 and i!=0) or i==len(train_loader)-1:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch, 
                    args.end_epoch, 
                    i, 
                    total_step, 
                    loss.item(),
                    ) 
                 )


    return loss_total / len(train_loader)