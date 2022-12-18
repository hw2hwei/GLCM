import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cnn_models import googlenet, alexnet, vgg16, resnet18
from self_attn import Self_Attn


class Encoder(nn.Module):
    def __init__(self, 
                 img_arch,
                 embed_dim,
                 vocab_size,
                 max_seq_length):
        """Load the pretrained ResNet and replace top fc layer."""
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        if img_arch == 'alexnet':
            self.cnn = alexnet(pretrained=True, embed_dim=self.embed_dim)
        elif img_arch == 'googlenet':
            self.cnn = googlenet(pretrained=True, embed_dim=self.embed_dim)
        elif img_arch == 'resnet18':
            self.cnn = resnet18(pretrained=True, embed_dim=self.embed_dim)
        elif img_arch == 'vgg16':
            self.cnn = vgg16(pretrained=True, embed_dim=self.embed_dim)

    def forward(self, images):
        """Extract feature vectors from input images."""
        feats_global, feats_local, attns = self.cnn(images)

        return feats_global, feats_local, attns


class Decoder(nn.Module):
    def __init__(self, 
                 vocab,
                 sen_arch,
                 embed_dim, 
                 vocab_size,
                 num_layers,
                 n_head,
                 max_seq_length=25):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab_size

        self.sen_arch = sen_arch
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.fusion1 = Self_Attn(embed_dim, max_seq_length)
        self.fusion2 = Self_Attn(embed_dim, max_seq_length)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.de_embedding = nn.Linear(embed_dim, vocab_size)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)
        self.norm_c = nn.LayerNorm(embed_dim)

        self.linear = nn.Sequential(
                nn.Linear(embed_dim, embed_dim*2),
                nn.ReLU(), 
                nn.Linear(embed_dim*2, embed_dim)
            )

    def generate_mask(self, size):
        mask = (torch.triu(torch.ones(size, size))==1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.float().masked_fill(mask==0, float(1)).masked_fill(mask == 1, float(0.0))
        mask = (mask==1)
        # mask = (mask==0).t()

        # print (mask)
        return mask.cuda().detach()

    def forward(self, feats_global, feats_local, capts):
        """Decode image feature vectors and generates captions."""
        feats_global = feats_global.unsqueeze(dim=1)
        trg_mask = self.generate_mask(self.max_seq_length-1)
        # print (trg_mask)
        capts_embed = self.embedding(capts)

        if self.sen_arch == 'global_A':
            capts_embed = capts_embed + self.dropout1(feats_global.expand_as(capts_embed))
            capts_embed = capts_embed + self.fusion1(capts_embed, capts_embed, capts_embed, trg_mask)
            capts_embed = self.norm_a(capts_embed)

        if self.sen_arch == 'global_B':
            capts_embed = capts_embed + self.dropout1(feats_global.expand_as(capts_embed))

            capts_embed = capts_embed + self.fusion1(capts_embed, capts_embed, capts_embed, trg_mask)
            capts_embed = self.norm_a(capts_embed)
            capts_embed = capts_embed + self.fusion2(capts_embed, capts_embed, capts_embed, trg_mask)
            capts_embed = self.norm_b(capts_embed)

        elif self.sen_arch == 'local':
            capts_embed = capts_embed + self.fusion1(capts_embed, capts_embed, capts_embed, trg_mask)
            capts_embed = self.norm_a(capts_embed)
            capts_embed = capts_embed + self.fusion2(capts_embed, feats_local, feats_local)
            capts_embed = self.norm_b(capts_embed)

        elif self.sen_arch == 'global_local':
            capts_embed = capts_embed + self.dropout1(feats_global.expand_as(capts_embed))
            capts_embed = capts_embed + self.fusion1(capts_embed, capts_embed, capts_embed, trg_mask)
            capts_embed = self.norm_a(capts_embed)
            capts_embed = capts_embed + self.fusion2(capts_embed, feats_local, feats_local)
            capts_embed = self.norm_b(capts_embed)
            

        capts_embed = capts_embed + self.dropout2(self.linear(capts_embed))
        capts_embed = self.norm_c(capts_embed)

        sentences = self.de_embedding(capts_embed)

        return sentences
    
    def sample(self, feat_global, feat_local):
        """Generate captions for given image features using greedy search."""
        feat_global = feat_global.unsqueeze(dim=1)
        seq = torch.ones(1,1).long().cuda()

        for i in range(self.max_seq_length-1):
            seq_embed = self.embedding(seq)

            if self.sen_arch == 'global_A':
                seq_embed = seq_embed + feat_global.expand_as(seq_embed)
                seq_embed = seq_embed + self.fusion1.sample(seq_embed, seq_embed, seq_embed)
                seq_embed = self.norm_a(seq_embed)

            if self.sen_arch == 'global_B':
                seq_embed = seq_embed + feat_global.expand_as(seq_embed)
                seq_embed = seq_embed + self.fusion1.sample(seq_embed, seq_embed, seq_embed)
                seq_embed = self.norm_a(seq_embed)
                seq_embed = seq_embed + self.fusion2.sample(seq_embed, seq_embed, seq_embed)
                seq_embed = self.norm_b(seq_embed)

            elif self.sen_arch == 'local':
                seq_embed = seq_embed + self.fusion1.sample(seq_embed, seq_embed, seq_embed)
                seq_embed = self.norm_a(seq_embed)
                seq_embed = seq_embed + self.fusion2.sample(seq_embed, feat_local, feat_local)
                seq_embed = self.norm_b(seq_embed)

            elif self.sen_arch == 'global_local':
                seq_embed = seq_embed + feat_global.expand_as(seq_embed)
                seq_embed = seq_embed + self.fusion1.sample(seq_embed, seq_embed, seq_embed)
                seq_embed = self.norm_a(seq_embed)
                seq_embed = seq_embed + self.fusion2.sample(seq_embed, feat_local, feat_local)
                seq_embed = self.norm_b(seq_embed)

            seq_embed = seq_embed + self.linear(seq_embed)
            seq_embed = self.norm_c(seq_embed)

            prob = self.de_embedding(seq_embed[:, -1, :])
            _, next_word = torch.max(prob, dim=-1)
            next_word = next_word.unsqueeze(dim=0)
            seq = torch.cat([seq, next_word], dim=1)

            if self.vocab.idx2word[int(next_word[0,0].cpu().numpy())] == '<end>':
                break

        return seq

    def calc_attn(self, feat_global, feat_local):
        """Generate captions for given image features using greedy search."""
        feat_global = feat_global.unsqueeze(dim=1)
        seq = torch.ones(1,1).long().cuda()

        for i in range(self.max_seq_length-1):
            trg_mask = self.generate_mask(i+1)
            seq_embed = self.embedding(seq)

            # seq_embed = seq_embed + feat_global.expand_as(seq_embed)

            seq_embed1, attn_w2w = self.fusion1.calc_attn(seq_embed, seq_embed, seq_embed, trg_mask)
            seq_embed = seq_embed + seq_embed1
            seq_embed = self.norm_a(seq_embed)

            seq_embed2, attn_w2i = self.fusion2.calc_attn(seq_embed, feat_local, feat_local)
            seq_embed = seq_embed + seq_embed2
            seq_embed = self.norm_b(seq_embed)

            seq_embed = seq_embed + self.dropout2(self.linear(seq_embed))
            seq_embed = self.norm_c(seq_embed)

            prob = self.de_embedding(seq_embed[:, -1, :])
            _, next_word = torch.max(prob, dim=-1)
            next_word = next_word.unsqueeze(dim=0)
            seq = torch.cat([seq, next_word], dim=1)

            if self.vocab.idx2word[int(next_word[0,0].cpu().numpy())] == '<end>':
                break

        _, attn_w2i = torch.max(attn_w2i, dim=2)
        return seq, attn_w2w, attn_w2i



class build_model(nn.Module):
    def __init__(self, 
                 img_arch,
                 sen_arch,
                 embed_dim,
                 vocab,
                 vocab_size,
                 max_seq_length):
        """Load the pretrained ResNet and replace top fc layer."""
        super(build_model, self).__init__()
        n_head = 1
        num_layers = 1
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.encoder = Encoder(img_arch,
                               embed_dim,
                               vocab_size,
                               max_seq_length)
        self.decoder = Decoder(vocab,
                                sen_arch,
                                embed_dim, 
                                vocab_size,
                                num_layers,
                                n_head, 
                                max_seq_length)

    def get_parameters(self):
        params = list(self.parameters())

        return params

    def forward(self, images, captions):
        feats_global, feats_local, attns_img = self.encoder(images)
        seqs = self.decoder(feats_global, feats_local, captions)

        return seqs

    def sample(self, image):
        feat_global, feat_local, attn_img = self.encoder(image)
        hypothese = self.decoder.sample(feat_global, feat_local)

        return hypothese

    def calc_attn(self, image):
        feat_global, feat_local, attn_img = self.encoder(image)
        hypothese, attn_w2w, attn_w2i = self.decoder.calc_attn(feat_global, feat_local)
        
        return hypothese, attn_img, attn_w2w, attn_w2i



