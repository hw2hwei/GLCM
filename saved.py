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
