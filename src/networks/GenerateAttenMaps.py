    # Generate Test Attention Maps
    x_atten, y_atten, pdb_files = load_pdb_train(data_folders, 0, 100, resize, True)
    for i in range(len(pdb_files)):
        p = net.model.predict(x_atten[i:i+1])
        atten_map = visualize_saliency(net.model, 10, [np.argmax(p[0])], x_atten[i], alpha=0.0)
        #atten_map = np.dot(atten_map[...,:3], [0.299, 0.587, 0.114])
        atten_map = misc.imresize(atten_map, (512, 512), interp='nearest')
        misc.imsave('../../data/valid/attenmaps/'+pdb_files[i].split('.')[0]+'.png', atten_map)
