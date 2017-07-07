'''
'''
from PDBClassifier import *

def load_pdb_train(encoded_folders, i, sample=None, resize=None, names=False):
    '''
    Method loads in training images from defined folder. Images values are normalized
    betweeen 0.0 and 1.0.
    '''

    x_train = []
    y_train = []

    # Read PDB IMG File Names
    pdb_imgs = []
    for line in sorted(os.listdir('../../data/final/' + encoded_folders[i] + '/')):
        if line.endswith('.png'): pdb_imgs.append(line)
    pdb_imgs = np.array(pdb_imgs)

    # Take Random Sample
    if sample:
        np.random.seed(seed)
        np.random.shuffle(pdb_imgs)
        pdb_imgs = pdb_imgs[:sample]

    if debug: print "Loading Encoded Images From", encoded_folders[i], '...'

    # Load PDB Images
    for j in tqdm(range(len(pdb_imgs))):
        img = misc.imread('../../data/final/' + encoded_folders[i] + '/' + pdb_imgs[j])
        img = img.astype('float')
        img[:,:,0] = img[:,:,0]/255.0
        img[:,:,1] = img[:,:,1]/255.0
        img[:,:,2] = img[:,:,2]/255.0
        if resize:
            img = misc.imresize(img, resize, interp='bicubic')
            #if j < 10:
                #plt.imshow(img)
                #plt.show()
        #img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        #img = np.expand_dims(img, axis=-1)
        y_ = [0 for z in range(len(encoded_folders))]
        y_[i] = 1
        x_train.append(img)
        y_train.append(y_)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    if names: return x_train, y_train, pdb_imgs
    else: return x_train, y_train

if __name__ == '__main__':

    net = ProteinNet(shape=[512, 512, 3])
    net.model.load_weights("weights.hdf5")

    # Load Training Data
    data_folders = ['RAS-MD512-HH', 'WD40-MD512-HH']
    sample = 100
    resize = None
    x_train = None
    pdb_names = None
    for i in range(len(data_folders)):
        x_, y_, p = load_pdb_train(data_folders, i, sample, resize, True)
        if x_train is None: x_train = x_
        else: x_train = np.concatenate([x_train, x_], axis=0)
        if pdb_names is None: pdb_names = p
        else:pdb_names= np.concatenate([pdb_names, p], axis=0)

    for i in range(len(pdb_names)):
        p = net.model.predict(x_train[i:i+1])
        atten_map = visualize_saliency(net.model, 10, [np.argmax(p[0])], x_train[i], alpha=0.0)
        atten_map = np.dot(atten_map[...,:3], [0.299, 0.587, 0.114])
        #atten_map = misc.imresize(atten_map, (512, 512), interp='nearest')
        misc.imsave('../../data/valid/attenmaps/'+pdb_names[i].split('.')[0]+'.png', atten_map)
