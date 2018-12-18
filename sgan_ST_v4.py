import os, time, itertools, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import tensorflow as tf
import h5py
import random
from keras import backend
import math
import cPickle as pickle
def two_p_corr(img):
    img = np.array(img > 0, dtype=int)
    d1, d2 = img.shape
    R = min(round(d1/2), round(d2/2))

    count = np.zeros((int(R+1), 1))
    Bn = np.zeros((int(R+1), 1))

    F = np.fft.fft2(img)
    c = np.fft.fftshift(np.fft.ifft2(F * np.conj(F)))/d1**2
    c = c.astype(float)

    y = np.amax(c)
    ic, jc = np.unravel_index(c.argmax(), c.shape)

    for i in xrange(d1):
        for j in xrange(d2):
            r = int(math.ceil(math.sqrt((i-ic)**2+(j-jc)**2)))
            if r<=round(R):
                Bn[r] = Bn[r] + c[i,j]
                count[r] = count[r] + 1
    Bn = Bn / count
    return Bn

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)
def relu(x):
    return tf.maximum(0., x)

# G(z)
def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, filter_num[0], [4, 4], strides=(2, 2), padding='same', name='conv1')
        relu1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training=isTrain), name='relu1')

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(relu1, filter_num[1], [4, 4], strides=(2, 2), padding='same', name='conv2')
        relu2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=isTrain), name='relu2')

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(relu2, filter_num[2], [4, 4], strides=(2, 2), padding='same', name='conv3')
        relu3 = tf.nn.relu(tf.layers.batch_normalization(conv3, training=isTrain), name='relu3')

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(relu3, filter_num[3], [4, 4], strides=(2, 2), padding='same', name='conv4')
        relu4 = tf.nn.relu(tf.layers.batch_normalization(conv4, training=isTrain), name='relu4')
        
        # conv5 = tf.layers.conv2d_transpose(relu4, filter_num[4], [4, 4], strides=(2, 2), padding='same')
        # relu5 = relu(tf.layers.batch_normalization(conv5, training=isTrain))
        # output layer

        conv6 = tf.layers.conv2d_transpose(relu4, 1, [4, 4], strides=(2, 2), padding='same', name='conv6')
        o = tf.nn.tanh(conv6, name='output')
        tf.add_to_collection('G_conv1', conv1)
        tf.add_to_collection('G_relu1', relu1)
        tf.add_to_collection('G_conv2', conv2)
        tf.add_to_collection('G_relu2', relu2)
        tf.add_to_collection('G_conv3', conv3)
        tf.add_to_collection('G_relu3', relu3)
        tf.add_to_collection('G_conv4', conv4)
        tf.add_to_collection('G_relu4', relu4)
        tf.add_to_collection('G_conv6', conv6)
        tf.add_to_collection('G_z', o)
        return o

# D(x)
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        # conv0 = tf.layers.conv2d(x, filter_num[4], [4, 4], strides=(2, 2), padding='same')
        # lrelu0 = lrelu(conv0, 0.2)

        conv1 = tf.layers.conv2d(x, filter_num[3], [4, 4], strides=(2, 2), padding='same', name='conv1')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain, name='bn1'), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, filter_num[2], [4, 4], strides=(2, 2), padding='same', name='conv2')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain, name='bn2'), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, filter_num[1], [4, 4], strides=(2, 2), padding='same', name='conv3')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain, name='bn3'), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, filter_num[0], [4, 4], strides=(2, 2), padding='same', name='conv4')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain, name='bn4'), 0.2)

        # output layer
        conv5 = tf.layers.conv2d(lrelu4, 1, [8, 8], strides=(1, 1), padding='valid', name='conv5')
        o = tf.nn.sigmoid(conv5, name='output')
        tf.add_to_collection('D_conv1', conv1)
        tf.add_to_collection('D_lrelu1', lrelu1)
        tf.add_to_collection('D_conv2', conv2)
        tf.add_to_collection('D_lrelu2', lrelu2)
        tf.add_to_collection('D_conv3', conv3)
        tf.add_to_collection('D_lrelu3', lrelu3)
        tf.add_to_collection('D_conv4', conv4)
        tf.add_to_collection('D_lrelu4', lrelu4)
        tf.add_to_collection('D_conv5', conv5)
        tf.add_to_collection('D_z', o)
        return o, conv5

def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    size_figure_grid = 10
    fixed_z_ = np.random.uniform(0, 1, (size_figure_grid*size_figure_grid,)+feature_vector_dim)
    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})

    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(size_figure_grid, size_figure_grid))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    two_p_corr_list = []
    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        temp_img = np.reshape(test_images[k], (128, 128))
        # binary_img = np.array(temp_img > 0., dtype=int) ###
        # two_p_corr_list.append(two_p_corr(binary_img)) ####
        ax[i, j].imshow(temp_img, cmap='Greys_r', interpolation='nearest')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)
        plt.clf()
        # with open(path[:-3]+'pickle', 'wb') as f:
        #     pickle.dump(two_p_corr_list, f)
        if num_epoch == train_epoch or num_epoch == train_epoch-1:
            # print ('safafasf')
            with open(path[:-4] + '_data.pkl', 'w') as f:
                pickle.dump(test_images, f)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)
        plt.clf()

    if show:
        plt.show()
    else:
        plt.close()


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
#     features_mean,features_var =tf.nn.moments(features,axes=[0])    
    features_mean = tf.reduce_mean(features,0)
    features = (features-features_mean)/1
    gram = backend.dot(features, backend.transpose(features))  
    return gram

def style_loss(style, combination):
    loss_temp=0.
    channels = 3
    size = height * width
    
    for i in range(batch_size):
        C = gram_matrix(combination[i])
        S = gram_matrix(style[i])
        loss_temp = tf.add(loss_temp,backend.sum(backend.square(S - C))/(4. * (channels ** 2) * (size ** 2))*3e-1)
    
    return loss_temp

def claps_loss(x):
    z_d_gen=backend.batch_flatten(x)          
    nom = tf.matmul(z_d_gen, tf.transpose(z_d_gen, perm=[1, 0]))
    denom = tf.sqrt(tf.reduce_sum(tf.square(z_d_gen), reduction_indices=[1], keep_dims=True))
    pt = tf.square(tf.transpose((nom / denom), (1, 0)) / denom)
    pt = pt - tf.diag(tf.diag_part(pt))
    pulling_term = tf.reduce_sum(pt) / (batch_size * (batch_size - 1))*4e3
    
    return pulling_term

def conv2d(x, W, stride, padding="SAME"):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    
def max_pool(x, k_size, stride, padding="SAME"):
    # use avg pooling instead, as described in the paper
    return tf.nn.max_pool(x, ksize=[1, k_size, k_size, 1], 
            strides=[1, stride, stride, 1], padding=padding)  

def vgg_layers(x):
    ##################  VGG16  ############
    f = h5py.File(path+'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5','r')
    ks = f.keys()
    # get weights and bias of VGG
    vgg16_weights=[]
    vgg16_bias=[]
    for i in range(18):
        if (len(f[ks[i]].values())) != 0:        
            vgg16_weights.append(f[ks[i]].values()[0][:])
            vgg16_bias.append(f[ks[i]].values()[1][:])
        else:
            continue
    del f
    W_conv1 = (tf.constant(vgg16_weights[0]))
    W_conv2 = (tf.constant(vgg16_weights[1]))
    W_conv3 = (tf.constant(vgg16_weights[2]))
    W_conv4 = (tf.constant(vgg16_weights[3]))
    # W_conv5 = (tf.constant(vgg16_weights[4]))
    # W_conv6 = (tf.constant(vgg16_weights[5]))
    # W_conv7 = (tf.constant(vgg16_weights[6]))
    # W_conv8 = (tf.constant(vgg16_weights[7]))
    # W_conv9 = (tf.constant(vgg16_weights[8]))
    # W_conv10= (tf.constant(vgg16_weights[9]))
    # W_conv11= (tf.constant(vgg16_weights[10]))
    # W_conv12= (tf.constant(vgg16_weights[11]))
    # W_conv13= (tf.constant(vgg16_weights[12]))
    b_conv1 = tf.reshape(tf.constant(vgg16_bias[0]),[-1])
    b_conv2 = tf.reshape(tf.constant(vgg16_bias[1]),[-1])
    b_conv3 = tf.reshape(tf.constant(vgg16_bias[2]),[-1])
    b_conv4 = tf.reshape(tf.constant(vgg16_bias[3]),[-1])
    # b_conv5 = tf.reshape(tf.constant(vgg16_bias[4]),[-1])
    # b_conv6 = tf.reshape(tf.constant(vgg16_bias[5]),[-1])
    # b_conv7 = tf.reshape(tf.constant(vgg16_bias[6]),[-1])
    # b_conv8 = tf.reshape(tf.constant(vgg16_bias[7]),[-1])
    # b_conv9 = tf.reshape(tf.constant(vgg16_bias[8]),[-1])
    # b_conv10 = tf.reshape(tf.constant(vgg16_bias[9]),[-1])
    # b_conv11 = tf.reshape(tf.constant(vgg16_bias[10]),[-1])
    # b_conv12 = tf.reshape(tf.constant(vgg16_bias[11]),[-1])
    # b_conv13 = tf.reshape(tf.constant(vgg16_bias[12]),[-1])
    del vgg16_bias
    del vgg16_weights
    #########  VGG  ################
    # style transfer for generated images
    ######### block 1 ########
    conv_out1 = conv2d(x, W_conv1, stride=1, padding='SAME')
    conv_out1 = tf.nn.bias_add(conv_out1, b_conv1)
    conv_out1 = tf.nn.relu(conv_out1)

    conv_out2 = conv2d(conv_out1, W_conv2, stride=1, padding='SAME')
    conv_out2 = tf.nn.bias_add(conv_out2, b_conv2)
    conv_out2 = tf.nn.relu(conv_out2)
    conv_out2 = max_pool(conv_out2, k_size=2, stride=2, padding="SAME")

    ######### block 2 ########
    conv_out3 = conv2d(conv_out2, W_conv3, stride=1, padding='SAME')
    conv_out3 = tf.nn.bias_add(conv_out3, b_conv3)
    conv_out3 = tf.nn.relu(conv_out3)

    conv_out4 = conv2d(conv_out3, W_conv4, stride=1, padding='SAME')
    conv_out4 = tf.nn.bias_add(conv_out4, b_conv4)
    conv_out4 = tf.nn.relu(conv_out4)
    conv_out4 = max_pool(conv_out4, k_size=2, stride=2, padding="SAME")

    ######### block 3 ########
    # conv_out5 = conv2d(conv_out4, W_conv5, stride=1, padding='SAME')
    # conv_out5 = tf.nn.bias_add(conv_out5, b_conv5)
    # conv_out5 = tf.nn.relu(conv_out5)

    # conv_out6 = conv2d(conv_out5, W_conv6, stride=1, padding='SAME')
    # conv_out6 = tf.nn.bias_add(conv_out6, b_conv6)
    # conv_out6 = tf.nn.relu(conv_out6)

    # conv_out7 = conv2d(conv_out6, W_conv7, stride=1, padding='SAME')
    # conv_out7 = tf.nn.bias_add(conv_out7, b_conv7)
    # conv_out7 = tf.nn.relu(conv_out7)
    # conv_out7 = max_pool(conv_out7, k_size=2, stride=2, padding="SAME")

    # ######### block 4 ########
    # conv_out8 = conv2d(conv_out7, W_conv8, stride=1, padding='SAME')
    # conv_out8 = tf.nn.bias_add(conv_out8, b_conv8)
    # conv_out8 = tf.nn.relu(conv_out8)

    # conv_out9 = conv2d(conv_out8, W_conv9, stride=1, padding='SAME')
    # conv_out9 = tf.nn.bias_add(conv_out9, b_conv9)
    # conv_out9 = tf.nn.relu(conv_out9)

    # conv_out10= conv2d(conv_out9, W_conv10, stride=1, padding='SAME')
    # conv_out10= tf.nn.bias_add(conv_out10, b_conv10)
    # conv_out10= tf.nn.relu(conv_out10)
    # conv_out10 = max_pool(conv_out10, k_size=2, stride=2, padding="SAME")

    # ######### block 5 ########
    # conv_out11= conv2d(conv_out10, W_conv11, stride=1, padding='SAME')
    # conv_out11= tf.nn.bias_add(conv_out11, b_conv11)
    # conv_out11= tf.nn.relu(conv_out11)

    # conv_out12= conv2d(conv_out11, W_conv12, stride=1, padding='SAME')
    # conv_out12= tf.nn.bias_add(conv_out12, b_conv12)
    # conv_out12= tf.nn.relu(conv_out12)

    # conv_out13= conv2d(conv_out12, W_conv13, stride=1, padding='SAME')
    # conv_out13= tf.nn.bias_add(conv_out13, b_conv12)
    # conv_out13= tf.nn.relu(conv_out13)
    return conv_out1, conv_out2, conv_out3, conv_out4



# training parameters
filter_num = [128, 64, 32, 16] # five numbers from big to small
feature_vector_dim = (4, 4, 1)
batch_size = 30
lr = 0.0005
train_epoch = int(1.5e1)
img_dim = (128,128,1)
height, width = img_dim[0], img_dim[1]
nowtime = 'test_restore'
D_steps = 3
G_steps = 1
snapshot_interval = 500 # save image generated by generator 

# load data
path = '/raid/zyz293/PSED_data/'
data_path = path + 'data_-8_3_0.25(0.1).mat' #needs to be a .mat file
data_var_name = 'IMG'
img_collection_data = h5py.File(data_path, 'r')
img_collection = np.transpose(img_collection_data[data_var_name])
del img_collection_data
# variables : input
x = tf.placeholder(tf.float32, shape=(None,) + img_dim, name='x')
z = tf.placeholder(tf.float32, shape=(None,) + feature_vector_dim, name='z')
isTrain = tf.placeholder(dtype=tf.bool, name='isTrain')
train_set_orig = img_collection.reshape(((len(img_collection),)+ img_dim))
style_array = np.repeat(train_set_orig, 3, axis=-1)
del img_collection

# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)


######## style transfer for generated images and real images ########
combination_image_temp=tf.reshape(G_z,[batch_size, height, width, 1])
combination_image = tf.concat([combination_image_temp, combination_image_temp,combination_image_temp], 3)
style_image = tf.placeholder(tf.float32, shape=(batch_size,height,width,3), name='style_image')

conv_out1, conv_out2, conv_out3, conv_out4 = vgg_layers(combination_image)
conv_out1_S, conv_out2_S, conv_out3_S, conv_out4_S = vgg_layers(style_image)
# style loss
sl1 = style_loss(conv_out2_S,conv_out2)
sl2 = style_loss(conv_out4_S,conv_out4)
sl3 = style_loss(conv_out1_S,conv_out1)
sl4 = style_loss(conv_out3_S,conv_out3)
sl_loss = tf.reduce_mean(sl1 + sl2 + sl3 + sl4)
# # claps cost
cl1 = claps_loss(conv_out2)
cl2 = claps_loss(conv_out4)
cl3 = claps_loss(conv_out1)
cl4 = claps_loss(conv_out3)
cl_loss = tf.reduce_mean(cl1 + cl2 + cl3+cl4)


# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))
G_loss = G_loss + 0.03*cl_loss + 0.03*sl_loss

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-7).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-7).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# results save folder
dir_name = path + 'log/' + nowtime + '/model'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
plot_name = path + 'log/' + nowtime + '/plot'
if not os.path.exists(plot_name):
    os.makedirs(plot_name)

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()

D_starting_idx = 0
G_starting_idx = 0
D_num_samples = train_epoch * D_steps * batch_size
G_num_samples = train_epoch * G_steps * batch_size
D_idx_permutation = np.array([])
G_idx_permutation = np.array([])

while D_idx_permutation.shape[0] < D_num_samples:
    D_idx_permutation = np.concatenate((D_idx_permutation, np.random.permutation(train_set_orig.shape[0])), 0)

while G_idx_permutation.shape[0] < G_num_samples:    
    G_idx_permutation = np.concatenate((G_idx_permutation, np.random.permutation(train_set_orig.shape[0])), 0)
D_idx_permutation = D_idx_permutation.astype(int)
G_idx_permutation = G_idx_permutation.astype(int)

for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for D_i in xrange(D_steps):
        #trainID = random.sample(range(train_set_orig.shape[0]), batch_size)
        trainID = D_idx_permutation[D_starting_idx: D_starting_idx+batch_size]
        # print trainID
        x_ = train_set_orig[trainID,:,:,:]
        
        D_starting_idx += batch_size
        z_ = np.random.uniform(0, 1, (batch_size,)+feature_vector_dim)

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
        D_losses.append(loss_d_)
    for G_i in xrange(G_steps):
        # update generator
        #trainID = random.sample(range(train_set_orig.shape[0]), batch_size)
        trainID = G_idx_permutation[G_starting_idx: G_starting_idx+batch_size]
        #x_ = train_set_orig[trainID,:,:,:]
        
        G_starting_idx += batch_size

        z_ = np.random.uniform(0, 1, (batch_size,)+feature_vector_dim)
        style_image_input = style_array[trainID, :, :, :]
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, style_image:style_image_input, isTrain: True})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    if epoch % snapshot_interval in [snapshot_interval-2, snapshot_interval-1,0]:
        fixed_p = plot_name + '/' + str(epoch + 1) + '.png'
        # print (fixed_p)
        show_result((epoch + 1), save=True, path=fixed_p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(path + '/log/' + nowtime + '/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=path + '/log/' + nowtime + '/train_hist.png')

# images = []
# for e in range(train_epoch):
#     img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
#     images.append(imageio.imread(img_name))
# imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)
print (nowtime)
saver = tf.train.Saver()
saver.save(sess, dir_name+'/model', global_step=train_epoch-1)
# saver.save(sess, dir_name+'/model', global_step=train_epoch-1)
# saver.export_meta_graph(dir_name+'/model', global_step=train_epoch-1)
sess.close()