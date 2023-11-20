# import modules
import numpy as np
import matplotlib.pyplot as plt
import imageio

# load and plot input image
img = imageio.imread('gfx/image.png')/255

# model parameters
[h,w] = img.shape # get width & height of image
num_vars = w*h    # number of variables = width * height
num_states = 2    # binary segmentation -> two states

# initialize factors (list of dictionaries), each factor comprises:
#   vars: array of variables involved
#   vals: vector/matrix of factor values
factors = []
# add unary factors
for u in range(w):
  for v in range(h):
    factors.append({'vars':np.array([v*w+u]), 'vals':np.array([1-img[v,u],img[v,u]])})

# add pairwise factors
alpha = 0.4 # smoothness weight
E = alpha*np.array([[1,0],[0,1]]) # energy matrix for pairwise factor
for u in range(w):
  for v in range(h):
    if v<h-1:
      factors.append({'vars':np.array([v*w+u,(v+1)*w+u]), 'vals':E})
    if u<w-1:
      factors.append({'vars':np.array([v*w+u,v*w+u+1]), 'vals':E})
      
      
# initialize all messages
msg_fv = {} # f->v messages (dictionary)
msg_vf = {} # v->f messages (dictionary)
ne_var = [[] for i in range(num_vars)] # neighboring factors of variables (list of list)

# set messages to zero; determine factors neighboring each variable
for [f_idx,f] in enumerate(factors):
    for v_idx in f['vars']:
        msg_fv[(f_idx,v_idx)] = np.zeros(num_states) # factor->variable message
        msg_vf[(v_idx,f_idx)] = np.zeros(num_states) # variable->factor message
        ne_var[v_idx].append(f_idx) # factors neighboring variable v_idx

# status message
print("Messages initialized!")


# run inference
for it in range(30):
    # for all factor-to-variable messages do
    for [key,msg] in msg_fv.items():

        # shortcuts to variables
        f_idx = key[0] # factor (source)
        v_idx = key[1] # variable (target)
        f_vars = factors[f_idx]['vars'] # variables connected to factor
        f_vals = factors[f_idx]['vals'] # vector/matrix of factor values 

        # unary factor-to-variable message
        if np.size(f_vars)==1:
            msg_fv[(f_idx,v_idx)] = f_vals

        # pairwise factor-to-variable-message
        else:
            # if target variable is first variable of factor
            if v_idx==f_vars[0]:
                msg_in = np.tile(msg_vf[(f_vars[1],f_idx)],(num_states,1))
                msg_fv[(f_idx,v_idx)] = (f_vals+msg_in).max(1) # max over columns

            # if target variable is second variable of factor
            else:
                msg_in = np.tile(msg_vf[(f_vars[0],f_idx)],(num_states,1))
                msg_fv[(f_idx,v_idx)] = (f_vals+msg_in.transpose()).max(0) # max over rows

        # normalize
        msg_fv[(f_idx,v_idx)] = msg_fv[(f_idx,v_idx)] - np.mean(msg_fv[(f_idx,v_idx)])

    # for all variable-to-factor messages do
    for [key,msg] in msg_vf.items():

        # shortcuts to variables
        v_idx = key[0] # variable (source)
        f_idx = key[1] # factor (target)

        # add messages from all factors send to this variable (except target factor)
        # and send the result to the target factor
        msg_vf[(v_idx,f_idx)] = np.zeros(num_states)
        for f_idx2 in ne_var[v_idx]:
            if f_idx2 != f_idx:
                msg_vf[(v_idx,f_idx)] += msg_fv[(f_idx2,v_idx)]

        # normalize
        msg_vf[(v_idx,f_idx)] = msg_vf[(v_idx,f_idx)] - np.mean(msg_vf[(v_idx,f_idx)])



# calculate max-marginals (num_vars x num_states matrix)
max_marginals = np.zeros([num_vars,num_states])
for v_idx in range(num_vars):

    # add messages from all factors sent to this variable
    max_marginals[v_idx] = np.zeros(num_states)
    for f_idx in ne_var[v_idx]:
        max_marginals[v_idx] += msg_fv[(f_idx,v_idx)]
    #print max_marginals[v_idx]

# get MAP solution
map_est = np.argmax(max_marginals,axis=1)

# plot MAP estimate
plt.imshow(map_est.reshape(h,w),interpolation='nearest');
plt.gray()
plt.show()