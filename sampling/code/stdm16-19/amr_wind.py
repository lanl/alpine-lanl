#/gpfs/alpine/world-shared/csc340/Nyx/conduit_ayan_nyx_example.py
# /gpfs/alpine/world-shared/csc340/software/ascent/current/summit/openmp/gnu/python-install/bin/python 
#====
 
import conduit
import conduit.relay.io
import numpy as np
#root_file = "density_full.cycle_000001.root"
root_file = "AMR-Wind-Dam-Break.cycle_000000.root"
 
data = conduit.Node()
 
# load everythign in
# (note the raw data is about 17 gb, make sure you have enough ram
conduit.relay.io.blueprint.read_mesh(data,root_file)
 
# check out how many domains we have
print(data.number_of_children())
 
# print short version of the full tree (this works, even for big data sets)
print(repr(data))
 
# DONT DO print(data) - that will try to trun the entire 17 gigs of data into yaml :-(
 
# look at the field names we have:
print(data[0]["fields"].child_names())
 
# each domain in this dataset has a field called "Density" and "ascent_ghosts"
 
# here are the values for the first domain:
 
# coords_node = data[0].fetch_existing("coordsets/coords")
# den_vals    = data[0].fetch_existing("fields/Density/values").value()
# ghost_vals  = data[0].fetch_existing("fields/ascent_ghosts/values").value()
 
# print(coords_node)
# print(den_vals)
# print(ghost_vals)
 
# if you want to look at the domain's spatial origin:
 
# print(coords_node["origin"])
# o_x = coords_node["origin/x"]
# o_y = coords_node["origin/y"]
# o_z = coords_node["origin/z"]
 
# print("origin: {0} {1} {2}".format(o_x,o_y,o_z))
 
# for i in range(data.number_of_children()):
#     # print("Processing domain {0}".format(i))
#     den_vals   = data[i].fetch_existing("fields/Density/values").value()
#     # have fun :-)
# dx_lst = []
# for i in range(data.number_of_children()):
#     x = str(data[i].fetch_existing("coordsets/coords/spacing/dx").value())
#     x = str(data[i].fetch_existing("coordsets/coords/spacing/y").value())
#     dx_lst.append(dx)

# for i in range(data.number_of_children()):
#     x = data[i].fetch_existing("coordsets/coords/values/x").value()
#     dx = x[1]-x[0]
#     y = data[i].fetch_existing("coordsets/coords/values/y").value()
#     dy = y[1]-y[0]
#     z = data[i].fetch_existing("coordsets/coords/values/z").value()
#     dz = z[1]-z[0]
#     print(dx,dy,dz)

# set(dx_lst)
#np.unique(np.asarray(dx_lst))

# get data from one domain

# domain_id = 0
# ghost = data[domain_id].fetch_existing("fields/ghost_indicator/values").value()

# var_name =  'rho.Y(HO2)'  #"rho.Y(H2O)"

# var_vals = data[domain_id].fetch_existing("fields/"+var_name+"/values").value()
# min_val = np.max(var_vals)
# max_val = np.min(var_vals)

# print(np.max(var_vals),np.min(var_vals))

var_name =  'velocity_magnitude' #'rho.Y(HO2)'  #"rho.Y(H2O)"

glob_min = 99999999.9
glob_max = -99999999.9
min_dx = 999999999.0
for domain_id in range(data.number_of_children()):
    x = data[domain_id].fetch_existing("coordsets/coords/values/x").value()
    dx = x[1]-x[0]
    y = data[domain_id].fetch_existing("coordsets/coords/values/y").value()
    dy = y[1]-y[0]
    z = data[domain_id].fetch_existing("coordsets/coords/values/z").value()
    dz = z[1]-z[0]
    if dx < min_dx:
        min_dx = dx
    # mult_fact = int(str(dx)[0])
    # print(dx,dy,dz,mult_fact)
    var_vals = data[domain_id].fetch_existing("fields/"+var_name+"/values/c0").value()
    ghost = data[domain_id].fetch_existing("fields/avtGhostZones/values/c0").value()
    print(ghost)
    # only where ghost is 0
    max_val = np.max(var_vals[ghost<0.5])
    min_val = np.min(var_vals[ghost<0.5])
    if glob_max<max_val:
        glob_max=max_val
    if glob_min>min_val:
        glob_min=min_val

print(glob_min,glob_max)

nbins = 32
# hist_arr = np.zeros((nbins),'int64')
# orig_hist_arr = np.zeros((nbins),'int64')

hist_arr = np.zeros((nbins),'float64')
orig_hist_arr = np.zeros((nbins),'float64')

for domain_id in range(data.number_of_children()):
    x = data[domain_id].fetch_existing("coordsets/coords/values/x").value()
    dx = x[1]-x[0]
    y = data[domain_id].fetch_existing("coordsets/coords/values/y").value()
    dy = y[1]-y[0]
    z = data[domain_id].fetch_existing("coordsets/coords/values/z").value()
    dz = z[1]-z[0]    
    #print(dx,dy,dz)
    mult_fact = (dx/min_dx)**3
    # dx = data[domain_id].fetch_existing("coordsets/coords/spacing/dx").value()
    # dy = data[domain_id].fetch_existing("coordsets/coords/spacing/dy").value()
    # dz = data[domain_id].fetch_existing("coordsets/coords/spacing/dz").value()
    # mult_fact = int(str(dx)[0])
    #print(dx,dy,dz,mult_fact)
    var_vals = data[domain_id].fetch_existing("fields/"+var_name+"/values/c0").value()
    ghost = data[domain_id].fetch_existing("fields/avtGhostZones/values/c0").value()
    arr, edgs = np.histogram(var_vals[ghost<0.5],bins=nbins, range = [glob_min,glob_max])
    hist_arr = hist_arr + arr*mult_fact
    orig_hist_arr = orig_hist_arr + arr
    print(np.sum(arr),np.sum(arr*mult_fact))


print("scaled histogram:",hist_arr, np.sum(hist_arr))
print("original histogram:",orig_hist_arr, np.sum(orig_hist_arr))


# nbins = 32
# orig_hist_arr = np.zeros((nbins),'int64')

# for domain_id in range(data.number_of_children()):
#     var_vals = data[domain_id].fetch_existing("fields/"+var_name+"/values").value()
#     ghost = data[domain_id].fetch_existing("fields/ghost_indicator/values").value()
#     arr, edgs = np.histogram(var_vals[ghost<0.5],bins=nbins, range = [glob_min,glob_max])
#     orig_hist_arr = orig_hist_arr + arr
# print("original histogram:",orig_hist_arr)

## Now apply sampling

# first create the importance histogram

def create_acceptance_histogram(frac, count, nbins=32):
    tot_samples = frac*np.sum(count)
    print('looking for',tot_samples,'samples')
    # create a dictionary first
    my_dict = dict() 
    ind = 0
    for i in count:
        my_dict[ind] = i
        ind = ind + 1
    print(my_dict)
    sorted_count = sorted(my_dict, key=lambda k: my_dict[k])
    print(sorted_count)
    ## now distribute across bins
    target_bin_vals = int(tot_samples/nbins)
    print('ideal cut:',target_bin_vals)
    new_count = np.copy(count)
    ind = 0
    remain_tot_samples = tot_samples
    for i in sorted_count:
        if my_dict[i]>target_bin_vals:
            val = target_bin_vals
        else:
            val = my_dict[i]
            remain = target_bin_vals-my_dict[i]
        new_count[i]=val
        #print(new_count[i], target_bin_vals)
        ind = ind + 1
        remain_tot_samples = remain_tot_samples-val
        if ind < nbins:
            target_bin_vals = int(remain_tot_samples/(nbins-ind))
    print(new_count)  
    print(count) 
    acceptance_hist = new_count/count
    np.nan_to_num(acceptance_hist,nan=0.0)
    return acceptance_hist

def sample_using_acceptance(data,acceptance_hist,bound_min, bound_max):
    s = data
    tot_pts = np.size(s)
    prob_vals = np.zeros_like(s)
    stencil = np.zeros_like(s)
    rand_vals = np.random.random_sample(tot_pts)
    # bound_min = np.min(s)
    # bound_max = np.max(s)
    for i in range(tot_pts):
        loc = s.flatten()[i]
        x_id = int(nbins * (loc - bound_min) / (bound_max - bound_min))
        if x_id == nbins:
            x_id = x_id - 1
        prob_vals[i]=acceptance_hist[x_id]
    stencil[rand_vals<prob_vals]=1
    print("actually generating samples: ",np.sum(stencil))
    # now use this stencil array to store the locations
    int_inds = np.where(stencil>0.5)
    # plt.imshow(one_img, cmap='Greys')
    # plt.show()
    # plt.imshow(stencil.reshape((28,28)), cmap='Greys')
    # plt.show()
    return int_inds,stencil

#accept_hist = create_acceptance_histogram(0.03,orig_hist_arr)
accept_hist = create_acceptance_histogram(0.03,hist_arr)
print(accept_hist)
samps_taken = 0
gx_list = []
gy_list = []
gz_list = []
gval_list = []
for domain_id in range(data.number_of_children()):
    print("DOMAIN:",domain_id)
    x_list = []
    y_list = []
    z_list = []
    val_list = []
    ghost = data[domain_id].fetch_existing("fields/avtGhostZones/values/c0").value()
    print(np.shape(ghost))
    x = data[domain_id].fetch_existing("coordsets/coords/values/x").value()
    print(np.shape(x),np.min(x),np.max(x))
    y = data[domain_id].fetch_existing("coordsets/coords/values/y").value()
    print(np.shape(y),np.min(y),np.max(y))
    z = data[domain_id].fetch_existing("coordsets/coords/values/z").value()
    print(np.shape(z),np.min(z),np.max(z))
    var_vals = data[domain_id].fetch_existing("fields/"+var_name+"/values/c0").value()
    local_data = var_vals[ghost<0.5]
    data_locs = np.where(ghost<0.5)
    data_locs_inds = np.asarray(data_locs)[0]
    arr, edgs = np.histogram(local_data,bins=nbins, range = [glob_min,glob_max])
    print('local histogram:',arr)
    inds, stencil = sample_using_acceptance(local_data, accept_hist,glob_min, glob_max)
    print(np.sum(stencil))
    samps_taken += np.sum(stencil)
    val_list = np.append(val_list, local_data[inds])
    gval_list = np.append(gval_list, local_data[inds])
    inds = np.asarray(inds)[0]
    inds = data_locs_inds[inds]
    print(inds)

    XDIM = np.size(x)-1
    YDIM = np.size(y)-1
    ZDIM = np.size(z)-1
    zids = (inds/(XDIM*YDIM)).astype(int)
    yids = ((inds%(XDIM*YDIM))/XDIM).astype(int)
    xids = (inds%(XDIM*YDIM))%XDIM
    x_list = np.append(x_list,x[xids])
    y_list = np.append(y_list,y[yids])
    z_list = np.append(z_list,z[zids])
    np.save('x_vals'+str(domain_id)+'.npy',np.asarray(x_list))
    np.save('y_vals'+str(domain_id)+'.npy',np.asarray(y_list))
    np.save('z_vals'+str(domain_id)+'.npy',np.asarray(z_list))
    np.save('var_vals'+str(domain_id)+'.npy',np.asarray(val_list))

    gx_list = np.append(gx_list,x[xids])
    gy_list = np.append(gy_list,y[yids])
    gz_list = np.append(gz_list,z[zids])
print("samples actually taken:",samps_taken)
# print(x_list)
np.save('x_vals.npy',np.asarray(gx_list))
np.save('y_vals.npy',np.asarray(gy_list))
np.save('z_vals.npy',np.asarray(gz_list))
np.save('var_vals.npy',np.asarray(gval_list))