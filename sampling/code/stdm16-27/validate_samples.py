import conduit
import conduit.relay.io
import numpy as np
#root_file = "density_full.cycle_000001.root"
#root_file = "AMR-Wind-Dam-Break.cycle_000000.root"
root_file = "the_sampled.cycle_000545.root"

data = conduit.Node()

# load everythign in
# (note the raw data is about 17 gb, make sure you have enough ram
conduit.relay.io.blueprint.read_mesh(data,root_file)

# check out how many domains we have
#print(data.number_of_children())

# print short version of the full tree (this works, even for big data sets)
#print(repr(data))

# DONT DO print(data) - that will try to trun the entire 17 gigs of data into yaml :-(

# look at the field names we have:
print(data[0]["fields"].child_names())
print(data[0]["coordsets"].child_names())

# each domain in this dataset has a field called "Density" and "ascent_ghosts"

# here are the values for the first domain:

coords_node = data[0].fetch_existing("coordsets/coords")
den_vals    = data[0].fetch_existing("fields/Density/values").value()
ghost_vals  = data[0].fetch_existing("fields/ascent_ghosts/values").value()

print(coords_node)
print(den_vals)
print(ghost_vals)

# if you want to look at the domain's spatial origin:


glob_min = 9999999999999999999999
glob_max = -9999999999999999999999

for i in range(data.number_of_children()):
    # print("Processing domain {0}".format(i))
    den_vals   = data[i].fetch_existing("fields/Density/values").value()
    # have fun :-)
    #print("domain:",i,"values=",np.min(den_vals),np.max(den_vals))
    if np.min(den_vals)<glob_min:
        glob_min=np.min(den_vals)
    if np.max(den_vals)>glob_max:
        glob_max=np.max(den_vals)

print("Global min and max:",glob_min, glob_max)

for i in range(data.number_of_children()):
    x = data[i].fetch_existing("coordsets/coords/values/x").value()
    y = data[i].fetch_existing("coordsets/coords/values/y").value()
    z = data[i].fetch_existing("coordsets/coords/values/z").value()
    print(np.shape(x),np.shape(y),np.shape(z))
    den_vals   = data[i].fetch_existing("fields/Density/values").value()
    print(np.shape(den_vals))