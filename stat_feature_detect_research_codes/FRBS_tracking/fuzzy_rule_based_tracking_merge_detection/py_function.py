###############################################
## Imports
###############################################
import numpy as np
import pandas as pd
import os
import math
from numpy import linalg as LA
#####################################################

#%matplotlib inline
#Global variables
# my_id = -1
# cur_jac=-1

### dice Similarity function of 2 set of points
def dice_similarity(x,y):
  
 intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
 union_cardinality = len(set(x)) + len(set(y))
 return 2*intersection_cardinality/float(union_cardinality)

### Jaccard Similarity function of 2 set of points
def jaccard_similarity(x,y):
  
 intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
 union_cardinality = len(set.union(*[set(x), set(y)]))
 return intersection_cardinality/float(union_cardinality)

## computes best estimate using jaccard match    
def get_best_jaccard_match(X_new,target_pts,count):
    
    ## this lets code know that my_id and cur_jac are global vars
    #global my_id
    #global cur_jac
    #cur_jac = 0
   
    list_of_lists = []
    list1 = []
    
    # first object
    for i in range(np.size(X_new[:,0])):
        list1.append((X_new[i,0],X_new[i,1],X_new[i,2]))
    
    # second object with possible rotations
    list2 = []
    for i in range(np.size(target_pts[:,0])):
        list2.append((target_pts[i,0],target_pts[i,1],target_pts[i,2]))
    list_of_lists.append(list2)
    list2 = []
    for i in range(np.size(target_pts[:,0])):
        list2.append((target_pts[i,0],-target_pts[i,1],target_pts[i,2]))
    list_of_lists.append(list2)
    list2 = []
    for i in range(np.size(target_pts[:,0])):
        list2.append((-target_pts[i,0],target_pts[i,1],target_pts[i,2]))
    list_of_lists.append(list2)
    list2 = []
    for i in range(np.size(target_pts[:,0])):
        list2.append((-target_pts[i,0],-target_pts[i,1],target_pts[i,2]))
    list_of_lists.append(list2)

    list2 = []
    for i in range(np.size(target_pts[:,0])):
        list2.append((target_pts[i,0],target_pts[i,1],-target_pts[i,2]))
    list_of_lists.append(list2)
    list2 = []
    for i in range(np.size(target_pts[:,0])):
        list2.append((target_pts[i,0],-target_pts[i,1],-target_pts[i,2]))
    list_of_lists.append(list2)
    list2 = []
    for i in range(np.size(target_pts[:,0])):
        list2.append((-target_pts[i,0],target_pts[i,1],-target_pts[i,2]))
    list_of_lists.append(list2)
    list2 = []
    for i in range(np.size(target_pts[:,0])):
        list2.append((-target_pts[i,0],-target_pts[i,1],-target_pts[i,2]))
    list_of_lists.append(list2)

    jac_list = []
    dice_list = []
    
    # find best jaccard by rotating
    for i in range(8):
        jac_val = dice_similarity(list1,list_of_lists[i])
        jac_list.append(jac_val)        

    ########################################   
    #jac_list.sort()
    jac_val = max(jac_list)
    max_index = jac_list.index(max(jac_list))
        
    #print "dice index: " + str(jac_val) + "  feature id jaccard: " +  str(count)

    return jac_val


########################################################################################

def main_func(timestep):

    cur_jac =0
    ## read the target feature
    from numpy import genfromtxt
    filepath = "/home/soumya/comb_features/matched/matched_feature_" + str(timestep+1)+".csv" 
    target_feature = genfromtxt(filepath,delimiter=',')
        
    ################################################################################################
    ## Process and compare with other features
    ################################################################################################    

    DIR = '/home/soumya/comb_features/'
    num_feature_comb = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

    ## prepare the matched feature for comparison as well
    matched_feature = '/home/soumya/comb_features/comb_feature_'+str(timestep)+"_"+str(num_feature_comb-1)+'.csv' 
    os.rename('/home/soumya/comb_features/temp.csv', matched_feature)
    
    ## Loop over all the features
    for count in range(0,num_feature_comb):

        ## read the comb feature
        from numpy import genfromtxt
        current_feature_comb = genfromtxt("/home/soumya/comb_features/comb_feature_"+str(timestep)+"_"+str(count)+".csv",delimiter=',')
        
        #compute jaccard                
        ret_val = get_best_jaccard_match(target_feature,current_feature_comb,count)

        if ret_val > cur_jac:
        	my_id = count
      		cur_jac = ret_val

    print "Detected id " + str(my_id) + " with jaccard val = " + str(cur_jac)

    return my_id
##############################################################################################        
