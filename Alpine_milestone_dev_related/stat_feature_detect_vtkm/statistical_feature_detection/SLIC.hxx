//Header for SLIC filter
#ifndef vtk_m_filter_SLIC_hxx
#define vtk_m_filter_SLIC_hxx

#include "SLICHelper.h"

namespace vtkm
{
    namespace filter
    {
        //-----------------------------------------------------------------------------
        template <typename T, typename StorageType, typename DerivedPolicy>
        inline VTKM_CONT vtkm::cont::DataSet SLIC::DoExecute(
        const vtkm::cont::DataSet& inDataSet, 
        const vtkm::cont::ArrayHandle<T, StorageType>& inField,
        const vtkm::filter::FieldMetadata& fieldMetadata, 
        const vtkm::filter::PolicyBase<DerivedPolicy>&)
        {
            vtkm::Float64 eps = 1e+8;

            //// compute number of blocks/clusters at each dimension
            const vtkm::Id tot_num_pts = this->xdim*this->ydim*this->zdim;
            const vtkm::Id num_xblocks=this->xdim/this->blockXSize;
            const vtkm::Id num_yblocks=this->ydim/this->blockYSize;
            const vtkm::Id num_zblocks=this->zdim/this->blockZSize;
            vtkm::Id num_clusters = num_xblocks*num_yblocks*num_zblocks;

            ////Get datavariable array into an array handle
            vtkm::cont::ArrayHandle<vtkm::FloatDefault> dataval_array;
            inDataSet.GetField(this->SlicFieldName).GetData().AsArrayHandle(dataval_array);

            //Normalize the data array
            vtkm::cont::ArrayHandle<vtkm::Range> dataval_range = vtkm::cont::ArrayRangeCompute(dataval_array);
            // auto dataval_array_normalized = vtkm::cont::make_ArrayHandleTransform(dataval_array,
            //                 NormalizeArray<vtkm::Float64>(vtkm::make_Pair(dataval_range.ReadPortal().Get(0).Min,
            //                                                               dataval_range.ReadPortal().Get(0).Max)));


            vtkm::cont::ArrayHandle<vtkm::Float64> dataval_array_normalized;
            vtkm::cont::Invoker invoke6;
            NormalizeArrayNew normArray(dataval_range.ReadPortal().Get(0).Min,
                                        dataval_range.ReadPortal().Get(0).Max);
            invoke6(normArray,
                    dataval_array,
                    dataval_array_normalized);

            //// Initialize the initial cluster centers using existing vtkm functionality
            vtkm::cont::ArrayHandleUniformPointCoordinates init_centers(vtkm::Id3(num_xblocks,num_yblocks,num_zblocks),
                                                            vtkm::Id3( this->blockXSize/2,
                                                                       this->blockYSize/2,
                                                                       this->blockZSize/2 ), // TODO
                                                            vtkm::Id3 (this->blockXSize,
                                                                       this->blockYSize,
                                                                       this->blockZSize) );

            // This line creates an arrayhandle with values 0,1,..num_clusters. It does not allocate any memory
            vtkm::cont::ArrayHandleIndex unique_clusterids(num_clusters);
            vtkm::cont::ArrayHandle<vtkm::Id> intermediate_unique_cluster_ids;
            // Copy unique_clusterids to an actual arrayhandle intermediate_unique_cluster_ids that can be used in the code
            vtkm::cont::Algorithm::Copy(unique_clusterids, intermediate_unique_cluster_ids);

            vtkm::cont::ArrayHandle<vtkm::Id> init_centersX;
            vtkm::cont::ArrayHandle<vtkm::Id> init_centersY;
            vtkm::cont::ArrayHandle<vtkm::Id> init_centersZ;
            vtkm::cont::Invoker invoke5;
            CopyCentersTo1DArray copyCenterIds;
            invoke5(copyCenterIds,
                    init_centers,
                    init_centersX,
                    init_centersY,
                    init_centersZ);

            /////////////////////////////////////////////////////////////////////////
            //Generate the point ids for all data points. Does not allocate memory
            vtkm::cont::ArrayHandleIndex input_index_arr_allpts(tot_num_pts);

            //Generate 3D ids from 1D ids
            vtkm::cont::ArrayHandle <vtkm::Float64> all_pointsX1;
            vtkm::cont::ArrayHandle <vtkm::Float64> all_pointsY1;
            vtkm::cont::ArrayHandle <vtkm::Float64> all_pointsZ1;
            vtkm::cont::Invoker invoke3;
            ComputeAllPointIDs compPtsIds(vtkm::Id3(xdim,ydim,zdim));
            invoke3(compPtsIds,
                    input_index_arr_allpts,
                    all_pointsX1,
                    all_pointsY1,
                    all_pointsZ1);

            /// Ollie: This is also unnecessary. Now we have coordinates for the points/cell center, we can
            /// easily create an unstructured grid of points. Take a look at Chapter 7.1.3 on how to build
            /// a DataSet with an CellSetExplicit and Chapter 7.2.2 on the CellSetExplicit itself and CellSetSingleType.
            /// Ollie: Once you are done, take a look at the rather undocumented filter/Probe.h and worklet/Probe.h and
            /// see if you can use them for your purpose.
            ////Convert 3D indices to 1D indices before look up
            auto oneD_centers = vtkm::cont::make_ArrayHandleTransform(init_centers,
                                                                      ComputeThreeDtoOneDIndex<vtkm::Id>(this->xdim,
                                                                                                        this->ydim));

            //Look up data values for initial cluster centers
            vtkm::cont::ArrayHandle <vtkm::Float64> init_vals;
            vtkm::cont::Invoker invoke;
            invoke(ComputeInitValues{},
                    oneD_centers, 
                    dataval_array_normalized,
                    init_vals);


            // Initialize cluster ids for each point
            ////////////////////////////////////////////////////////////////////////////////////////////////
            vtkm::cont::ArrayHandle <vtkm::Int32> cluster_ids;
            vtkm::cont::Invoker invoke4;
            AssignClusterIDs assignClusterIds(vtkm::Id3(xdim,ydim,zdim),
                                            vtkm::Id3(blockXSize,blockYSize,blockZSize));
            invoke4(assignClusterIds,
                    input_index_arr_allpts,
                    cluster_ids);

            /////////////////////////////////////////////////////////////////////////
            ///At this point we have initialized the cluster centers and cluster ids
            /////////////////////////////////////////////////////////////////////////


            // //////////////////////////////////////////////////////////////////////////////////////////////////
            // SLIC iteration loop: Only immediate neighboring cluster centers are considered
            /////////////////////////////////////////////////////////////////////////////////////////////////////
            vtkm::Id iter = 0;
            vtkm::Id current_cluster_num = 0;
            
            // While loop will continue untill cluster centers are converged
            while( (eps > this->halt_condition) && (iter < this->iterLimit) )
            {
                vtkm::Id prev_cluster_num = init_centersX.GetNumberOfValues();

                vtkm::cont::ArrayHandle <vtkm::Id> mapped_indices;
                ComputeClusterIdMapForEachPoint computemapping;
                vtkm::cont::Invoker invokecm;
                invokecm(computemapping,cluster_ids,intermediate_unique_cluster_ids,mapped_indices);


                vtkm::cont::ArrayHandle <vtkm::Id> curr_cluster_id_arr;
                vtkm::cont::Invoker invoke1;
                // This functor finds the best neighbor cluster for each data point
                ComputeDistanceForEachPoint computedistance(vtkm::Id3(this->xdim,this->ydim,this->zdim),
                                                         vtkm::Id3(this->blockXSize,this->blockYSize,this->blockZSize),
                                                         this->weight);

                invoke1(computedistance,
                        vtkm::cont::make_ArrayHandleCompositeVector(all_pointsX1,all_pointsY1,all_pointsZ1),
                        init_centersX,
                        init_centersY,
                        init_centersZ,
                        init_vals,
                        intermediate_unique_cluster_ids,
                        dataval_array_normalized,
                        cluster_ids,
                        mapped_indices,
                        curr_cluster_id_arr);

                //Calculate average of the data values according to new cluster ids.
                vtkm::cont::ArrayHandle<vtkm::Id> unique_ids;
                vtkm::cont::ArrayHandle<vtkm::Float64> dataval_avg_clusterwise;
                vtkm::worklet::AverageByKey::Run(curr_cluster_id_arr, dataval_array_normalized, unique_ids, dataval_avg_clusterwise);
                //Calculate average of the xdim of cluster centers according to new cluster ids.
                vtkm::cont::ArrayHandle<vtkm::Id> unique_ids_1;
                vtkm::cont::ArrayHandle<vtkm::Float64> cluster_center_avgX;
                vtkm::worklet::AverageByKey::Run(curr_cluster_id_arr, all_pointsX1, unique_ids_1, cluster_center_avgX);
                //Calculate average of the ydim of cluster centers according to new cluster ids.
                vtkm::cont::ArrayHandle<vtkm::Id> unique_ids_2;
                vtkm::cont::ArrayHandle<vtkm::Float64> cluster_center_avgY;
                vtkm::worklet::AverageByKey::Run(curr_cluster_id_arr, all_pointsY1, unique_ids_2, cluster_center_avgY);
                //Calculate average of the zdim of cluster centers according to new cluster ids.
                vtkm::cont::ArrayHandle<vtkm::Id> unique_ids_3;
                vtkm::cont::ArrayHandle<vtkm::Float64> cluster_center_avgZ;
                vtkm::worklet::AverageByKey::Run(curr_cluster_id_arr, all_pointsZ1, unique_ids_3, cluster_center_avgZ);

                // update cluster id array with new cluster id
                vtkm::cont::ArrayCopy(curr_cluster_id_arr, cluster_ids);

                current_cluster_num = cluster_center_avgX.GetNumberOfValues();
                                
                // Now compute epsilon between previous centers and current centers
                //////////////////////////////////////////////////////////
                vtkm::cont::ArrayHandle <vtkm::Float64> epsilon_vals;
                vtkm::cont::Invoker invoke2;
                ComputeEpsilonForEachCluster computeeps(vtkm::Id3(this->blockXSize,this->blockYSize,this->blockZSize),
                                                    this->weight);

                //If cluster number has reduced, adjust array size, subset the arrays so that one to one
                // lookup is possible to compute epsilon in parallel for each cluster center
                if (current_cluster_num < prev_cluster_num)
                {
                    vtkm::cont::ArrayHandle <vtkm::Id> lookup_indices;
                    vtkm::cont::Algorithm::LowerBounds(intermediate_unique_cluster_ids, unique_ids_1, lookup_indices);

                    auto permutedArrayX = vtkm::cont::make_ArrayHandlePermutation(lookup_indices, init_centersX);
                    auto permutedArrayY = vtkm::cont::make_ArrayHandlePermutation(lookup_indices, init_centersY);
                    auto permutedArrayZ = vtkm::cont::make_ArrayHandlePermutation(lookup_indices, init_centersZ);
                    auto permutedArrayVals = vtkm::cont::make_ArrayHandlePermutation(lookup_indices, init_vals);

                    invoke2(computeeps,
                        permutedArrayX,
                        permutedArrayY,
                        permutedArrayZ,
                        permutedArrayVals,
                        cluster_center_avgX,
                        cluster_center_avgY,
                        cluster_center_avgZ,
                        dataval_avg_clusterwise,
                        epsilon_vals);
                }
                else
                {
                    invoke2(computeeps,
                        init_centersX,
                        init_centersY,
                        init_centersZ,
                        init_vals,
                        cluster_center_avgX,
                        cluster_center_avgY,
                        cluster_center_avgZ,
                        dataval_avg_clusterwise,
                        epsilon_vals);
                }

                //std::cout<<iter<<" curr. cluster num: "<<current_cluster_num<< 
                //" prev. cluster num: "<< prev_cluster_num<<" epsilon: "<<eps<<std::endl;
                
                //Use parallel sum to compute total residual/epsilon
                eps = vtkm::cont::Algorithm::Reduce(epsilon_vals, 0.0);

                // Adjust array size if only the number of cluster has reduced at current iteration
                if (current_cluster_num < prev_cluster_num)
                {
                    init_centersX.Allocate(current_cluster_num, vtkm::CopyFlag::On);
                    init_centersY.Allocate(current_cluster_num, vtkm::CopyFlag::On);
                    init_centersZ.Allocate(current_cluster_num, vtkm::CopyFlag::On);
                    init_vals.Allocate(current_cluster_num, vtkm::CopyFlag::On);
                    intermediate_unique_cluster_ids.Allocate(current_cluster_num, vtkm::CopyFlag::On);
                }

                // Update cluster centers with new centers
                vtkm::cont::ArrayCopy(cluster_center_avgX, init_centersX); //xdim 
                vtkm::cont::ArrayCopy(cluster_center_avgY, init_centersY); //ydim
                vtkm::cont::ArrayCopy(cluster_center_avgZ, init_centersZ); //zdim
                vtkm::cont::ArrayCopy(dataval_avg_clusterwise, init_vals); //dataval
                vtkm::cont::ArrayCopy(unique_ids_1, intermediate_unique_cluster_ids);

                iter++;

            } // end while loop
		
	        std::cout<<"Final Slic itereration num: "<<iter<<" and final epsilon: "<<eps<<std::endl;

            return CreateResult(inDataSet, 
                                cluster_ids, 
                                outFieldName, 
                                fieldMetadata);
            
        }
    }
} // namespace vtkm::filter


#endif // vtk_m_filter_SLIC_hxx
