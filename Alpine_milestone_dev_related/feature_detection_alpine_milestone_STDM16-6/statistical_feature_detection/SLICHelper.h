//Header for SLIC filter
#ifndef vtk_m_filter_SLIC_Helper_h
#define vtk_m_filter_SLIC_Helper_h

#include <vtkm/filter/FilterField.h>

// This functor initializes cluster center for x,y,z-dims. Given oneD id, it computes initial locations of cluster centers
// There is a bug in this functor in the logic of index calculation
template<typename T>
struct ComputeInitIndex {

    T BlockXSize;
    T Num_xblocks;
    T BlockYSize;
    T Num_yblocks;
    T BlockZSize;
    T Num_zblocks;

    //constructor when two input parameters are passed
    VTKM_EXEC_CONT
    ComputeInitIndex(T blockXSize = T(0), T num_xblocks = T(1),
                     T blockYSize = T(2), T num_yblocks = T(3),
                     T blockZSize = T(4), T num_zblocks = T(5)) 
    : BlockXSize(blockXSize), Num_xblocks(num_xblocks),
      BlockYSize(blockYSize), Num_yblocks(num_yblocks),
      BlockZSize(blockZSize), Num_zblocks(num_zblocks) { }

    VTKM_EXEC
    vtkm::Id3 operator()(vtkm::Id index_val) const
    {
        vtkm::Id x_idx = (index_val%this->Num_xblocks)*this->BlockXSize + this->BlockXSize/2;
        vtkm::Id y_idx = (index_val%this->Num_yblocks)*this->BlockYSize + this->BlockYSize/2;
        vtkm::Id z_idx = (index_val%this->Num_zblocks)*this->BlockZSize + this->BlockZSize/2;
        return vtkm::Id3(x_idx,y_idx,z_idx);
    }
};

// This functor normalizes an array.
template<typename T>
struct NormalizeArray {

    T Minval;
    T Maxval;

    //default constructor
    VTKM_EXEC_CONT
    NormalizeArray() = default;

    //constructor when two input parameters are passed as a pair
    VTKM_EXEC_CONT
    NormalizeArray(vtkm::Pair<vtkm::Float64, vtkm::Float64> range)
            :Minval(range.first), Maxval(range.second) { }

    VTKM_EXEC
    T operator()(vtkm::Float64 current_val ) const
    {
        if (Minval != Maxval)
          return vtkm::Float64((current_val-Minval)/(Maxval-Minval));
        else
          return vtkm::Float64(current_val);
    }
};

struct ComputeAllPointIDs : public vtkm::worklet::WorkletMapField {

    private:
        vtkm::Id xdim;
        vtkm::Id ydim;
        vtkm::Id zdim;

    public:

        using ControlSignature = void(FieldIn, FieldOut, FieldOut, FieldOut);
        using ExecutionSignature = void(_1, _2, _3, _4);
        using InputDomain = _1;

        // constructor
        VTKM_EXEC_CONT
        ComputeAllPointIDs(vtkm::Id3 dims)
        : xdim(dims[0]), ydim(dims[1]), zdim(dims[2]) {}

        template<typename Point,
                 typename OutType1, 
                 typename OutType2,
                 typename OutType3>
        VTKM_EXEC void operator()(const Point& index,
                                  OutType1& out1,
                                  OutType2& out2,
                                  OutType3& out3) const
        {
            vtkm::Id val = index;
            out3 = vtkm::Float64(val/(xdim*ydim));
            val -= (out3 * xdim * ydim);
            out2 = vtkm::Float64(val / xdim);
            out1 = vtkm::Float64(val % xdim);
        }
}; 

struct ComputeInitValues : public vtkm::worklet::WorkletMapField {

    using ControlSignature = void(FieldIn, WholeArrayIn, FieldOut);
    using ExecutionSignature = void(_1, _2, _3);
    using InputDomain = _1;

    template<typename Point,
             typename FieldPortal,
             typename OutType >
    VTKM_EXEC void operator()(const Point& point,
                              const FieldPortal& field,
                               OutType& out) const
    {   
        out = field.Get(point);
    } 
};

template<typename T>
struct ComputeThreeDtoOneDIndex {

    T Xdim;
    T Ydim;
   
    //constructor when two input parameters are passed
    VTKM_EXEC_CONT
    ComputeThreeDtoOneDIndex(T xdim = T(0), T ydim = T(1)) 
    : Xdim(xdim), Ydim(ydim) { }

    VTKM_EXEC
    T operator()(vtkm::Id3 index_val) const
    {
        
        vtkm::Id oneDindex = index_val[0] + this->Xdim*(index_val[1] + this->Ydim*index_val[2]); 
        return oneDindex;
    }
};


struct ComputeClusterIdMapForEachPoint : public vtkm::worklet::WorkletMapField {

    public:
        using ControlSignature = void(FieldIn, WholeArrayIn, FieldOut);
        using ExecutionSignature = void(_1, _2, _3);
        using InputDomain = _1;

        VTKM_EXEC_CONT
        ComputeClusterIdMapForEachPoint() {}

        template<typename Point,
                 typename FieldPortal,
                 typename OutType >
        VTKM_EXEC void operator()(const Point& curr_cid,
                                  const FieldPortal& unique_cids,
                                  OutType& out) const
        {
            //Search the actual look up id that will be looked up during distance computation
            for (vtkm::Id dd = 0; dd < unique_cids.GetNumberOfValues(); ++dd)
            {
              if (unique_cids.Get(dd) == curr_cid)
              {
                out = dd;
                break;
              }
            }
        } 
};

// Parallelize for each point with neighbors only: cluster center locations are passed as 3 1D separate arrays
/*struct ComputeDistanceForEachPoint : public vtkm::worklet::WorkletMapField {

    private:
        vtkm::Id xdim;
        vtkm::Id ydim;
        vtkm::Id zdim;
        vtkm::Float64 weight;
        vtkm::Id blockXSize;
        vtkm::Id blockYSize;
        vtkm::Id blockZSize;

    public:
        using ControlSignature = void(FieldIn, WholeArrayIn, WholeArrayIn, WholeArrayIn, 
                                      WholeArrayIn, WholeArrayIn, WholeArrayIn, WholeArrayIn, FieldOut);
        using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);
        using InputDomain = _1;

        VTKM_EXEC_CONT
        ComputeDistanceForEachPoint(vtkm::Id3 dims, vtkm::Id3 blocksize,  vtkm::Float64 weight)
        : xdim(dims[0]), ydim(dims[1]), zdim(dims[2]),  
          blockXSize(blocksize[0]), blockYSize(blocksize[1]), blockZSize(blocksize[2]), 
          weight(weight) {}

        template<typename Point,
                 typename FieldPortal1,
                 typename FieldPortal2,
                 typename FieldPortal3,
                 typename FieldPortal4,
                 typename FieldPortal5,
                 typename FieldPortal6,
                 typename FieldPortal7,
                 typename OutType >
        VTKM_EXEC void operator()(const Point& point,
                                  const FieldPortal1& center_locsX,
                                  const FieldPortal2& center_locsY,
                                  const FieldPortal3& center_locsZ,
                                  const FieldPortal4& center_vals,
                                  const FieldPortal5& unique_cids,
                                  const FieldPortal6& norm_dataval_arr,
                                  const FieldPortal7& cluster_field_arr,
                                  OutType& out) const
        {
            vtkm::Float64 min_dist = 0; 
            vtkm::Id best_cid;
            vtkm::Float64 val1=0;
            vtkm::Float64 val2=0;
            vtkm::Float64 val=0;
            // vtkm::Id halfblockXSize = blockXSize/2;
            // vtkm::Id halfblockYSize = blockYSize/2;
            // vtkm::Id halfblockZSize = blockZSize/2;
            vtkm::Id halfblockXSize = blockXSize;
            vtkm::Id halfblockYSize = blockYSize;
            vtkm::Id halfblockZSize = blockZSize;
            vtkm::Id oneDindex;
            vtkm::Id3 pt;

            /////////////////////////////////////////////
            /// Find immediate overlapped neighbors
            vtkm::Id num_neighbors = 18;
            std::vector<vtkm::Id> neighbors_cluster_ids;
            neighbors_cluster_ids.resize(num_neighbors);

            //-x,y,z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), point[1], point[2]);
            oneDindex = pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[0] = cluster_field_arr.Get(oneDindex);

            //+x,y,z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), point[1], point[2]);
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[1] = cluster_field_arr.Get(oneDindex);

            //x,-y,z
            pt = vtkm::Id3(point[0], vtkm::Max(point[1] - halfblockYSize,ydim-1), point[2]);
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[2] = cluster_field_arr.Get(oneDindex);

            //x,+y,z
            pt = vtkm::Id3(point[0], vtkm::Min(point[1] + halfblockYSize,ydim-1), point[2]);
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[3] = cluster_field_arr.Get(oneDindex);

            //x,y,-z
            pt = vtkm::Id3(point[0], point[1], vtkm::Max(point[2] - halfblockZSize,zdim-1));
            oneDindex = pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[4] = cluster_field_arr.Get(oneDindex);

            //x,y,+z
            pt = vtkm::Id3(point[0], point[1], vtkm::Min(point[2] + halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[5] = cluster_field_arr.Get(oneDindex);

            //-x,-y,z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), vtkm::Max(point[1] - halfblockYSize,0), point[2]);  
            oneDindex = pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[6] = cluster_field_arr.Get(oneDindex);

            //-x,+y,z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), vtkm::Min(point[1] + halfblockYSize, ydim-1), point[2]);
            oneDindex = pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[7] = cluster_field_arr.Get(oneDindex);

            //+x,-y,z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), vtkm::Max(point[1]-halfblockYSize,0), point[2]);
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[8] = cluster_field_arr.Get(oneDindex);

            //+x,+y,z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), vtkm::Min(point[1]+ halfblockYSize,ydim-1), point[2]);
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[9] = cluster_field_arr.Get(oneDindex);

            //-x,-y,-z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), vtkm::Max(point[1]-halfblockYSize,0), vtkm::Max(point[2] - halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[10] = cluster_field_arr.Get(oneDindex);

            //-x,-y,+z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), vtkm::Max(point[1]-halfblockYSize,0), vtkm::Min(point[2] + halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[11] = cluster_field_arr.Get(oneDindex);

            //-x,+y,-z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), vtkm::Min(point[1]+ halfblockYSize,ydim-1), vtkm::Max(point[2] - halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[12] = cluster_field_arr.Get(oneDindex);

            //-x,+y,+z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), vtkm::Min(point[1]+ halfblockYSize,ydim-1), vtkm::Min(point[2] + halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[13] = cluster_field_arr.Get(oneDindex);

            //+x,-y,-z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), vtkm::Max(point[1]-halfblockYSize,0), vtkm::Max(point[2] - halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[14] = cluster_field_arr.Get(oneDindex);

            //+x,-y,+z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), vtkm::Max(point[1]-halfblockYSize,0), vtkm::Min(point[2] + halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[15] = cluster_field_arr.Get(oneDindex);
            
            //+x,+y,-z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), vtkm::Min(point[1]+ halfblockYSize,ydim-1), vtkm::Max(point[2] - halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[16] = cluster_field_arr.Get(oneDindex);

            //+x,+y,+z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), vtkm::Min(point[1]+ halfblockYSize,ydim-1), vtkm::Min(point[2] + halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[17] = cluster_field_arr.Get(oneDindex);

            ///////////////////////////////////
            //// now compute the best cluster for the current point among all neighbors
            vtkm::Id currentOneDindex =  point[0] + xdim*(point[1] + ydim*point[2]);
            vtkm::Id curr_cluster_id = cluster_field_arr.Get(currentOneDindex);

            // This search should be parallel// TODO
            for (vtkm::Id dd = 0; dd < unique_cids.GetNumberOfValues() ; ++dd)
            {
              if (unique_cids.Get(dd) == curr_cluster_id)
              {
                curr_cluster_id = dd;
                break;
              }
            }

            // Compute the distance from the first cluster first
            val1 = vtkm::Sqrt((point[0] - center_locsX.Get(curr_cluster_id))*(point[0] - center_locsX.Get(curr_cluster_id)) +
                              (point[1] - center_locsY.Get(curr_cluster_id))*(point[1] - center_locsY.Get(curr_cluster_id)) +
                              (point[2] - center_locsZ.Get(curr_cluster_id))*(point[2] - center_locsZ.Get(curr_cluster_id)));
            val1 = val1/(2*blockXSize*blockYSize*blockZSize);
            val2 = vtkm::Sqrt( (norm_dataval_arr.Get(currentOneDindex) - center_vals.Get(curr_cluster_id))
                              *(norm_dataval_arr.Get(currentOneDindex) - center_vals.Get(curr_cluster_id)) );
            min_dist = weight*val1 + (1-weight)*val2;
            best_cid = cluster_field_arr.Get(currentOneDindex);

            //// Consider only the neighboring clusters
            for (vtkm::Id index = 0; index < num_neighbors; ++index)
            {
                vtkm::Float64 min_dist_curr = 0;
                vtkm::Id actualIndex;
                // This search should be parallel// TODO
                for (vtkm::Id dd = 0; dd < unique_cids.GetNumberOfValues() ; ++dd)
                {
                  if (unique_cids.Get(dd) == neighbors_cluster_ids[index])
                  {
                      actualIndex = dd;
                      break;
                  }
                }
               
                // Compute the distance from the first cluster first
                val1 = vtkm::Sqrt((point[0] - center_locsX.Get(actualIndex))
                                 *(point[0] - center_locsX.Get(actualIndex)) +
                                  (point[1] - center_locsY.Get(actualIndex))
                                 *(point[1] - center_locsY.Get(actualIndex)) +
                                  (point[2] - center_locsZ.Get(actualIndex))
                                 *(point[2] - center_locsZ.Get(actualIndex)));

                val1 = val1/(2*blockXSize*blockYSize*blockZSize);
                val2 = vtkm::Sqrt( (norm_dataval_arr.Get(currentOneDindex) - center_vals.Get(actualIndex))
                                  *(norm_dataval_arr.Get(currentOneDindex) - center_vals.Get(actualIndex)));
                min_dist_curr = weight*val1 + (1-weight)*val2;

                if (min_dist_curr < min_dist)
                {
                    min_dist = min_dist_curr;
                    best_cid = neighbors_cluster_ids[index];
                }   
            }

            out = best_cid;
        } 
};*/

// Parallelize for each point with neighbors only: cluster center locations are passed as three 1D separate arrays
struct ComputeDistanceForEachPoint : public vtkm::worklet::WorkletMapField {

    private:
        vtkm::Id xdim;
        vtkm::Id ydim;
        vtkm::Id zdim;
        vtkm::Float64 weight;
        vtkm::Id blockXSize;
        vtkm::Id blockYSize;
        vtkm::Id blockZSize;

    public:
        using ControlSignature = void(FieldIn, WholeArrayIn, WholeArrayIn, WholeArrayIn, 
                                      WholeArrayIn, WholeArrayIn, WholeArrayIn, WholeArrayIn, WholeArrayIn, FieldOut);
        using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10);
        using InputDomain = _1;

        VTKM_EXEC_CONT
        ComputeDistanceForEachPoint(vtkm::Id3 dims, vtkm::Id3 blocksize,  vtkm::Float64 weight)
        : xdim(dims[0]), ydim(dims[1]), zdim(dims[2]),  
          blockXSize(blocksize[0]), blockYSize(blocksize[1]), blockZSize(blocksize[2]), 
          weight(weight) {}

        template<typename Point,
                 typename FieldPortal1,
                 typename FieldPortal2,
                 typename FieldPortal3,
                 typename FieldPortal4,
                 typename FieldPortal5,
                 typename FieldPortal6,
                 typename FieldPortal7,
                 typename FieldPortal8,
                 typename OutType >
        VTKM_EXEC void operator()(const Point& point,
                                  const FieldPortal1& center_locsX,
                                  const FieldPortal2& center_locsY,
                                  const FieldPortal3& center_locsZ,
                                  const FieldPortal4& center_vals,
                                  const FieldPortal5& unique_cids,
                                  const FieldPortal6& norm_dataval_arr,
                                  const FieldPortal7& cluster_field_arr,
                                  const FieldPortal8& mapped_arr,
                                  OutType& out) const
        {
            vtkm::Float64 min_dist = 0; 
            vtkm::Id best_cid;
            vtkm::Float64 val1=0;
            vtkm::Float64 val2=0;
            vtkm::Float64 val=0;
            // vtkm::Id halfblockXSize = blockXSize/2;
            // vtkm::Id halfblockYSize = blockYSize/2;
            // vtkm::Id halfblockZSize = blockZSize/2;
            vtkm::Id halfblockXSize = blockXSize;
            vtkm::Id halfblockYSize = blockYSize;
            vtkm::Id halfblockZSize = blockZSize;
            vtkm::Id oneDindex;
            vtkm::Id3 pt;

            /////////////////////////////////////////////
            /// Find immediate overlapped neighbors
            const vtkm::Id num_neighbors = 18;
            vtkm::Id neighbors_cluster_ids[num_neighbors];
            vtkm::Id mapped_neighbors_cluster_ids[num_neighbors];

            //-x,y,z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), point[1], point[2]);
            oneDindex = pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[0] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[0] = mapped_arr.Get(oneDindex);

            //+x,y,z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), point[1], point[2]);
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[1] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[1] = mapped_arr.Get(oneDindex);

            //x,-y,z
            pt = vtkm::Id3(point[0], vtkm::Max(point[1] - halfblockYSize,ydim-1), point[2]);
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[2] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[2] = mapped_arr.Get(oneDindex);

            //x,+y,z
            pt = vtkm::Id3(point[0], vtkm::Min(point[1] + halfblockYSize,ydim-1), point[2]);
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[3] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[3] = mapped_arr.Get(oneDindex);

            //x,y,-z
            pt = vtkm::Id3(point[0], point[1], vtkm::Max(point[2] - halfblockZSize,zdim-1));
            oneDindex = pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[4] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[4] = mapped_arr.Get(oneDindex);

            //x,y,+z
            pt = vtkm::Id3(point[0], point[1], vtkm::Min(point[2] + halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[5] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[5] = mapped_arr.Get(oneDindex);

            //-x,-y,z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), vtkm::Max(point[1] - halfblockYSize,0), point[2]);  
            oneDindex = pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[6] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[6] = mapped_arr.Get(oneDindex);

            //-x,+y,z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), vtkm::Min(point[1] + halfblockYSize, ydim-1), point[2]);
            oneDindex = pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[7] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[7] = mapped_arr.Get(oneDindex);

            //+x,-y,z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), vtkm::Max(point[1]-halfblockYSize,0), point[2]);
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[8] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[8] = mapped_arr.Get(oneDindex);

            //+x,+y,z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), vtkm::Min(point[1]+ halfblockYSize,ydim-1), point[2]);
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[9] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[9] = mapped_arr.Get(oneDindex);

            //-x,-y,-z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), vtkm::Max(point[1]-halfblockYSize,0), vtkm::Max(point[2] - halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[10] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[10] = mapped_arr.Get(oneDindex);

            //-x,-y,+z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), vtkm::Max(point[1]-halfblockYSize,0), vtkm::Min(point[2] + halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[11] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[11] = mapped_arr.Get(oneDindex);

            //-x,+y,-z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), vtkm::Min(point[1]+ halfblockYSize,ydim-1), vtkm::Max(point[2] - halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[12] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[12] = mapped_arr.Get(oneDindex);

            //-x,+y,+z
            pt = vtkm::Id3(vtkm::Max(point[0] - halfblockXSize,0), vtkm::Min(point[1]+ halfblockYSize,ydim-1), vtkm::Min(point[2] + halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[13] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[13] = mapped_arr.Get(oneDindex);

            //+x,-y,-z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), vtkm::Max(point[1]-halfblockYSize,0), vtkm::Max(point[2] - halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[14] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[14] = mapped_arr.Get(oneDindex);

            //+x,-y,+z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), vtkm::Max(point[1]-halfblockYSize,0), vtkm::Min(point[2] + halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[15] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[15] = mapped_arr.Get(oneDindex);
            
            //+x,+y,-z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), vtkm::Min(point[1]+ halfblockYSize,ydim-1), vtkm::Max(point[2] - halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[16] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[16] = mapped_arr.Get(oneDindex);

            //+x,+y,+z
            pt = vtkm::Id3(vtkm::Min(point[0] + halfblockXSize,xdim-1), vtkm::Min(point[1]+ halfblockYSize,ydim-1), vtkm::Min(point[2] + halfblockZSize,zdim-1));
            oneDindex =  pt[0] + xdim*(pt[1] + ydim*pt[2]);
            neighbors_cluster_ids[17] = cluster_field_arr.Get(oneDindex);
            mapped_neighbors_cluster_ids[17] = mapped_arr.Get(oneDindex);

            ///////////////////////////////////
            //// now compute the best cluster for the current point among all neighbors
            vtkm::Id currentOneDindex =  point[0] + xdim*(point[1] + ydim*point[2]);
            vtkm::Id curr_cluster_id = cluster_field_arr.Get(currentOneDindex);
            vtkm::Id curr_lookup_index = mapped_arr.Get(currentOneDindex);

            // Compute the distance from the first cluster first
            val1 = vtkm::Sqrt((point[0] - center_locsX.Get(curr_lookup_index))*(point[0] - center_locsX.Get(curr_lookup_index)) +
                              (point[1] - center_locsY.Get(curr_lookup_index))*(point[1] - center_locsY.Get(curr_lookup_index)) +
                              (point[2] - center_locsZ.Get(curr_lookup_index))*(point[2] - center_locsZ.Get(curr_lookup_index)));
            val1 = val1/(2*blockXSize*blockYSize*blockZSize);
            val2 = vtkm::Sqrt( (norm_dataval_arr.Get(currentOneDindex) - center_vals.Get(curr_lookup_index))
                              *(norm_dataval_arr.Get(currentOneDindex) - center_vals.Get(curr_lookup_index)) );
            min_dist = weight*val1 + (1-weight)*val2;
            best_cid = curr_cluster_id;

            //// Consider only the neighboring clusters for distance computation
            for (vtkm::Id index = 0; index < num_neighbors; ++index)
            {
                vtkm::Float64 min_dist_curr = 0;
                vtkm::Id actualIndex = mapped_neighbors_cluster_ids[index];
               
                // Compute the distance from the first cluster first
                val1 = vtkm::Sqrt((point[0] - center_locsX.Get(actualIndex))
                                 *(point[0] - center_locsX.Get(actualIndex)) +
                                  (point[1] - center_locsY.Get(actualIndex))
                                 *(point[1] - center_locsY.Get(actualIndex)) +
                                  (point[2] - center_locsZ.Get(actualIndex))
                                 *(point[2] - center_locsZ.Get(actualIndex)));

                val1 = val1/(2*blockXSize*blockYSize*blockZSize);
                val2 = vtkm::Sqrt( (norm_dataval_arr.Get(currentOneDindex) - center_vals.Get(actualIndex))
                                  *(norm_dataval_arr.Get(currentOneDindex) - center_vals.Get(actualIndex)));
                min_dist_curr = weight*val1 + (1-weight)*val2;

                if (min_dist_curr < min_dist)
                {
                    min_dist = min_dist_curr;
                    best_cid = neighbors_cluster_ids[index];
                }   
            }

            //return best matched cluster id
            out = best_cid;
        } 
};

//Compute epsilon for each cluster center at every iteration
struct ComputeEpsilonForEachCluster : public vtkm::worklet::WorkletMapField {

    private:
        vtkm::Float64 weight;
        vtkm::Id blockXSize;
        vtkm::Id blockYSize;
        vtkm::Id blockZSize;

    public:
        using ControlSignature = void(FieldIn, FieldIn, FieldIn, FieldIn, 
                                      FieldIn, FieldIn, FieldIn, FieldIn, FieldOut);
        using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);
        using InputDomain = _1;

        VTKM_EXEC_CONT
        ComputeEpsilonForEachCluster(vtkm::Id3 blocksize,  vtkm::Float64 weight)
        : blockXSize(blocksize[0]), blockYSize(blocksize[1]), blockZSize(blocksize[2]), 
          weight(weight) {}

        template<typename Point1,
                 typename Point2,
                 typename Point3,
                 typename Point4,
                 typename Point5,
                 typename Point6,
                 typename Point7,
                 typename Point8,
                 typename OutType >
        VTKM_EXEC void operator()(const Point1& xco,
                                  const Point2& yco,
                                  const Point3& zco,
                                  const Point4& vo,
                                  const Point5& xcn,
                                  const Point6& ycn,
                                  const Point7& zcn,
                                  const Point8& vn,
                                  OutType& out) const
        {
            // Compute distances for estimating epsilon
            vtkm::Float64 val1 = 0; 
            vtkm::Float64 val2 = 0;
            val1 = vtkm::Sqrt( (xco - xcn)*(xco - xcn) + (yco - ycn)*(yco - ycn) + (zco - zcn)*(zco - zcn) );
            val1 = val1/(2*blockXSize*blockYSize*blockZSize);
            val2 = vtkm::Sqrt( (vo-vn)*(vo-vn) );
            out =  weight*val1 + (1-weight)*val2;
        } 
};

#include "SLIC.hxx"

#endif // vtk_m_filter_SLIC_Helper_h