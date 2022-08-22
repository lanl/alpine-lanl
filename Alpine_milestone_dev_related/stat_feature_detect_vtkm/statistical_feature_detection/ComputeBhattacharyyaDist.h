#ifndef vtk_m_filter_ComputeBhattacharyyaDist_h
#define vtk_m_filter_ComputeBhattacharyyaDist_h

// This functor computes Bhattacharyya distance between two Gaussian distributions.
template<typename T>
struct ComputeBhattacharyyaDist {

    T Mean;
    T Stdev;

    //constructor when two input parameters are passed separately
    VTKM_EXEC_CONT
    ComputeBhattacharyyaDist(T mean = T(0), T stdev = T(1))
            :Mean(mean), Stdev(stdev) { }

    //constructor when two input parameters are passed as a pair
    VTKM_EXEC_CONT
    ComputeBhattacharyyaDist(vtkm::Pair<vtkm::Float32, vtkm::Float32> feature)
            :Mean(feature.first), Stdev(feature.second) { }

    VTKM_EXEC
    T operator()(vtkm::Pair<vtkm::Float32, vtkm::Float32> clusterGauss) const
    {
        vtkm::Float32 bhatta_dist = ((clusterGauss.first-this->Mean)*(clusterGauss.first-this->Mean))
                /(4*(clusterGauss.second+this->Stdev));

        if (clusterGauss.second>0) {
            vtkm::Float32 num = 0.5*(clusterGauss.second+this->Stdev);
            vtkm::Float32 denom = vtkm::Sqrt(clusterGauss.second*this->Stdev);
            vtkm::Float32 val = 0.5*vtkm::Log(num/denom);
            bhatta_dist = bhatta_dist+val;
        }

        return bhatta_dist;
    }
};

// New functor that computes Bhattacharyya distance between two Gaussian distributions. 
struct ComputeBhattacharyyaDistNew : public vtkm::worklet::WorkletMapField {

    private:
        vtkm::Float64 Mean;
        vtkm::Float64 Stdev;

    public:
        using ControlSignature = void(FieldIn, FieldOut);
        using ExecutionSignature = void(_1, _2);
        using InputDomain = _1;

        // constructor
        VTKM_EXEC_CONT
        ComputeBhattacharyyaDistNew(vtkm::Float64 mean, vtkm::Float64 stdev) : Mean(mean), Stdev(stdev) {}

        template<typename Point,
                 typename OutType>
        VTKM_EXEC void operator()(const Point& clusterGauss,
                                  OutType& bhatta_dist) const
        {
            bhatta_dist = ((clusterGauss.first-Mean)*(clusterGauss.first-Mean))
                /(4.0*(clusterGauss.second+Stdev));

            if (clusterGauss.second>0) 
            {
                vtkm::Float64 num = 0.5*(clusterGauss.second+Stdev);
                vtkm::Float64 denom = vtkm::Sqrt(clusterGauss.second*Stdev);
                vtkm::Float64 val = 0.5*vtkm::Log(num/denom);
                bhatta_dist = bhatta_dist+val;
            }

        }
}; 

// Generates output simiarity field by look up
struct GenerateOutSimField : public vtkm::worklet::WorkletMapField {

    private:
        vtkm::Float64 SimFieldMin;
        vtkm::Float64 SimFieldMax;
        
    public:
    using ControlSignature = void(FieldIn, WholeArrayIn, FieldOut);
    using ExecutionSignature = void(_1, _2, _3);
    using InputDomain = _1;

    VTKM_EXEC_CONT
    GenerateOutSimField(vtkm::Float64 min, vtkm::Float64 max)
    : SimFieldMin(min), SimFieldMax(max) {}

    template<typename Point,
             typename FieldPortal,
             typename OutType >
    VTKM_EXEC void operator()(const Point& current_index,
                              const FieldPortal& lookup_vals_array,
                              OutType& out) const
    {
        out = 1.0 - (lookup_vals_array.Get(current_index)-SimFieldMin)/(SimFieldMax-SimFieldMin);
    } 
};

#endif