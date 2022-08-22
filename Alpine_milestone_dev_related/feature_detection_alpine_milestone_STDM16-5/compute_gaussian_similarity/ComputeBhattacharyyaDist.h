#ifndef vtk_m_filter_ComputeBhattacharyyaDist_h
#define vtk_m_filter_ComputeBhattacharyyaDist_h

// This functor computes Bhattacharyya distance between two Gaussians.
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

    VTKM_EXEC_CONT
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

#endif