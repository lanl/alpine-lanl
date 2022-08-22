#ifndef vtk_m_filter_FieldGaussianSimilarity_hxx
#define vtk_m_filter_FieldGaussianSimilarity_hxx

#include "ComputeBhattacharyyaDist.h"

namespace vtkm
{
	namespace filter
	{
		//-----------------------------------------------------------------------------
		template <typename T, typename StorageType, typename DerivedPolicy>
		inline VTKM_CONT vtkm::cont::DataSet FieldGaussianSimilarity::DoExecute(
		const vtkm::cont::DataSet& inDataSet, 
	    const vtkm::cont::ArrayHandle<T, StorageType>& inField,
	    const vtkm::filter::FieldMetadata& fieldMetadata, 
	    const vtkm::filter::PolicyBase<DerivedPolicy>&)
		{
		    // Shallow copy of the fields in the input dataset.
		    vtkm::cont::ArrayHandle<vtkm::Int32> id_shallow;
		    vtkm::cont::ArrayHandle<vtkm::Float64> density_shallow;
		    // GetData() returns a VariantArrayHandle that is generic to the data types it contains,
		    // it needs to be cast to (CopyTo) an ArrarHandle of a concrete type.
		    // Ref: Chapter 33 of the user's guide.
		    inDataSet.GetField(this->PrimaryFieldName).GetData().CopyTo(id_shallow);
		    inDataSet.GetField(this->SecondaryFieldName).GetData().CopyTo((density_shallow));

		    // 1. Calculate mean of the data values according to the cluster ids.
		    vtkm::cont::ArrayHandle<vtkm::Int32> unique_ids;
		    vtkm::cont::ArrayHandle<vtkm::Float64> density_avg;
		    vtkm::worklet::AverageByKey::Run(id_shallow, density_shallow, unique_ids, density_avg);
		    // 2. Calculate stddev of the data values according to the cluster ids.
		    vtkm::cont::ArrayHandle<vtkm::Int32> unique_ids1;
		    vtkm::cont::ArrayHandle<vtkm::Float64> density_stdev;
		    vtkm::worklet::StdevByKey::Run(id_shallow, density_shallow, unique_ids1, density_stdev);
		    
		    //Print for checking the values
		    for (vtkm::IdComponent i = 0; i < unique_ids1.GetNumberOfValues(); ++i) {
		        std::cout << "cluster id: " << unique_ids1.ReadPortal().Get(i)
		                  << ", mean: " << density_avg.ReadPortal().Get(i)
		                  << ", stdev: " << density_stdev.ReadPortal().Get(i)
		                  << std::endl;
		    }

		    // Zip the two arrays to they can be passed to the functor
		    auto gaussianArr = vtkm::cont::make_ArrayHandleZip(density_avg, density_stdev);
		    //// Calling the functor to compute Bhattacharya distance for each cluster
		    auto bhattacharyyaVals = vtkm::cont::make_ArrayHandleTransform(gaussianArr,
		            ComputeBhattacharyyaDist<vtkm::Float32>(vtkm::make_Pair(this->FeatureMean, this->FeatureStdev)));
		    // //Print the values just for testing
		    // for (vtkm::Id index = 0; index < bhattacharyyaVals.GetNumberOfValues(); ++index) {
		    //      std::cout << bhattacharyyaVals.ReadPortal().Get(index) << std::endl;
		    //  }

		    //// Create vtkm data set with similairty values
		    std::vector<vtkm::Float32> simValueBuffer;
		    //populate the buffer: can we do this in parallel
		    for (vtkm::Id index = 0; index < id_shallow.GetNumberOfValues(); ++index) {
		        vtkm::Id cid = id_shallow.ReadPortal().Get(index);
		        simValueBuffer.push_back(bhattacharyyaVals.ReadPortal().Get(cid));
		    }
		    vtkm::cont::ArrayHandle<vtkm::Float32> outField = vtkm::cont::make_ArrayHandle(simValueBuffer);
			
			return CreateResult(inDataSet, outField, outFieldName, fieldMetadata);
		}
	}
} // namespace vtkm::filter


#endif