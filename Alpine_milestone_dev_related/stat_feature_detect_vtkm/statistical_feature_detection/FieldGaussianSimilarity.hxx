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
		    vtkm::cont::ArrayHandle<vtkm::FloatDefault> density_longlong;
			//vtkm::cont::ArrayHandle<vtkm::Float64> density_shallow;

		    // GetData() returns a VariantArrayHandle that is generic to the data types it contains,
		    // it needs to be cast to (CopyTo) an ArrarHandle of a concrete type.
		    // Ref: Chapter 33 of the user's guide.
		    inDataSet.GetField(this->PrimaryFieldName).GetData().AsArrayHandle(id_shallow);
		    inDataSet.GetField(this->SecondaryFieldName).GetData().AsArrayHandle((density_longlong));

		    auto density_shallow = vtkm::cont::make_ArrayHandleCast<vtkm::FloatDefault>(density_longlong);

		    // 1. Calculate mean of the data values according to the cluster ids.
		    vtkm::cont::ArrayHandle<vtkm::Int32> unique_ids;
		    vtkm::cont::ArrayHandle<vtkm::FloatDefault> density_avg;
		    vtkm::worklet::AverageByKey::Run(id_shallow, density_shallow, unique_ids, density_avg);
            //std::cout << "unique_ids.GetNumberOfValues(): " << unique_ids.GetNumberOfValues() << std::endl;

		    // 2. Calculate stddev of the data values according to the cluster ids.
		    vtkm::cont::ArrayHandle<vtkm::Int32> unique_ids1;
		    vtkm::cont::ArrayHandle<vtkm::FloatDefault> density_stdev;
		    vtkm::worklet::StdevByKey::Run(id_shallow, density_shallow, unique_ids1, density_stdev);
            //std::cout << "unique_ids1.GetNumberOfValues(): " << unique_ids1.GetNumberOfValues() << std::endl;

            ////Print for checking the values
		    // for (vtkm::IdComponent i = 0; i < unique_ids1.GetNumberOfValues(); ++i) {
		    //     std::cout << "cluster id: " << unique_ids1.ReadPortal().Get(i)
		    //               << ", mean: " << density_avg.ReadPortal().Get(i)
		    //               << ", stdev: " << density_stdev.ReadPortal().Get(i)
		    //               << std::endl;
		    // }

		    // Zip the two arrays to they can be passed to the functor
		    auto gaussianArr = vtkm::cont::make_ArrayHandleZip(density_avg, density_stdev);
		    //// Compute Bhattacharya distance for each cluster
		    vtkm::cont::ArrayHandle<vtkm::Float64> bhattacharyyaVals;
		    vtkm::cont::Invoker invoke;
            ComputeBhattacharyyaDistNew compBhattacharyyaDist(this->FeatureMean, this->FeatureStdev);
            invoke(compBhattacharyyaDist,
                    gaussianArr,
                    bhattacharyyaVals);

		    vtkm::cont::ArrayHandle<vtkm::Range> rangeArray = 
        	vtkm::cont::ArrayRangeCompute(bhattacharyyaVals);
        	vtkm::Range componentRange = rangeArray.ReadPortal().Get(0);

		    // Generate the similarity field
		    vtkm::cont::ArrayHandle <vtkm::Id> lookup_indices;
            vtkm::cont::Algorithm::LowerBounds(unique_ids1, id_shallow, lookup_indices);

		    vtkm::cont::ArrayHandle <vtkm::Float64> outField;
        	vtkm::cont::Invoker invoke1;

            GenerateOutSimField genSimField(componentRange.Min,componentRange.Max);
            invoke1(genSimField,
                lookup_indices,
                bhattacharyyaVals,
             	outField);


			return CreateResult(inDataSet, 
				                outField, 
				                outFieldName, 
				                fieldMetadata);
		}
	}
} // namespace vtkm::filter

#endif
