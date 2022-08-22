//Header for Gaussian similarity field filter
#ifndef vtk_m_filter_FieldGaussianSimilarity_h
#define vtk_m_filter_FieldGaussianSimilarity_h

#include <vtkm/filter/FilterField.h>

namespace vtkm
{
    namespace filter
    {
        class FieldGaussianSimilarity : public vtkm::filter::FilterField <FieldGaussianSimilarity> 
        {
            private:
            std::string SecondaryFieldName;
            std::string PrimaryFieldName;
            std::string outFieldName;
            vtkm::Float32 FeatureMean;
            vtkm::Float32 FeatureStdev;

            public:

            //Constructor
            VTKM_CONT 
            FieldGaussianSimilarity() 
            {
                this->outFieldName = "feature_similarity";
            };

            //Method to set feature gaussian parameters
            VTKM_CONT
            void SetFeatureGaussian(vtkm::Pair<vtkm::Float32,vtkm::Float32> GaussDist)
            {
                this->FeatureMean = GaussDist.first;
                this->FeatureStdev = GaussDist.second;
            }

            //Method to set field names
            VTKM_CONT
            void SetFieldNames(std::string FieldName1, std::string FieldName2)
            {
                this->PrimaryFieldName = FieldName1;
                this->SecondaryFieldName = FieldName2;
            }

            // DoExecute function
            template <typename T, typename StorageType, typename DerivedPolicy>
            VTKM_CONT vtkm::cont::DataSet DoExecute(const vtkm::cont::DataSet& inDataSet, 
                                                    const vtkm::cont::ArrayHandle<T, StorageType>& inField,
                                                    const vtkm::filter::FieldMetadata& fieldMetadata, 
                                                    const vtkm::filter::PolicyBase<DerivedPolicy>& policy);
        };
    } 
}

#include "FieldGaussianSimilarity.hxx"

#endif // vtk_m_filter_FieldGaussianSimilarity_h
