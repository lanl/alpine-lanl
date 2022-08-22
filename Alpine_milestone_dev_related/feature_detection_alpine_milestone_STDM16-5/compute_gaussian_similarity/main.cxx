#include <vtkm/cont/Initialize.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayPortalToIterators.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/cont/EnvironmentTracker.h>
#include <vtkm/cont/Initialize.h>
#include <vtkm/io/VTKDataSetWriter.h>
#include <vtkm/cont/ArrayHandleTransform.h>
#include <vtkm/io/VTKDataSetReader.h>
#include <vtkm/worklet/AverageByKey.h>

#include "StdevByKey.h"
#include "FieldGaussianSimilarity.h"


///////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    vtkm::cont::Initialize(argc, argv);

    // Read a vtk dataset with vtkm reader
    vtkm::io::VTKDataSetReader reader_vtkm1("../../data/cluster_100.vtk");
    vtkm::cont::DataSet cluster_dataSet = reader_vtkm1.ReadDataSet();
    vtkm::io::VTKDataSetReader reader_vtkm2("../../data/fcc10000.vtk");
    vtkm::cont::DataSet density_dataSet = reader_vtkm2.ReadDataSet();
    std::string fieldname1 = "ClusterIds";
    std::string fieldname2 = "ImageScalars";

    //Create a new data set with fields combined from cluster dataset and density dataset as filter input
    vtkm::cont::DataSet combined_dataset;
    combined_dataset.CopyStructure(density_dataSet);
    combined_dataset.AddField(cluster_dataSet.GetField(0));
    combined_dataset.AddField(density_dataSet.GetField(0));
    // for (vtkm::IdComponent i = 0; i < combined_dataset.GetNumberOfFields(); ++i) {
    //     std::cout << combined_dataset.GetField(i).GetName() << std::endl;
    // }

    // Calling the FieldGaussianSimilarity filter 
    vtkm::Pair<vtkm::Float32,vtkm::Float32> Gauss_dist = vtkm::make_Pair(2.2, 5.0);
    vtkm::filter::FieldGaussianSimilarity gsimilarity;
    gsimilarity.SetActiveField(fieldname1);
    gsimilarity.SetFieldNames(fieldname1,fieldname2);
    gsimilarity.SetFeatureGaussian(Gauss_dist);
    vtkm::cont::DataSet outField = gsimilarity.Execute(combined_dataset);

    //Write output dataset
    vtkm::io::VTKDataSetWriter writer("../../output/out_field_filter.vtk");
    writer.WriteDataSet(outField);

    return 0;
}