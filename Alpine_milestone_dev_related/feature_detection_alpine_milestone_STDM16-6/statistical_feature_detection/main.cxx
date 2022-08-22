#include "VTKMHeaders.h"

/////////////////////////////////////////////////////////////////////
// Includes new filter and worklets developed for this algorithm
#include "SLIC.h"
#include "ComputeParticleDensity.h"
#include "StdevByKey.h"
#include "FieldGaussianSimilarity.h"

///////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    vtkm::cont::Initialize(argc, argv);

    //// how to query dimenison so we can get xdim,ydim,zdim from data set?
    vtkm::Id xdim=128;
    vtkm::Id ydim=16;
    vtkm::Id zdim=128;
    vtkm::Id blockXSize=8;
    vtkm::Id blockYSize=8;
    vtkm::Id blockZSize=8;
    vtkm::Float64 weight=0.1;
    vtkm::Float64 halt_cond=0.2;
    vtkm::Id iter_limit=50;
    vtkm::Pair<vtkm::Float32,vtkm::Float32> feature_Gauss = vtkm::make_Pair(2.5, 10.0);
    std::string fieldname1 = "ClusterIds";
    std::string fieldname2 = "density";

    vtkm::cont::Timer timer_IO;
    timer_IO.Start();
    // read particle data
    vtkm::io::VTKDataSetReader reader("../data/fcc_legacy_400.vtk");
    vtkm::cont::DataSet input = reader.ReadDataSet();

    timer_IO.Stop();
    vtkm::Float64 readTime = timer_IO.GetElapsedTime();
    std::cout << "Data is loaded and time taken: " << readTime <<std::endl;

    ///////////////////////////////////////////////////////////////////////
    //1. Compute density field
    vtkm::cont::Timer timer_density;
    timer_density.Start();

    vtkm::filter::ParticleDensity particleDensity;
    particleDensity.SetUseCoordinateSystemAsField(true);
    // TODO: change to SetPointDimension
    particleDensity.SetCellDimensions(vtkm::Id3{ xdim-1, ydim-1, zdim-1 }); // Bins
    particleDensity.SetOutFieldName(fieldname2);
    vtkm::cont::DataSet density_field = particleDensity.Execute(input);

    timer_density.Stop();
    vtkm::Float64 elapsedTime = timer_density.GetElapsedTime();
    std::cout << "Density field is created and time taken: " << elapsedTime <<std::endl;

    // ///////////////////////////////////////////////////////////////////////
    // //Write final output field
    // vtkm::io::VTKDataSetWriter writer("density_field.vtk");
    // writer.WriteDataSet(density_field);
    /////////////////////////////////////////////////////////////////////
    
    // //2. Compute slic
    vtkm::cont::Timer timer_slic;
    timer_slic.Start();

    vtkm::filter::SLIC slic;
    slic.SetFieldDimension(vtkm::Id3(xdim,ydim,zdim));
    slic.SetInitClusterSize(vtkm::Id3(blockXSize,blockYSize,blockZSize));
    slic.SetWeight(weight);
    slic.SetHaltCond(halt_cond);
    slic.SetMaxIter(iter_limit);
    slic.SetSlicFieldName(fieldname2);
    slic.SetActiveField(fieldname2);
    vtkm::cont::DataSet outSlicField = slic.Execute(density_field);

    timer_slic.Stop();
    vtkm::Float64 elapsedTime1 = timer_slic.GetElapsedTime();
    std::cout << "Slic is done and time taken: " << elapsedTime1 <<std::endl;

    // // ///////////////////////////////////////////////////////////////////////
    // // //Write final output field
    // // vtkm::io::VTKDataSetWriter writer1("slic_field.vtk");
    // // writer1.WriteDataSet(outSlicField);


    // ///////////////////////////////////////////////////////////////////////
    // //3. Compute statistical feature similarity field
    vtkm::cont::Timer timer_sim;
    timer_sim.Start();

    vtkm::filter::FieldGaussianSimilarity gsimilarity;
    gsimilarity.SetActiveField(fieldname1);
    gsimilarity.SetFieldNames(fieldname1,fieldname2);
    gsimilarity.SetFeatureGaussian(feature_Gauss);
    vtkm::cont::DataSet finalOutField = gsimilarity.Execute(outSlicField);

    timer_sim.Stop();
    vtkm::Float64 elapsedTime2 = timer_sim.GetElapsedTime();
    std::cout << "similarity field is generated and time taken: " << elapsedTime2 <<std::endl;

    std::cout<< "total time: "<<elapsedTime2 + elapsedTime1 + elapsedTime2 <<std::endl;

    // ///////////////////////////////////////////////////////////////////////
    // //Write final output field
    vtkm::io::VTKDataSetWriter writer2("../output/out_sim_field.vtk");
    writer2.WriteDataSet(finalOutField);

    return 0;
}
