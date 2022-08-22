// To submit a test case in Crusher with MPI on a compute node: srun -n 4 ./stat_feature_detect

#include "VTKMHeaders.h"

/////////////////////////////////////////////////////////////////////
// Includes new filter and worklets developed for this algorithm
#include "SLIC.h"
#include "StdevByKey.h"
#include "FieldGaussianSimilarity.h"
#include <vtkm/filter/ParticleDensityNearestGridPoint.h>
#include <vtkm/cont/DataSetBuilderExplicit.h>
#include <mpi.h>

///////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size,rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //std::cout<<rank<<" "<<world_size<<std::endl;    

    //Initialize  vtkm
    vtkm::cont::Initialize(argc, argv);

    //// how to query dimenison so we can get xdim,ydim,zdim from data set?
    vtkm::Id xdim=128;
    vtkm::Id ydim=16;
    vtkm::Id zdim=128;
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // This part of the code is only for Offline test, for in situ, the data will come directly at each process.
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    vtkm::cont::Timer timer_IO;
    timer_IO.Start();
    // read particle data
    vtkm::io::VTKDataSetReader reader("../data/fcc_legacy_400.vtk");
    //vtkm::io::VTKDataSetReader reader("/ccs/home/dutta/alpine_proj_space/Crusher/fcc_200mil_26500_1.vtk");
    vtkm::cont::DataSet input = reader.ReadDataSet();

    //Divide data for ech processor to be distributed
    int points_per_process = int(input.GetNumberOfPoints()/(float)world_size);
    
    int startIndex = 0;
    int endIndex = 0;
    if (rank<world_size-1)
    {
        startIndex = rank*points_per_process;
        endIndex = (rank+1)*points_per_process;
    }
    else
    {
        startIndex = rank*points_per_process;
        endIndex = input.GetNumberOfPoints()-1;
    }

    auto inPortal = input.GetCoordinateSystem().GetDataAsMultiplexer().ReadPortal();
    vtkm::cont::ArrayHandle<vtkm::Vec3f> positions;
    positions.Allocate(endIndex-startIndex);

    //populate the local arrayhandle with points
    int index=0;
    for (int i=startIndex; i < endIndex; i++)
    {
        positions.WritePortal().Set(index++,inPortal.Get(i));
    }

    //create a connectivity array
    vtkm::cont::ArrayHandle<vtkm::Id> connectivity;
    vtkm::cont::ArrayCopy(vtkm::cont::make_ArrayHandleIndex(endIndex-startIndex), connectivity);

    //create local data set
    auto data_local = vtkm::cont::DataSetBuilderExplicit::Create(
    positions, vtkm::CellShapeTagVertex{}, 1, connectivity);

    timer_IO.Stop();
    vtkm::Float64 readTime = timer_IO.GetElapsedTime();
    std::cout << "Data loading and preperation time: " << readTime <<std::endl;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // This part of the code is only for Offline test, for in situ, the data will come directly at each process.
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////////
    //1. Compute density field using MPI distributed communication

    vtkm::cont::Timer timer_density;
    timer_density.Start();

    //First compute global particle bounds from local bounds
    vtkm::Bounds local_bounds = data_local.GetCoordinateSystem().GetBounds();
    double min_bounds[3] = {local_bounds.X.Min, local_bounds.Y.Min, local_bounds.Z.Min};
    double max_bounds[3] = {local_bounds.X.Max, local_bounds.Y.Max, local_bounds.Z.Max};
    double global_min_bounds[3],global_max_bounds[3];
    MPI_Allreduce(min_bounds,global_min_bounds,3,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(max_bounds,global_max_bounds,3,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

    //Global bound
    vtkm::Bounds bounds;
    bounds.X.Min = global_min_bounds[0];
    bounds.X.Max = global_max_bounds[0];
    bounds.Y.Min = global_min_bounds[1];
    bounds.Y.Max = global_max_bounds[1];
    bounds.Z.Min = global_min_bounds[2];
    bounds.Z.Max = global_max_bounds[2];


    //Execute density filter
    vtkm::filter::density_estimate::ParticleDensityNearestGridPoint particleDensity{ 
                                                    vtkm::Id3{ xdim, ydim, zdim },
                                                    bounds
                                                    };
    //vtkm::filter::density_estimate::ParticleDensityNearestGridPoint particleDensity;
    particleDensity.SetComputeNumberDensity(true);
    particleDensity.SetDivideByVolume(false);
    //particleDensity.SetBounds(bounds);
    //particleDensity.SetDimension( vtkm::Id3{ xdim, ydim, zdim });

    auto local_density_field = particleDensity.Execute(data_local);

    //get the density field out to an vtkm arrayhandle
    vtkm::cont::ArrayHandle<vtkm::FloatDefault> vtkm_density_arr_local;
    local_density_field.GetCellField("density").GetData().AsArrayHandle<vtkm::FloatDefault>(vtkm_density_arr_local);

    //Get the raw C pointer so we can use MPI
    const float* c_arr_local_density = vtkm::cont::ArrayHandleBasic<vtkm::FloatDefault>(vtkm_density_arr_local).GetReadPointer();
    float *c_arr_global_density;
    c_arr_global_density = (float *)malloc(xdim*ydim*zdim*sizeof(float));
    // Reduce the local field into a global field
    MPI_Reduce(c_arr_local_density,c_arr_global_density,xdim*ydim*zdim,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);

    timer_density.Stop();
    vtkm::Float64 elapsedTime = timer_density.GetElapsedTime();

    double reduction_result = 0.0;
    MPI_Reduce(&elapsedTime, &reduction_result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //std::cout << "At Rank: "<<rank<<" Density field is created and time taken: " << elapsedTime <<std::endl;

    //Rest of the algorithm is run on a single MPI process
    if (rank==0)
    {   
	std::cout <<"Max Density field computation time: " << reduction_result <<std::endl;
        //Parameter:: SLIC cluster size
        vtkm::Id blockXSize=4;
        vtkm::Id blockYSize=16;
        vtkm::Id blockZSize=4;
        //Parameter:: SLIC weight in distance function
        vtkm::Float64 weight=0.1;
        //Parameter:: SLIC halt condition
        vtkm::Float64 halt_cond=0.2;
        //Parameter:: SLIC iteration limit
        vtkm::Id iter_limit=50;
        //Parameter:: Target feature distribution to be searched. 
        vtkm::Pair<vtkm::FloatDefault,vtkm::FloatDefault> feature_Gauss = vtkm::make_Pair(2.5, 10.0);

        //Create a uniform grid dataset from global density C array so it can be passed to SLIC filter.
        vtkm::Vec<vtkm::Float64, 3> origin{bounds.X.Min, bounds.Y.Min, bounds.Z.Min};
        vtkm::Vec<vtkm::Float64, 3> spacing{
                (bounds.X.Max - bounds.X.Min) / vtkm::Float64(xdim-1),
                (bounds.Y.Max - bounds.Y.Min) / vtkm::Float64(ydim-1),
                (bounds.Z.Max - bounds.Z.Min) / vtkm::Float64(zdim-1)};

        auto density_field = vtkm::cont::DataSetBuilderUniform::Create(
        //vtkm::Id3{ xdim, ydim, zdim } + vtkm::Id3{ 1, 1, 1 }, origin, spacing);
        vtkm::Id3{ xdim, ydim, zdim }, origin, spacing);

        //Create array handle from the global density C array pointer
        vtkm::cont::ArrayHandle <vtkm::FloatDefault> density_arr_handle =
        vtkm::cont::make_ArrayHandle(c_arr_global_density, xdim*ydim*zdim, vtkm::CopyFlag::On);

        //Add density arrayhanlde to field
        //density_field.AddField(vtkm::cont::make_FieldCell("density", density_arr_handle));
        density_field.AddField(vtkm::cont::make_FieldPoint("density", density_arr_handle));

        // ///////////////////////////////////////////////////////////////////////
        // //Write density field output
        // std::stringstream ss;
        // ss<<rank;
        // std::string fname = "density_field.vtk";
        // vtkm::io::VTKDataSetWriter writer(fname);
        // writer.WriteDataSet(density_field);
        // ///////////////////////////////////////////////////////////////////////


        // //2. Compute slic
        vtkm::cont::Timer timer_slic;
        timer_slic.Start();

        std::string fieldname2 = "density";

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

        std::string fieldname1 = "ClusterIds";
        
        vtkm::filter::FieldGaussianSimilarity gsimilarity;
        gsimilarity.SetActiveField(fieldname1);
        gsimilarity.SetFieldNames(fieldname1,fieldname2);
        gsimilarity.SetFeatureGaussian(feature_Gauss);
        vtkm::cont::DataSet finalOutField = gsimilarity.Execute(outSlicField);

        timer_sim.Stop();
        vtkm::Float64 elapsedTime2 = timer_sim.GetElapsedTime();
        std::cout << "similarity field is generated and time taken: " << elapsedTime2 <<std::endl;

        std::cout<< "total time: "<<reduction_result + elapsedTime1 + elapsedTime2 <<std::endl;

        // ///////////////////////////////////////////////////////////////////////
        // //Write final output field
        vtkm::io::VTKDataSetWriter writer2("out_sim_field.vtk");
        writer2.WriteDataSet(finalOutField); 
    }

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}
