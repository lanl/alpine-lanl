/*=========================================================================
// Testing vtkPFeatureAnalysis  filter
// Soumya Dutta
// July 2020
=========================================================================*/

#include <iostream>
#include <mpi.h>
//VTK headers
#include <vtkXMLPolyDataReader.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkPFeatureAnalysis.h>

#include <vtkXMLUnstructuredGridReader.h>
#include <vtkDataReader.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>

int main(int argc, char* argv[])
{
  //std::string inputfile = "/Users/sdutta/Desktop/fullmer_test_data/combined.vtu";
  std::string inputfile = "/Users/sdutta/Desktop/boyce_1mm_test/combined_boyce.vtu";
  //std::string inputfile = "../data/fcc_allpoints.vtp";

  std::string outputfile = "feature_similarity_field.vti";

  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName(inputfile.c_str());
  reader->Update();

  vtkSmartPointer<vtkPolyData> pdata = vtkSmartPointer<vtkPolyData>::New();
  pdata->SetPoints(reader->GetOutput()->GetPoints());

  MPI_Init(NULL, NULL);
  
  vtkSmartPointer<vtkPFeatureAnalysis> gsimilarity = vtkSmartPointer<vtkPFeatureAnalysis>::New();
  gsimilarity->SetFeatureGaussian(2,10.19);
  gsimilarity->SetClusterBlockSize(3,3,3);
  gsimilarity->SetInputData(pdata);
  gsimilarity->Update();

  vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
  writer->SetFileName(outputfile.c_str());
  writer->SetInputConnection(gsimilarity->GetOutputPort());
  writer->Write();

  MPI_Finalize();

  return 0;
}
