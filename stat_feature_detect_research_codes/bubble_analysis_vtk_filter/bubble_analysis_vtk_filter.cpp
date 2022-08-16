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


int main(int argc, char* argv[])
{
  std::string inputfile = "../data/fcc_allpoints.vtp";
  std::string outputfile = "feature_similarity_field.vti";

  vtkSmartPointer<vtkXMLPolyDataReader> reader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
  reader->SetFileName(inputfile.c_str());
  reader->Update();

  MPI_Init(NULL, NULL);
  
  vtkSmartPointer<vtkPFeatureAnalysis> gsimilarity = vtkSmartPointer<vtkPFeatureAnalysis>::New();
  gsimilarity->SetFeatureGaussian(2.5,10.19);
  gsimilarity->SetClusterBlockSize(3,3,3);
  gsimilarity->SetInputConnection(reader->GetOutputPort());
  gsimilarity->Update();

  vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
  writer->SetFileName(outputfile.c_str());
  writer->SetInputConnection(gsimilarity->GetOutputPort());
  writer->Write();

  MPI_Finalize();

  return 0;
}
