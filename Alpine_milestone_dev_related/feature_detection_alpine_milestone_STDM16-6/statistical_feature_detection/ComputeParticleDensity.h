//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2014 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
//  Copyright 2014 UT-Battelle, LLC.
//  Copyright 2018 Los Alamos National Security.
//
//  Under the terms of Contract DE-NA0003525 with NTESS,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================

#include <vtkm/cont/BoundsCompute.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/filter/FilterDataSetWithField.h>
#include <vtkm/filter/FilterField.h>
#include <vtkm/filter/CreateResult.h>
#include <vtkm/io/reader/VTKDataSetReader.h>
#include <vtkm/io/writer/VTKDataSetWriter.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/worklet/connectivities/ImageConnectivity.h>

namespace vtkm {
    namespace filter {

        // given a Particle dataset, return their density (count) in a uniform grid
        class ParticleDensity : public vtkm::filter::FilterDataSetWithField<ParticleDensity> {
        public:
            ParticleDensity() {}

            class CountParticlesInCell : public vtkm::worklet::WorkletMapField {
            public:
                using ControlSignature =
                void(FieldIn
                coord,
                AtomicArrayInOut counts
                );
                using ExecutionSignature =
                void(_1, _2
                );
                using InputDomain = _1;

                VTKM_CONT
                CountParticlesInCell(const vtkm::Vec<vtkm::FloatDefault, 3> &_origin,
                                     const vtkm::Vec<vtkm::FloatDefault, 3> &_spacing,
                                     const vtkm::Vec<vtkm::Id, 3> &_cellDimensions)
                        : Origin(_origin), Spacing(_spacing), CellDimensions(_cellDimensions) {
                }

                template<typename AtomicArrayType>
                VTKM_EXEC void operator()(const vtkm::Vec<vtkm::FloatDefault, 3> &coord,
                                          const AtomicArrayType &counts) const {
                    // TODO: pointId calculation will be incorrect if particle coordinate is out side of
                    // the bounding box. AtomicArray::Add does not check on index.
                    vtkm::Vec<vtkm::Id, 3> ijk = (coord - Origin) / Spacing;

                    auto pointId =
                            ijk[0] + ijk[1] * CellDimensions[0] + ijk[2] * CellDimensions[0] * CellDimensions[1];

                    counts.Add(pointId, 1);
                }

            private:
                vtkm::Vec<vtkm::FloatDefault, 3> Origin;
                vtkm::Vec<vtkm::FloatDefault, 3> Spacing;
                vtkm::Vec<vtkm::Id, 3> CellDimensions;
            };

            template<typename ArrayType, typename DerivedPolicy>
            VTKM_CONT vtkm::cont::DataSet DoExecute(const vtkm::cont::DataSet &input,
                                                    const ArrayType &field,
                                                    const vtkm::filter::FieldMetadata &,
                                                    const vtkm::filter::PolicyBase<DerivedPolicy> &) {
                using Algorithm = vtkm::cont::Algorithm;

                // Calculate bounding box of particles.
                vtkm::Bounds bounds = BoundsCompute(input, 0);
                vtkm::Vec<vtkm::Float64, 3> origin{bounds.X.Min, bounds.Y.Min, bounds.Z.Min};
                vtkm::Vec<vtkm::Float64, 3> spacing{
                        (bounds.X.Max - bounds.X.Min) / vtkm::Float64(CellDimensions[0]),
                        (bounds.Y.Max - bounds.Y.Min) / vtkm::Float64(CellDimensions[1]),
                        (bounds.Z.Max - bounds.Z.Min) / vtkm::Float64(CellDimensions[2])
                };

                vtkm::cont::DataSet output;

                vtkm::cont::CellSetStructured<3> cellSet;
                cellSet.SetPointDimensions(PointDimensions);
                output.SetCellSet(cellSet);

                output.AddCoordinateSystem(
                        vtkm::cont::CoordinateSystem("coordinates", cellSet.GetPointDimensions(), origin, spacing));

                vtkm::cont::ArrayHandleConstant<vtkm::Id> zeros(0, cellSet.GetNumberOfPoints());
                vtkm::cont::ArrayHandle<vtkm::Id> counts;
                Algorithm::Copy(zeros, counts);

                CountParticlesInCell countParticlesWorklet(origin, spacing, PointDimensions);
                vtkm::worklet::DispatcherMapField<CountParticlesInCell> dispatchCounting(
                        countParticlesWorklet);
                dispatchCounting.Invoke(field, counts);
                
                //std::cout << "cellDimension: " << cellSet.GetCellDimensions() << std::endl;
                //std::cout << "pointDimension:" << cellSet.GetPointDimensions() << std::endl;
                //std::cout << "counts.GetNumberOfValues(): " << counts.GetNumberOfValues() << std::endl;

                output.AddPointField(std::string(this->outFieldName), counts);

                return output;
            }

            template<typename T, typename StorageType, typename DerivedPolicy>
            VTKM_EXEC bool DoMapField(vtkm::cont::DataSet &,
                                      const vtkm::cont::ArrayHandle<T, StorageType> &,
                                      const vtkm::filter::FieldMetadata &,
                                      const vtkm::filter::PolicyBase<DerivedPolicy> &) {
                // FIXME: this method is not actually called for our case since we don't really have
                // any input field (only coordinate systems)
                return true;
            }

            void SetCellDimensions(const vtkm::Id3 &dimension) {
                CellDimensions = dimension;
                PointDimensions = CellDimensions + vtkm::Id3{ 1, 1, 1 };
            }

            void SetOutFieldName(std::string outFieldName) {
                this->outFieldName = outFieldName;
            }

        private:
            vtkm::Id3 CellDimensions{1, 1, 1};
            vtkm::Id3 PointDimensions = CellDimensions + vtkm::Id3{ 1, 1, 1 };
            std::string outFieldName;
        };


    } // filter
} // vtkm
