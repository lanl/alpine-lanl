//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_worklet_StdevByKey_h
#define vtk_m_worklet_StdevByKey_h

#include <vtkm/VecTraits.h>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/worklet/DescriptiveStatistics.h>
#include <vtkm/worklet/DispatcherReduceByKey.h>
#include <vtkm/worklet/Keys.h>

namespace vtkm
{
namespace worklet
{

struct StdevByKey
{

  struct ExtractKey
  {
    template <typename First, typename Second>
    VTKM_EXEC First operator()(const vtkm::Pair<First, Second>& pair) const
    {
      return pair.first;
    }
  };

  struct ExtractStdev
  {
    template <typename KeyType, typename ValueType>
    VTKM_EXEC ValueType operator()(
      const vtkm::Pair<KeyType, vtkm::worklet::DescriptiveStatistics::StatState<ValueType>>& pair)
      const
    {
      //return pair.second.SampleStddev();
      return pair.second.PopulationStddev();
    }
  };

  /// \brief Compute average values based on an array of keys.
  ///
  /// This method uses an array of keys and an equally sized array of values. The keys in that
  /// array are collected into groups of equal keys, and the values corresponding to those groups
  /// are averaged.
  ///
  /// This method is less sensitive to constructing large groups with the keys than doing the
  /// similar reduction with a \c Keys object. For example, if you have only one key, the reduction
  /// will still be parallel. However, if you need to run the average of different values with the
  /// same keys, you will have many duplicated operations.
  ///
  template <class KeyType,
            class ValueType,
            class KeyInStorage,
            class KeyOutStorage,
            class ValueInStorage,
            class ValueOutStorage>
  VTKM_CONT static void Run(const vtkm::cont::ArrayHandle<KeyType, KeyInStorage>& keyArray,
                            const vtkm::cont::ArrayHandle<ValueType, ValueInStorage>& valueArray,
                            vtkm::cont::ArrayHandle<KeyType, KeyOutStorage>& outputKeyArray,
                            vtkm::cont::ArrayHandle<ValueType, ValueOutStorage>& outputValueArray)
  {
    auto results = vtkm::worklet::DescriptiveStatistics::Run(keyArray, valueArray);

    // Copy/TransformCopy from results to outputKeyArray and outputValueArray
    auto results_key = vtkm::cont::make_ArrayHandleTransform(results, ExtractKey{});
    auto results_stdev = vtkm::cont::make_ArrayHandleTransform(results, ExtractStdev{});

    vtkm::cont::ArrayCopy(results_key, outputKeyArray);
    vtkm::cont::ArrayCopy(results_stdev, outputValueArray);
  }
};
}
} // vtkm::worklet

#endif //vtk_m_worklet_StdevByKey_h
