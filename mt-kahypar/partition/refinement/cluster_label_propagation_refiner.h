/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
 *
 * KaHyPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaHyPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaHyPar.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#pragma once

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/parallel/stl/scalable_vector.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/refinement/i_refiner.h"

namespace mt_kahypar {
template <typename TypeTraits>
class ClusterLabelPropagationRefinerT final : public IRefinerT<TypeTraits> {
 private:
  using HyperGraph = typename TypeTraits::template PartitionedHyperGraph<>;

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

 public:
  explicit ClusterLabelPropagationRefinerT(HyperGraph&,
                                    const Context& context,
                                    const TaskGroupID task_group_id) :
    _context(context),
    _task_group_id(task_group_id) { }

  ClusterLabelPropagationRefinerT(const ClusterLabelPropagationRefinerT&) = delete;
  ClusterLabelPropagationRefinerT(ClusterLabelPropagationRefinerT&&) = delete;

  ClusterLabelPropagationRefinerT & operator= (const ClusterLabelPropagationRefinerT &) = delete;
  ClusterLabelPropagationRefinerT & operator= (ClusterLabelPropagationRefinerT &&) = delete;

 private:
  bool refineImpl(HyperGraph&,
                  const parallel::scalable_vector<HypernodeID>&,
                  kahypar::Metrics&) override final {

    return false;
  }

  void initializeImpl(HyperGraph&) override final {

  }

  const Context& _context;
  const TaskGroupID _task_group_id;
};

using ClusterLabelPropagationRefiner = ClusterLabelPropagationRefinerT<GlobalTypeTraits>;
}  // namespace kahypar
