/*******************************************************************************
 * This file is part of MT-KaHyPar.
 *
 * Copyright (C) 2020 Lars Gottesbüren <lars.gottesbueren@kit.edu>
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

#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/context_enum_classes.h"

#include "mt-kahypar/datastructures/delta_partitioned_hypergraph.h"
#include "mt-kahypar/parallel/stl/scalable_queue.h"
#include "mt-kahypar/partition/refinement/fm/fm_commons.h"
#include "mt-kahypar/partition/refinement/fm/strategies/gain_cache_strategy.h"
#include "mt-kahypar/partition/refinement/greedy/greedy_shared_data.h"
#include "mt-kahypar/partition/refinement/greedy/kway_greedy.h"
#include "mt-kahypar/partition/refinement/i_refiner.h"

namespace mt_kahypar {

using HypernodeIDMessageMatrix = vec<vec<HypernodeID>>;
class BasicGreedyRefiner final : public IRefiner {

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

public:
  BasicGreedyRefiner(const Hypergraph &hypergraph, const Context &c,
                     const TaskGroupID taskGroupID)
      : initial_num_nodes(hypergraph.initialNumNodes()), context(c),
        taskGroupID(taskGroupID),
        sharedData(hypergraph.initialNumNodes(), context),
        _greedy_shared_data(context.shared_memory.num_threads),
        ets_bgf([&] { return constructKWayGreedySearch(); }) {
    if (context.refinement.greedy.obey_minimal_parallelism) {
      sharedData.finishedTasksLimit = std::min(8UL, context.shared_memory.num_threads);
    }
    if (context.refinement.greedy.sync_with_mq) {
      size_t num_threads = context.shared_memory.num_threads;
      _greedy_shared_data.messages = HypernodeIDMessageMatrix(num_threads * num_threads, vec<HypernodeID>());
      tbb::parallel_for(0UL, num_threads * num_threads, [&](const size_t i){
          _greedy_shared_data.messages[i].reserve(1 << 10);
          });
    }
  }

  bool
  refineImpl(PartitionedHypergraph &phg,
             const parallel::scalable_vector<HypernodeID> &refinement_nodes,
             kahypar::Metrics &metrics, double) final;

  void initializeImpl(PartitionedHypergraph &phg) final;

  void roundInitialization(PartitionedHypergraph &phg);

  void staticAssignment(PartitionedHypergraph &phg);
  void partitionAssignment(PartitionedHypergraph &phg);

  KWayGreedy constructKWayGreedySearch() {
    return KWayGreedy(context, initial_num_nodes, sharedData, _greedy_shared_data);
  }

  void printMemoryConsumption();

private:
  /* TODO: refactor all vars to snake_case and private (_var) <27-10-20,
   * @noahares> */
  bool _is_initialized = false;
  const HypernodeID initial_num_nodes;
  const Context& context;
  const TaskGroupID taskGroupID;
  FMSharedData sharedData;
  GreedySharedData _greedy_shared_data;
  tbb::enumerable_thread_specific<KWayGreedy> ets_bgf;
};

} // namespace mt_kahypar
