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

#include <tbb/parallel_for_each.h>

#include <mt-kahypar/parallel/numa_work_queue.h>
#include <mt-kahypar/partition/context.h>
#include <mt-kahypar/utils/timer.h>

#include <atomic>
#include <external_tools/kahypar/kahypar/partition/metrics.h>

#include "localized_kway_fm_core.h"
#include "global_rollback.h"


namespace mt_kahypar {
namespace refinement {

// TODO try variant in which, a bunch of searches are stored in a PQ, findMoves(..) yields frequently, and then the most promising search is scheduled next

class MultiTryKWayFM {
public:
  MultiTryKWayFM(const Context& context, TaskGroupID taskGroupID, size_t numNodes, size_t numHyperedges) :
          context(context),
          taskGroupID(taskGroupID),
          sharedData(numNodes, context.partition.k, context.shared_memory.num_threads),
          refinementNodes(numNodes),
          globalRollback(numNodes, numHyperedges, context.partition.k),
          ets_fm(context, numNodes, sharedData.vertexPQHandles.data())
  { }

  Gain refine(PartitionedHypergraph& phg, kahypar::Metrics& metrics) {
    LOG << "before FM" << V(metrics::km1(phg));
    Gain improvement = refine(phg);
    LOG << V(metrics.km1) << V(improvement) << V(metrics::km1(phg));
    metrics.km1 -= improvement;
    metrics.imbalance = metrics::imbalance(phg, context);
    assert(metrics.km1 == metrics::km1(phg));
    return improvement;
  }

  Gain refine(PartitionedHypergraph& phg) {
    utils::Timer& timer = utils::Timer::instance();
    timer.start_timer("fm", "FM");
    timer.start_timer("fm_unnecessary_init", "Initialize FM datastructures that could also be updated by LP Refiner ");

    phg.initializeGainInformation();                // initialization only as long as LP refiner does not use these datastructures
    globalRollback.setRemainingOriginalPins(phg);   // initialization only as long as LP refiner does not use these datastructures

    timer.stop_timer("fm_unnecessary_init");
    //sharedData.partition_weight_budgets.initialize(phg, context.partition.max_part_weights);          // only for version with budgets

    Gain overall_improvement = 0;
    for (size_t round = 0; round < context.refinement.fm.multitry_rounds; ++round) {                    // global multi try rounds
      timer.start_timer("collect_border_nodes", "Collect Border Nodes");

      initialize(phg);

      timer.stop_timer("collect_border_nodes");
      timer.start_timer("find_moves", "Find Moves");

      auto task = [&](const int socket, const int socket_local_task_id, const int task_id) {
        unused(socket_local_task_id); unused(task_id);
        HypernodeID u = std::numeric_limits<HypernodeID>::max();
        LocalizedKWayFM& fm = ets_fm.local();
        while (refinementNodes.tryPop(u, socket)) {
          if (sharedData.nodeTracker.canNodeStartNewSearch(u)) {
            fm.findMoves(phg, u, sharedData, ++sharedData.nodeTracker.highestActiveSearchID);
          }
        }
      };
      TBBNumaArena::instance().run_max_concurrency_tasks_on_all_sockets(taskGroupID, task);
      //task(0,0,0);
      refinementNodes.clear();  // calling clear is necessary since tryPop will reduce the size to -(num calling threads)
      LOG << V(round) << "multitry fm" << V(metrics::km1(phg));

      timer.stop_timer("find_moves");
      timer.start_timer("rollback", "Rollback to Best Solution");

      HyperedgeWeight improvement = globalRollback.globalRollbackToBestPrefix(phg, sharedData);
      LOG << V(improvement) << "after rollback" << V(metrics::km1(phg));
      overall_improvement += improvement;

      timer.stop_timer("rollback");

      if (improvement <= 0) {
        break;
      }
    }

    timer.stop_timer("fm");
    // sharedData.partition_weight_budgets.updatePartWeights(phg, context.partition.max_part_weights);  // only for version with budgets
    return overall_improvement;
  }

  void initialize(PartitionedHypergraph& phg) {
    // insert border nodes into work queues
    refinementNodes.clear();
    tbb::parallel_for(HypernodeID(0), phg.initialNumNodes(), [&](const HypernodeID u) {
    //for (NodeID u = 0; u < phg.initialNumNodes(); ++u)
      if (phg.isBorderNode(u)) {
        refinementNodes.push(u, common::get_numa_node_of_vertex(u));
      }
    });

    // requesting new searches activates all nodes by raising the deactivated node marker
    // also clears the array tracking search IDs in case of overflow
    sharedData.nodeTracker.requestNewSearches(static_cast<SearchID>(refinementNodes.unsafe_size()));

    // shuffle work queues if requested
    if (context.refinement.fm.shuffle) {
      refinementNodes.shuffleQueues();
    }

  }
//protected:

  const Context& context;
  const TaskGroupID taskGroupID;
  FMSharedData sharedData;
  NumaWorkQueue<HypernodeID> refinementNodes;
  GlobalRollback globalRollback;
  tbb::enumerable_thread_specific<LocalizedKWayFM> ets_fm;
};

}
}