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

#include <mt-kahypar/definitions.h>
#include <mt-kahypar/datastructures/priority_queue.h>
#include <mt-kahypar/partition/context.h>
#include <mt-kahypar/parallel/work_stack.h>

#include "external_tools/kahypar/kahypar/datastructure/fast_reset_flag_array.h"

#include <tbb/parallel_for.h>

namespace mt_kahypar {

using BlockPriorityQueue = ds::ExclusiveHandleHeap< ds::MaxHeap<Gain, PartitionID> >;
using VertexPriorityQueue = ds::MaxHeap<Gain, HypernodeID>;    // these need external handles

struct GlobalMoveTracker {
  vec<Move> moveOrder;
  vec<MoveID> moveOfNode;
  CAtomic<MoveID> runningMoveID;
  MoveID firstMoveID = 1;

  explicit GlobalMoveTracker(size_t numNodes = 0) :
          moveOrder(numNodes),
          moveOfNode(numNodes, 0),
          runningMoveID(1) { }

  // Returns true if stored move IDs should be reset
  bool reset() {
    if (runningMoveID.load() >= std::numeric_limits<MoveID>::max() - moveOrder.size() - 20) {
      tbb::parallel_for(0UL, moveOfNode.size(), [&](size_t i) { moveOfNode[i] = 0; }, tbb::static_partitioner());
      firstMoveID = 1;
      runningMoveID.store(1);
      return true;
    } else {
      firstMoveID = ++runningMoveID;
      return false;
    }
  }

  MoveID insertMove(Move &m) {
    const MoveID move_id = runningMoveID.fetch_add(1, std::memory_order_relaxed);
    assert(move_id - firstMoveID < moveOrder.size());
    moveOrder[move_id - firstMoveID] = m;
    moveOfNode[m.node] = move_id;
    return move_id;
  }

  Move& getMove(MoveID move_id) {
    assert(move_id - firstMoveID < moveOrder.size());
    return moveOrder[move_id - firstMoveID];
  }

  bool isMoveStillValid(MoveID move_id) {
    return getMove(move_id).gain != invalidGain;
  }

  bool isMoveStillValid(const Move& m) const {
    return m.gain != invalidGain;
  }

  void invalidateMove(MoveID move_id) {
    getMove(move_id).gain = invalidGain;
  }

  void invalidateMove(Move& m) {
    m.gain = invalidGain;
  }

  MoveID numPerformedMoves() const {
    return runningMoveID.load(std::memory_order_relaxed) - firstMoveID;
  }

  bool isMoveStale(const MoveID move_id) const {
    return move_id < firstMoveID;
  }
};

struct NodeTracker {
  vec<CAtomic<SearchID>> searchOfNode;

  SearchID deactivatedNodeMarker = 1;
  CAtomic<SearchID> highestActiveSearchID { 1 };

  explicit NodeTracker(size_t numNodes = 0) : searchOfNode(numNodes, CAtomic<SearchID>(0)) { }

  // only the search that owns u is allowed to call this
  void deactivateNode(HypernodeID u, SearchID search_id) {
    assert(searchOfNode[u].load() == search_id);
    unused(search_id);
    searchOfNode[u].store(deactivatedNodeMarker, std::memory_order_acq_rel);
  }

  bool isLocked(HypernodeID u) {
    return searchOfNode[u].load(std::memory_order_relaxed) == deactivatedNodeMarker;
  }

  // should not be called when searches try to claim nodes
  void releaseNode(HypernodeID u) {
    searchOfNode[u].store(0, std::memory_order_relaxed);
  }

  bool isSearchInactive(SearchID search_id) const {
    return search_id < deactivatedNodeMarker;
  }

  bool canNodeStartNewSearch(HypernodeID u) const {
    return isSearchInactive( searchOfNode[u].load(std::memory_order_acq_rel) );
  }

  void requestNewSearches(SearchID max_num_searches) {
    if (highestActiveSearchID.load(std::memory_order_relaxed) >= std::numeric_limits<SearchID>::max() - max_num_searches - 20) {
      tbb::parallel_for(0UL, searchOfNode.size(), [&](const size_t i) {
        searchOfNode[i].store(0, std::memory_order_relaxed);
      });
      highestActiveSearchID.store(0, std::memory_order_relaxed);
    }
    deactivatedNodeMarker = ++highestActiveSearchID;
  }
};


struct FMSharedData {

  // ! Nodes to initialize the localized FM searches with
  WorkContainer<HypernodeID> refinementNodes;

  // ! PQ handles shared by all threads (each vertex is only held by one thread)
  vec<PosT> vertexPQHandles;

  // ! num parts
  PartitionID numParts;

  // ! Stores the sequence of performed moves and assigns IDs to moves that can be used in the global rollback code
  GlobalMoveTracker moveTracker;

  // ! Tracks the current search of a node, and if a node can still be added to an active search
  NodeTracker nodeTracker;

  // ! Indicates whether a node was a seed in a localized search that found no improvement.
  // ! Used to distinguish whether or not the node should be reinserted into the task queue
  // ! (if it was removed but could not be claimed for a search)
  kahypar::ds::FastResetFlagArray<> fruitlessSeed;

  // ! Stores the designated target part of a vertex, i.e. the part with the highest gain to which moving is feasible
  vec<PartitionID> targetPart;

  // ! Stop parallel refinement if finishedTasks > finishedTasksLimit to avoid long-running single searches
  CAtomic<size_t> finishedTasks;
  size_t finishedTasksLimit = std::numeric_limits<size_t>::max();


  FMSharedData(size_t numNodes = 0, PartitionID numParts = 0, size_t numThreads = 0) :
          refinementNodes(), //numNodes, numThreads),
          vertexPQHandles(), //numNodes, invalid_position),
          numParts(numParts),
          moveTracker(), //numNodes),
          nodeTracker(), //numNodes),
          fruitlessSeed(numNodes),
          targetPart()
  {
    finishedTasks.store(0, std::memory_order_relaxed);

    tbb::parallel_invoke([&] {
      moveTracker.moveOrder.resize(numNodes);
    }, [&] {
      moveTracker.moveOfNode.resize(numNodes);
    }, [&] {
      nodeTracker.searchOfNode.resize(numNodes, CAtomic<SearchID>(0));
    }, [&] {
      vertexPQHandles.resize(numNodes, invalid_position);
    }, [&] {
      refinementNodes.tls_queues.resize(numThreads);
      refinementNodes.timestamps.resize(numNodes, 0);
    }, [&] {
      targetPart.resize(numNodes, kInvalidPartition);
    });
  }

  FMSharedData(size_t numNodes, const Context& context) :
    FMSharedData(numNodes, context.partition.k,
      TBBNumaArena::instance().total_number_of_threads())  { }

  std::unordered_map<std::string, size_t> memory_consumption() const {
    std::unordered_map<std::string, size_t> r;
    r["pq handles"] = vertexPQHandles.capacity() * sizeof(PosT);
    r["move tracker"] = moveTracker.moveOrder.capacity() * sizeof(Move)
                        + moveTracker.moveOfNode.capacity() * sizeof(MoveID);
    r["node tracker"] = nodeTracker.searchOfNode.capacity() * sizeof(SearchID);
    r["fruitless seed"] = moveTracker.moveOrder.size() * sizeof(uint16_t);
    r["task queue"] = refinementNodes.memory_consumption();
    return r;
  }

};

struct FMStats {
  size_t retries = 0;
  size_t extractions = 0;
  size_t pushes = 0;
  size_t moves = 0;
  size_t local_reverts = 0;
  size_t task_queue_reinsertions = 0;
  Gain estimated_improvement = 0;


  void clear() {
    retries = 0;
    extractions = 0;
    pushes = 0;
    moves = 0;
    local_reverts = 0;
    task_queue_reinsertions = 0;
    estimated_improvement = 0;
  }

  void merge(FMStats& other) {
    other.retries += retries;
    other.extractions += extractions;
    other.pushes += pushes;
    other.moves += moves;
    other.local_reverts += local_reverts;
    other.task_queue_reinsertions += task_queue_reinsertions;
    other.estimated_improvement += estimated_improvement;
    clear();
  }

  std::string serialize() {
    std::stringstream os;
    os << V(retries) << " " << V(extractions) << " " << V(pushes) << " " << V(moves) << " " << V(local_reverts);
    return os.str();
  }
};

}