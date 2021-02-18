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

#include <mt-kahypar/partition/context.h>
#include <mt-kahypar/partition/metrics.h>

#include "mt-kahypar/datastructures/delta_partitioned_hypergraph.h"
#include "mt-kahypar/datastructures/sparse_map.h"
#include "mt-kahypar/partition/refinement/fm/fm_commons.h"
#include "mt-kahypar/partition/refinement/fm/stop_rule.h"

namespace mt_kahypar {


template<typename FMStrategy>
class LocalizedKWayFM {
public:
  explicit LocalizedKWayFM(const Context& context, HypernodeID numNodes, FMSharedData& sharedData) :
          context(context),
          thisSearch(0),
          k(context.partition.k),
          deltaPhg(context.partition.k),
          neighborDeduplicator(numNodes, 0),
          fm_strategy(context, numNodes, sharedData, runStats),
          sharedData(sharedData)
          { }


  bool findMoves(PartitionedHypergraph& phg, size_t taskID, size_t numSeeds);

  void memoryConsumption(utils::MemoryTreeNode* parent) const ;

  void resetSearchID() {
    thisSearch = 0;
  }

  FMStats stats;

private:

  // ! Performs localized FM local search on the delta partitioned hypergraph.
  // ! Moves made by this search are not immediately visible to other concurrent local searches.
  // ! The best prefix of moves is applied to the global partitioned hypergraph after the search finishes.
  //void internalFindMovesOnDeltaHypergraph(PartitionedHypergraph& phg, FMSharedData& sharedData);


  template<bool use_delta>
  void internalFindMoves(PartitionedHypergraph& phg);

  template<typename PHG>
  MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE
  void acquireOrUpdateNeighbors(PHG& phg, const Move& move);


  // ! Makes moves applied on delta hypergraph visible on the global partitioned hypergraph.
  std::pair<Gain, size_t> applyMovesOnGlobalHypergraph(PartitionedHypergraph& phg,
                                                       size_t bestGainIndex,
                                                       Gain bestEstimatedImprovement);

  // ! Rollback to the best improvement found during local search in case we applied moves
  // ! directly on the global partitioned hypergraph.
  void revertToBestLocalPrefix(PartitionedHypergraph& phg, size_t bestGainIndex);

  void updateExpensiveMoveRevertCounter(size_t bestGainIndex);

  template<typename PHG>
  MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE
  void syncMessageQueues(PHG& phg);

  MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE
  bool moveForbidden(PartitionedHypergraph& phg, Move& move);

  void updateNeighborDeduplicator() {
    if (++deduplicationTime == 0) {
      neighborDeduplicator.assign(neighborDeduplicator.size(), 0);
      deduplicationTime = 1;
    }
  }

  void clearMessageQueues() {
    SearchID this_index = thisSearch - sharedData.nodeTracker.deactivatedNodeMarker - 1;
    size_t num_threads = context.shared_memory.num_threads;
    size_t mq_begin = this_index * num_threads;
    size_t mq_end = mq_begin + num_threads;
    for (size_t i = mq_begin; i < mq_end; ++i) {
      while(!sharedData.messages[i].try_clear()) {};
    }
  }

 private:

  const Context& context;

  // ! Unique search id associated with the current local search
  SearchID thisSearch;

  // ! Number of blocks
  PartitionID k;

  // ! Local data members required for one localized search run
  //FMLocalData localData;
  vec< std::pair<Move, MoveID> > localMoves;

  // ! Wrapper around the global partitioned hypergraph, that allows
  // ! to perform moves non-visible for other local searches
  ds::DeltaPartitionedHypergraph<PartitionedHypergraph> deltaPhg;

  // ! Used after a move. Stores whether a neighbor of the just moved vertex has already been updated.
  vec<HypernodeID> neighborDeduplicator;
  HypernodeID deduplicationTime = 0;

  // ! Stores hyperedges whose pins's gains may have changed after vertex move
  vec<HyperedgeID> edgesWithGainChanges;

  FMStats runStats;

  FMStrategy fm_strategy;

  FMSharedData& sharedData;

  size_t _local_moves_since_sync = 0;

  vec<HyperedgeID> touched_edges;
  vec<size_t> move_edges_begin;

  struct DelayedGainUpdate {
    HypernodeID node;
    HyperedgeID edge;
    HypernodeID pin_count_in_from_part_after;
    HypernodeID pin_count_in_to_part_after;
  };

  vec<DelayedGainUpdate> delayed_gain_updates;
};

}
