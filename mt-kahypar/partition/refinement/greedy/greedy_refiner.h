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

#include "mt-kahypar/datastructures/delta_partitioned_hypergraph.h"
#include "mt-kahypar/partition/refinement/fm/fm_commons.h"
#include "mt-kahypar/partition/refinement/fm/strategies/gain_cache_strategy.h"
#include "mt-kahypar/partition/refinement/i_refiner.h"

namespace mt_kahypar {

class BasicGreedyRefiner final : public IRefiner {

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

public:
  BasicGreedyRefiner(const Hypergraph &hypergraph, const Context &c,
                     const TaskGroupID taskGroupID)
      : context(c), taskGroupID(taskGroupID), thisSearch(0), k(c.partition.k),
        deltaPhg(context.partition.k),
        neighborDeduplicator(hypergraph.initialNumNodes(), 0),
        sharedData(hypergraph.initialNumNodes(), c),
        fm_strategy(context, hypergraph.initialNumNodes(), sharedData,
                    runStats) {}

  bool
  refineImpl(PartitionedHypergraph &phg,
             const parallel::scalable_vector<HypernodeID> &refinement_nodes,
             kahypar::Metrics &metrics, double) final;

  void initializeImpl(PartitionedHypergraph &phg) final;

  bool findMoves(PartitionedHypergraph &phg, size_t taskID, size_t numSeeds);

  void memoryConsumption(utils::MemoryTreeNode *parent) const;

  FMStats stats;

private:
  // ! Performs localized FM local search on the delta partitioned hypergraph.
  // ! Moves made by this search are not immediately visible to other concurrent
  // local searches. ! The best prefix of moves is applied to the global
  // partitioned hypergraph after the search finishes.
  // void internalFindMovesOnDeltaHypergraph(PartitionedHypergraph& phg,
  // FMSharedData& sharedData);

  template <bool use_delta> void internalFindMoves(PartitionedHypergraph &phg);

  template <typename PHG>
  MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE void updateNeighbors(PHG &phg,
                                                          const Move &move);

  // ! Makes moves applied on delta hypergraph visible on the global partitioned
  // hypergraph.
  //  std::pair<Gain, size_t>
  //  applyMovesOnGlobalHypergraph(PartitionedHypergraph& phg,
  //                                                       size_t bestGainIndex,
  //                                                       Gain
  //                                                       bestEstimatedImprovement);

  // ! Rollback to the best improvement found during local search in case we
  // applied moves ! directly on the global partitioned hypergraph.
  //  void revertToBestLocalPrefix(PartitionedHypergraph &phg,
  //                               size_t bestGainIndex);

private:
  const Context &context;

  const TaskGroupID taskGroupID;

  // ! Unique search id associated with the current local search
  SearchID thisSearch;

  // ! Number of blocks
  PartitionID k;

  // ! Local data members required for one localized search run
  // FMLocalData localData;
  vec<std::pair<Move, MoveID>> localMoves;

  // ! Wrapper around the global partitioned hypergraph, that allows
  // ! to perform moves non-visible for other local searches
  ds::DeltaPartitionedHypergraph<PartitionedHypergraph> deltaPhg;

  // ! Used after a move. Stores whether a neighbor of the just moved vertex has
  // already been updated.
  vec<HypernodeID> neighborDeduplicator;
  HypernodeID deduplicationTime = 0;

  // ! Stores hyperedges whose pins's gains may have changed after vertex move
  vec<HyperedgeID> edgesWithGainChanges;

  FMStats runStats;

  FMSharedData sharedData;

  GainCacheStrategy fm_strategy;
};

} // namespace mt_kahypar
