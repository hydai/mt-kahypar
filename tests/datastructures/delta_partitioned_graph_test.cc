/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2021 Nikolai Maas <nikolai.maas@student.kit.edu>
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

#include <atomic>
#include <mt-kahypar/parallel/tbb_numa_arena.h>

#include "gmock/gmock.h"

#include "tests/datastructures/hypergraph_fixtures.h"
#include "mt-kahypar/definitions.h"
#include "mt-kahypar/datastructures/hypergraph_common.h"
#include "mt-kahypar/datastructures/partitioned_graph.h"
#include "mt-kahypar/datastructures/delta_partitioned_graph.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {

class ADeltaPartitionedGraph : public Test {

 using DeltaPartitionedGraph = ds::DeltaPartitionedGraph<mt_kahypar::PartitionedHypergraph>;

 public:

  ADeltaPartitionedGraph() :
    hg(mt_kahypar::HypergraphFactory::construct(TBBNumaArena::GLOBAL_TASK_GROUP,
      7 , 6, { {1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6} }, nullptr, nullptr, true)),
    phg(3, TBBNumaArena::GLOBAL_TASK_GROUP, hg),
    delta_phg(3) {
    phg.setOnlyNodePart(0, 0);
    phg.setOnlyNodePart(1, 0);
    phg.setOnlyNodePart(2, 0);
    phg.setOnlyNodePart(3, 1);
    phg.setOnlyNodePart(4, 1);
    phg.setOnlyNodePart(5, 2);
    phg.setOnlyNodePart(6, 2);
    phg.initializePartition(TBBNumaArena::GLOBAL_TASK_GROUP);
    phg.initializeGainCache();
    delta_phg.setPartitionedHypergraph(&phg);
  }

  void verifyPinCounts(const HyperedgeID he,
                       const std::vector<HypernodeID>& expected_pin_counts) {
    ASSERT(expected_pin_counts.size() == static_cast<size_t>(phg.k()));
    for (PartitionID block = 0; block < 3; ++block) {
      ASSERT_EQ(expected_pin_counts[block], delta_phg.pinCountInPart(he, block)) << V(he) << V(block);
    }
  }

  void verifyMoveToPenalty(const HypernodeID hn,
                           const std::vector<HyperedgeWeight>& expected_penalties) {
    ASSERT(expected_penalties.size() == static_cast<size_t>(phg.k()));
    for (PartitionID block = 0; block < 3; ++block) {
      ASSERT_EQ(expected_penalties[block], delta_phg.moveToPenalty(hn, block)) << V(hn) << V(block);
    }
  }

  Hypergraph hg;
  mt_kahypar::PartitionedHypergraph phg;
  DeltaPartitionedGraph delta_phg;
};

TEST_F(ADeltaPartitionedGraph, VerifiesInitialPinCounts) {
  // edge 1 - 2
  verifyPinCounts(0, { 2, 0, 0 });
  verifyPinCounts(2, { 2, 0, 0 });
  // edge 1 - 4
  verifyPinCounts(1, { 1, 1, 0 });
  verifyPinCounts(5, { 1, 1, 0 });
  // edge 2 - 3
  verifyPinCounts(3, { 1, 1, 0 });
  verifyPinCounts(4, { 1, 1, 0 });
  // edge 4 - 5
  verifyPinCounts(6, { 0, 1, 1 });
  verifyPinCounts(8, { 0, 1, 1 });
  // edge 4 - 6
  verifyPinCounts(7, { 0, 1, 1 });
  verifyPinCounts(10, { 0, 1, 1 });
  // edge 5 - 6
  verifyPinCounts(9, { 0, 0, 2 });
  verifyPinCounts(11, { 0, 0, 2 });
}

TEST_F(ADeltaPartitionedGraph, VerifyInitialMoveToPenalties) {
  verifyMoveToPenalty(0, { 0, 0, 0 });
  verifyMoveToPenalty(1, { 0, 0, 1 });
  verifyMoveToPenalty(2, { 0, 0, 1 });
  verifyMoveToPenalty(3, { -1, 0, 0 });
  verifyMoveToPenalty(4, { -1, 0, -2 });
  verifyMoveToPenalty(5, { 1, 0, 0 });
  verifyMoveToPenalty(6, { 1, 0, 0 });
}

TEST_F(ADeltaPartitionedGraph, MovesVertices) {
  delta_phg.changeNodePartWithGainCacheUpdate(1, 0, 1, 1000);
  ASSERT_EQ(0, phg.partID(1));
  ASSERT_EQ(1, delta_phg.partID(1));

  // Verify Pin Counts
  verifyPinCounts(0, { 1, 1, 0 });
  verifyPinCounts(2, { 1, 1, 0 });
  verifyPinCounts(1, { 0, 2, 0 });
  verifyPinCounts(5, { 0, 2, 0 });

  // Verify Move To Penalty
  verifyMoveToPenalty(1, { 0, 0, 1 });
  verifyMoveToPenalty(2, { 0, -2, 0 });
  verifyMoveToPenalty(4, { 1, 0, -1 });

  delta_phg.changeNodePartWithGainCacheUpdate(4, 1, 2, 1000);
  ASSERT_EQ(1, phg.partID(4));
  ASSERT_EQ(2, delta_phg.partID(4));

  // Verify Pin Counts
  verifyPinCounts(1, { 0, 1, 1 });
  verifyPinCounts(5, { 0, 1, 1 });
  verifyPinCounts(6, { 0, 0, 2 });
  verifyPinCounts(8, { 0, 0, 2 });
  verifyPinCounts(7, { 0, 0, 2 });
  verifyPinCounts(10, { 0, 0, 2 });

  // Verify Move To Penalty
  verifyMoveToPenalty(4, { 2, 1, 0 });
  verifyMoveToPenalty(1, { -1, 0, -1 });
  verifyMoveToPenalty(5, { 2, 2, 0 });
  verifyMoveToPenalty(6, { 2, 2, 0 });
}

} // namespace ds
} // namespace mt_kahypar