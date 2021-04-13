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
#include "mt-kahypar/datastructures/static_graph.h"
#include "mt-kahypar/datastructures/static_graph_factory.h"
#include "mt-kahypar/datastructures/partitioned_graph.h"
#include "mt-kahypar/partition/metrics.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {

template< typename PartitionedHG,
          typename HG,
          typename HGFactory>
struct PartitionedGraphTypeTraits {
  using PartitionedGraph = PartitionedHG;
  using Hypergraph = HG;
  using Factory = HGFactory;
};

template<typename TypeTraits>
class APartitionedGraph : public Test {

 using PartitionedGraph = typename TypeTraits::PartitionedGraph;
 using Factory = typename TypeTraits::Factory;

 public:
 using Hypergraph = typename TypeTraits::Hypergraph;

  APartitionedGraph() :
    hypergraph(Factory::construct(TBBNumaArena::GLOBAL_TASK_GROUP,
      7 , 6, { {1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6} }, nullptr, nullptr, true)),
    partitioned_hypergraph(3, TBBNumaArena::GLOBAL_TASK_GROUP, hypergraph) {
    initializePartition();
  }

  void initializePartition() {
    if ( hypergraph.nodeIsEnabled(0) ) partitioned_hypergraph.setNodePart(0, 0);
    if ( hypergraph.nodeIsEnabled(1) ) partitioned_hypergraph.setNodePart(1, 0);
    if ( hypergraph.nodeIsEnabled(2) ) partitioned_hypergraph.setNodePart(2, 0);
    if ( hypergraph.nodeIsEnabled(3) ) partitioned_hypergraph.setNodePart(3, 1);
    if ( hypergraph.nodeIsEnabled(4) ) partitioned_hypergraph.setNodePart(4, 1);
    if ( hypergraph.nodeIsEnabled(5) ) partitioned_hypergraph.setNodePart(5, 2);
    if ( hypergraph.nodeIsEnabled(6) ) partitioned_hypergraph.setNodePart(6, 2);
  }

  void verifyPartitionPinCountsAndConnectivity(const HyperedgeID he,
                                               const std::vector<HypernodeID>& expected_pin_counts) {
    ASSERT(expected_pin_counts.size() == static_cast<size_t>(partitioned_hypergraph.k()));
    for (PartitionID block = 0; block < 3; ++block) {
      const HypernodeID pin_count = partitioned_hypergraph.pinCountInPart(he, block);
      ASSERT_EQ(expected_pin_counts[block], pin_count) << V(he) << V(block);
    }
    HypernodeID expected_connectivity = 0;
    for (const PartitionID& block : partitioned_hypergraph.connectivitySet(he)) {
      ASSERT_TRUE(expected_pin_counts[block] > 0) << V(he) << V(block);
      expected_connectivity++;
    }
    ASSERT_EQ(expected_connectivity, partitioned_hypergraph.connectivity(he));
  }

  void verifyGains(const HypernodeID node, const std::vector<HyperedgeWeight>& expected_gains) {
    ASSERT(expected_gains.size() == static_cast<size_t>(partitioned_hypergraph.k()));
    const PartitionID part_id = partitioned_hypergraph.partID(node);
    for (PartitionID block = 0; block < 3; ++block) {
      if (block != part_id) {
        ASSERT_EQ(expected_gains[block], partitioned_hypergraph.km1Gain(node, part_id, block)) << V(node) << V(block);
      }
    }
  }

  void verifyPins(const Hypergraph& hg,
                  const std::vector<HyperedgeID> hyperedges,
                  const std::vector< std::set<HypernodeID> >& references,
                  bool log = false) {
    ASSERT(hyperedges.size() == references.size());
    for (size_t i = 0; i < hyperedges.size(); ++i) {
      const HyperedgeID he = hyperedges[i];
      const std::set<HypernodeID>& reference = references[i];
      size_t count = 0;
      for (const HypernodeID& pin : hg.pins(he)) {
        if (log) LOG << V(he) << V(pin);
        ASSERT_TRUE(reference.find(pin) != reference.end()) << V(he) << V(pin);
        count++;
      }
      ASSERT_EQ(count, reference.size());
    }
  }

  HyperedgeWeight compute_km1() {
    HyperedgeWeight km1 = 0;
    for (const HyperedgeID& he : partitioned_hypergraph.edges()) {
      km1 += std::max(partitioned_hypergraph.connectivity(he) - 1, 0) * partitioned_hypergraph.edgeWeight(he);
    }
    return km1;
  }

  void verifyAllKm1GainValues() {
    for ( const HypernodeID hn : hypergraph.nodes() ) {
      const PartitionID from = partitioned_hypergraph.partID(hn);
      for ( PartitionID to = 0; to < partitioned_hypergraph.k(); ++to ) {
        if ( from != to ) {
          const HyperedgeWeight km1_before = compute_km1();
          const HyperedgeWeight km1_gain = partitioned_hypergraph.km1Gain(hn, from, to);
          partitioned_hypergraph.changeNodePart(hn, from, to);
          const HyperedgeWeight km1_after = compute_km1();
          ASSERT_EQ(km1_gain, km1_before - km1_after);
          partitioned_hypergraph.changeNodePart(hn, to, from);
        }
      }
    }
  }

  Hypergraph hypergraph;
  PartitionedGraph partitioned_hypergraph;
};

template <class F1, class F2>
void executeConcurrent(const F1& f1, const F2& f2) {
  std::atomic<int> cnt(0);
  tbb::parallel_invoke([&] {
    cnt++;
    while (cnt < 2) { }
    f1();
  }, [&] {
    cnt++;
    while (cnt < 2) { }
    f2();
  });
}

using PartitionedGraphTestTypes =
  ::testing::Types<
          PartitionedGraphTypeTraits<
                          PartitionedGraph<StaticGraph, StaticGraphFactory>,
                          StaticGraph,
                          StaticGraphFactory>>;

TYPED_TEST_CASE(APartitionedGraph, PartitionedGraphTestTypes);

TYPED_TEST(APartitionedGraph, HasCorrectPartWeightAndSizes) {
  ASSERT_EQ(3, this->partitioned_hypergraph.partWeight(0));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(1));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(2));
}

TYPED_TEST(APartitionedGraph, HasCorrectPartWeightsIfOnlyOneThreadPerformsModifications) {
  ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(0, 0, 1));

  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(0));
  ASSERT_EQ(3, this->partitioned_hypergraph.partWeight(1));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(2));
}

TYPED_TEST(APartitionedGraph, PerformsConcurrentMovesWhereAllSucceed) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(0, 0, 1));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(3, 1, 2));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(2, 0, 2));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(5, 2, 1));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(6, 2, 0));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(4, 1, 2));
  });

  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(0));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(1));
  ASSERT_EQ(3, this->partitioned_hypergraph.partWeight(2));
}



TYPED_TEST(APartitionedGraph, HasCorrectInitialPartitionPinCounts) {
  // edge 1 - 2
  this->verifyPartitionPinCountsAndConnectivity(0, { 2, 0, 0 });
  this->verifyPartitionPinCountsAndConnectivity(2, { 2, 0, 0 });
  // edge 1 - 4
  this->verifyPartitionPinCountsAndConnectivity(1, { 1, 1, 0 });
  this->verifyPartitionPinCountsAndConnectivity(5, { 1, 1, 0 });
  // edge 2 - 3
  this->verifyPartitionPinCountsAndConnectivity(3, { 1, 1, 0 });
  this->verifyPartitionPinCountsAndConnectivity(4, { 1, 1, 0 });
  // edge 4 - 5
  this->verifyPartitionPinCountsAndConnectivity(6, { 0, 1, 1 });
  this->verifyPartitionPinCountsAndConnectivity(8, { 0, 1, 1 });
  // edge 4 - 6
  this->verifyPartitionPinCountsAndConnectivity(7, { 0, 1, 1 });
  this->verifyPartitionPinCountsAndConnectivity(10, { 0, 1, 1 });
  // edge 5 - 6
  this->verifyPartitionPinCountsAndConnectivity(9, { 0, 0, 2 });
  this->verifyPartitionPinCountsAndConnectivity(11, { 0, 0, 2 });
}

TYPED_TEST(APartitionedGraph, HasCorrectPartitionPinCountsIfTwoNodesMovesConcurrent) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(2, 0, 1));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(1, 0, 2));
  });

  // edge 1 - 2
  this->verifyPartitionPinCountsAndConnectivity(0, { 0, 1, 1 });
  this->verifyPartitionPinCountsAndConnectivity(2, { 0, 1, 1 });
  // edge 1 - 4
  this->verifyPartitionPinCountsAndConnectivity(1, { 0, 1, 1 });
  this->verifyPartitionPinCountsAndConnectivity(5, { 0, 1, 1 });
  // edge 2 - 3
  this->verifyPartitionPinCountsAndConnectivity(3, { 0, 2, 0 });
  this->verifyPartitionPinCountsAndConnectivity(4, { 0, 2, 0 });
}

TYPED_TEST(APartitionedGraph, HasCorrectPartitionPinCountsIfAllNodesMovesConcurrent) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(0, 0, 1));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(4, 1, 2));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(6, 2, 1));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(1, 0, 2));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(3, 1, 0));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(5, 2, 1));
  });

  // edge 1 - 2
  this->verifyPartitionPinCountsAndConnectivity(0, { 1, 0, 1 });
  this->verifyPartitionPinCountsAndConnectivity(2, { 1, 0, 1 });
  // edge 1 - 4
  this->verifyPartitionPinCountsAndConnectivity(1, { 0, 0, 2 });
  this->verifyPartitionPinCountsAndConnectivity(5, { 0, 0, 2 });
  // edge 2 - 3
  this->verifyPartitionPinCountsAndConnectivity(3, { 2, 0, 0 });
  this->verifyPartitionPinCountsAndConnectivity(4, { 2, 0, 0 });
  // edge 4 - 5
  this->verifyPartitionPinCountsAndConnectivity(6, { 0, 1, 1 });
  this->verifyPartitionPinCountsAndConnectivity(8, { 0, 1, 1 });
  // edge 4 - 6
  this->verifyPartitionPinCountsAndConnectivity(7, { 0, 1, 1 });
  this->verifyPartitionPinCountsAndConnectivity(10, { 0, 1, 1 });
  // edge 5 - 6
  this->verifyPartitionPinCountsAndConnectivity(9, { 0, 2, 0 });
  this->verifyPartitionPinCountsAndConnectivity(11, { 0, 2, 0 });
}

TYPED_TEST(APartitionedGraph, HasCorrectInitialBorderNodes) {
  ASSERT_FALSE(this->partitioned_hypergraph.isBorderNode(0));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(1));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(2));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(3));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(4));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(5));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(6));

  ASSERT_EQ(0, this->partitioned_hypergraph.numIncidentCutHyperedges(0));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(1));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(2));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(3));
  ASSERT_EQ(3, this->partitioned_hypergraph.numIncidentCutHyperedges(4));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(5));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(6));
}


// TYPED_TEST(APartitionedGraph, ExtractBlockZeroWithCutNetSplitting) {
//   auto extracted_hg = this->partitioned_hypergraph.extract(TBBNumaArena::GLOBAL_TASK_GROUP, 0, true);
//   auto& hg = extracted_hg.first;
//   auto& hn_mapping = extracted_hg.second;

//   ASSERT_EQ(3, hg.initialNumNodes());
//   ASSERT_EQ(2, hg.initialNumEdges());
//   ASSERT_EQ(2, hg.initialNumPins());
//   ASSERT_EQ(2, hg.maxEdgeSize());

//   auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
//     return hn_mapping[hn];
//   };
//   parallel::scalable_vector<HypernodeID> node_id = {
//     map_from_original_to_extracted_hg(0),
//     map_from_original_to_extracted_hg(1),
//     map_from_original_to_extracted_hg(2)
//   };

//   this->verifyPins(hg, {0, 1},
//     { {node_id[0], node_id[2]}, {node_id[0], node_id[1]} });
// }


// TYPED_TEST(APartitionedGraph, ExtractBlockOneWithCutNetSplitting) {
//   auto extracted_hg = this->partitioned_hypergraph.extract(TBBNumaArena::GLOBAL_TASK_GROUP, 1, true);
//   auto& hg = extracted_hg.first;
//   auto& hn_mapping = extracted_hg.second;

//   ASSERT_EQ(2, hg.initialNumNodes());
//   ASSERT_EQ(0, hg.initialNumEdges());
//   ASSERT_EQ(0, hg.initialNumPins());

//   auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
//     return hn_mapping[hn];
//   };
//   parallel::scalable_vector<HypernodeID> node_id = {
//     map_from_original_to_extracted_hg(3),
//     map_from_original_to_extracted_hg(4)
//   };

//   this->verifyPins(hg, {0, 1},
//     { {node_id[0], node_id[1]}, {node_id[0], node_id[1]} });
// }

// TYPED_TEST(APartitionedGraph, ExtractBlockTwoWithCutNetSplitting) {
//   auto extracted_hg = this->partitioned_hypergraph.extract(TBBNumaArena::GLOBAL_TASK_GROUP, 2, true);
//   auto& hg = extracted_hg.first;
//   auto& hn_mapping = extracted_hg.second;

//   ASSERT_EQ(2, hg.initialNumNodes());
//   ASSERT_EQ(2, hg.initialNumEdges());
//   ASSERT_EQ(2, hg.initialNumPins());
//   ASSERT_EQ(2, hg.maxEdgeSize());

//   auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
//     return hn_mapping[hn];
//   };
//   parallel::scalable_vector<HypernodeID> node_id = {
//     map_from_original_to_extracted_hg(5),
//     map_from_original_to_extracted_hg(6)
//   };

//   this->verifyPins(hg, {0},
//     { {node_id[0], node_id[1]} });
// }

// TYPED_TEST(APartitionedGraph, ExtractBlockZeroWithCutNetRemoval) {
//   auto extracted_hg = this->partitioned_hypergraph.extract(TBBNumaArena::GLOBAL_TASK_GROUP, 0, false);
//   auto& hg = extracted_hg.first;
//   auto& hn_mapping = extracted_hg.second;

//   ASSERT_EQ(3, hg.initialNumNodes());
//   ASSERT_EQ(1, hg.initialNumEdges());
//   ASSERT_EQ(2, hg.initialNumPins());
//   ASSERT_EQ(2, hg.maxEdgeSize());

//   auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
//     return hn_mapping[hn];
//   };
//   parallel::scalable_vector<HypernodeID> node_id = {
//     map_from_original_to_extracted_hg(0),
//     map_from_original_to_extracted_hg(1),
//     map_from_original_to_extracted_hg(2)
//   };
//   parallel::scalable_vector<HypernodeID> edge_id = { 0 };

//   this->verifyPins(hg, {0},
//     { {node_id[0], node_id[2]} });
// }

// TYPED_TEST(APartitionedGraph, ExtractBlockOneWithCutNetRemoval) {
//   this->partitioned_hypergraph.changeNodePart(6, 2, 1);
//   auto extracted_hg = this->partitioned_hypergraph.extract(TBBNumaArena::GLOBAL_TASK_GROUP, 1, false);
//   auto& hg = extracted_hg.first;
//   auto& hn_mapping = extracted_hg.second;

//   ASSERT_EQ(3, hg.initialNumNodes());
//   ASSERT_EQ(1, hg.initialNumEdges());
//   ASSERT_EQ(3, hg.initialNumPins());
//   ASSERT_EQ(3, hg.maxEdgeSize());

//   auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
//     return hn_mapping[hn];
//   };
//   parallel::scalable_vector<HypernodeID> node_id = {
//     map_from_original_to_extracted_hg(3),
//     map_from_original_to_extracted_hg(4),
//     map_from_original_to_extracted_hg(6)
//   };

//   this->verifyPins(hg, {0},
//     { {node_id[0], node_id[1], node_id[2]} });
// }

// TYPED_TEST(APartitionedGraph, ExtractBlockTwoWithCutNetRemoval) {
//   this->partitioned_hypergraph.changeNodePart(2, 0, 2);
//   auto extracted_hg = this->partitioned_hypergraph.extract(TBBNumaArena::GLOBAL_TASK_GROUP, 2, false);
//   auto& hg = extracted_hg.first;
//   auto& hn_mapping = extracted_hg.second;

//   ASSERT_EQ(3, hg.initialNumNodes());
//   ASSERT_EQ(1, hg.initialNumEdges());
//   ASSERT_EQ(3, hg.initialNumPins());
//   ASSERT_EQ(3, hg.maxEdgeSize());

//   auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
//     return hn_mapping[hn];
//   };
//   parallel::scalable_vector<HypernodeID> node_id = {
//     map_from_original_to_extracted_hg(2),
//     map_from_original_to_extracted_hg(5),
//     map_from_original_to_extracted_hg(6)
//   };

//   this->verifyPins(hg, {0},
//     { {node_id[0], node_id[1], node_id[2]} });
// }

// TYPED_TEST(APartitionedGraph, ExtractBlockZeroWithCommunityInformation) {
//   this->hypergraph.setCommunityID(0, 0);
//   this->hypergraph.setCommunityID(1, 1);
//   this->hypergraph.setCommunityID(2, 0);
//   this->hypergraph.setCommunityID(3, 2);
//   this->hypergraph.setCommunityID(4, 3);
//   this->hypergraph.setCommunityID(5, 4);
//   this->hypergraph.setCommunityID(6, 5);

//   auto extracted_hg = this->partitioned_hypergraph.extract(TBBNumaArena::GLOBAL_TASK_GROUP, 0, true);
//   auto& hg = extracted_hg.first;
//   auto& hn_mapping = extracted_hg.second;

//   auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
//     return hn_mapping[hn];
//   };

//   ASSERT_EQ(0, hg.communityID(map_from_original_to_extracted_hg(0)));
//   ASSERT_EQ(1, hg.communityID(map_from_original_to_extracted_hg(1)));
//   ASSERT_EQ(0, hg.communityID(map_from_original_to_extracted_hg(2)));
// }

// TYPED_TEST(APartitionedGraph, ExtractBlockOneWithCommunityInformation) {
//   this->hypergraph.setCommunityID(0, 0);
//   this->hypergraph.setCommunityID(1, 1);
//   this->hypergraph.setCommunityID(2, 0);
//   this->hypergraph.setCommunityID(3, 2);
//   this->hypergraph.setCommunityID(4, 3);
//   this->hypergraph.setCommunityID(5, 4);
//   this->hypergraph.setCommunityID(6, 5);

//   auto extracted_hg = this->partitioned_hypergraph.extract(TBBNumaArena::GLOBAL_TASK_GROUP, 1, true);
//   auto& hg = extracted_hg.first;
//   auto& hn_mapping = extracted_hg.second;

//   auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
//     return hn_mapping[hn];
//   };

//   ASSERT_EQ(2, hg.communityID(map_from_original_to_extracted_hg(3)));
//   ASSERT_EQ(3, hg.communityID(map_from_original_to_extracted_hg(4)));
// }

// TYPED_TEST(APartitionedGraph, ExtractBlockTwoWithCommunityInformation) {
//   this->hypergraph.setCommunityID(0, 0);
//   this->hypergraph.setCommunityID(1, 1);
//   this->hypergraph.setCommunityID(2, 0);
//   this->hypergraph.setCommunityID(3, 2);
//   this->hypergraph.setCommunityID(4, 3);
//   this->hypergraph.setCommunityID(5, 4);
//   this->hypergraph.setCommunityID(6, 5);

//   auto extracted_hg = this->partitioned_hypergraph.extract(TBBNumaArena::GLOBAL_TASK_GROUP, 2, true);
//   auto& hg = extracted_hg.first;
//   auto& hn_mapping = extracted_hg.second;

//   auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
//     return hn_mapping[hn];
//   };

//   ASSERT_EQ(4, hg.communityID(map_from_original_to_extracted_hg(5)));
//   ASSERT_EQ(5, hg.communityID(map_from_original_to_extracted_hg(6)));
// }


TYPED_TEST(APartitionedGraph, ComputesPartInfoCorrectlyIfNodePartsAreSetOnly) {
  this->partitioned_hypergraph.resetPartition();
  this->partitioned_hypergraph.setOnlyNodePart(0, 0);
  this->partitioned_hypergraph.setOnlyNodePart(1, 0);
  this->partitioned_hypergraph.setOnlyNodePart(2, 0);
  this->partitioned_hypergraph.setOnlyNodePart(3, 1);
  this->partitioned_hypergraph.setOnlyNodePart(4, 1);
  this->partitioned_hypergraph.setOnlyNodePart(5, 2);
  this->partitioned_hypergraph.setOnlyNodePart(6, 2);
  this->partitioned_hypergraph.initializePartition(TBBNumaArena::GLOBAL_TASK_GROUP);

  ASSERT_EQ(3, this->partitioned_hypergraph.partWeight(0));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(1));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(2));
}

TYPED_TEST(APartitionedGraph, ComputesGainsCorrectly) {
  this->partitioned_hypergraph.initializeGainCache();

  this->verifyGains(0, {0, 0, 0});
  this->verifyGains(1, {0, 0, -1});
  this->verifyGains(2, {0, 0, -1});
  this->verifyGains(3, {1, 0, 0});
  this->verifyGains(4, {1, 0, 2});
  this->verifyGains(5, {-1, 0, 0});
  this->verifyGains(6, {-1, 0, 0});
}

}  // namespace ds
}  // namespace mt_kahypar