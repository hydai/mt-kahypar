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

#include "gmock/gmock.h"

#include <atomic>

#include "tbb/parallel_invoke.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/utils/timer.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/partition/registries/register_flat_initial_partitioning_algorithms.h"
#include "mt-kahypar/partition/initial_partitioning/flat/pool_initial_partitioner.h"

using ::testing::Test;

namespace mt_kahypar {

using TBB = typename GlobalTypeTraits::TBB;
using PoolInitialPartitioner = PoolInitialPartitionerT<GlobalTypeTraits>;

template<PartitionID k, size_t runs>
struct TestConfig {
  static constexpr PartitionID K = k;
  static constexpr size_t RUNS = runs;
};

template<typename Config>
class APoolInitialPartitionerTest : public Test {
 private:
  using HyperGraph = typename GlobalTypeTraits::HyperGraph;
  using HyperGraphFactory = typename GlobalTypeTraits::HyperGraphFactory;
  using PartitionedHyperGraph = typename GlobalTypeTraits::template PartitionedHyperGraph<>;
  using HwTopology = typename GlobalTypeTraits::HwTopology;

 public:

  APoolInitialPartitionerTest() :
    hypergraph(),
    partitioned_hypergraph(),
    context() {
    context.partition.k = Config::K;
    context.partition.epsilon = 0.2;
    context.partition.objective = kahypar::Objective::km1;
    context.initial_partitioning.runs = Config::RUNS;
    context.refinement.label_propagation.algorithm = LabelPropagationAlgorithm::label_propagation_km1;
    hypergraph = io::readHypergraphFile<HyperGraph, HyperGraphFactory>(
      "../test_instances/test_instance.hgr", TBB::GLOBAL_TASK_GROUP);
    partitioned_hypergraph = PartitionedHyperGraph(
      context.partition.k, HwTopology::instance().num_cpus(),
      TBB::GLOBAL_TASK_GROUP, hypergraph);
    context.setupPartWeights(hypergraph.totalWeight());
    utils::Timer::instance().disable();
  }

  static void SetUpTestSuite() {
    TBB::instance(HwTopology::instance().num_cpus());
  }

  HyperGraph hypergraph;
  PartitionedHyperGraph partitioned_hypergraph;
  Context context;
};

typedef ::testing::Types<TestConfig<2, 1>,
                         TestConfig<2, 2>,
                         TestConfig<2, 5>,
                         TestConfig<3, 1>,
                         TestConfig<3, 2>,
                         TestConfig<3, 5>,
                         TestConfig<4, 1>,
                         TestConfig<4, 2>,
                         TestConfig<4, 5>,
                         TestConfig<5, 1>,
                         TestConfig<5, 2>,
                         TestConfig<5, 5>> TestConfigs;

TYPED_TEST_CASE(APoolInitialPartitionerTest, TestConfigs);

TYPED_TEST(APoolInitialPartitionerTest, HasValidImbalance) {
  PoolInitialPartitioner& initial_partitioner = *new(tbb::task::allocate_root())
    PoolInitialPartitioner(this->partitioned_hypergraph, this->context, TBB::GLOBAL_TASK_GROUP);
  tbb::task::spawn_root_and_wait(initial_partitioner);

  ASSERT_LE(metrics::imbalance(this->partitioned_hypergraph, this->context),
            this->context.partition.epsilon);
}

TYPED_TEST(APoolInitialPartitionerTest, AssginsEachHypernode) {
  PoolInitialPartitioner& initial_partitioner = *new(tbb::task::allocate_root())
    PoolInitialPartitioner(this->partitioned_hypergraph, this->context, TBB::GLOBAL_TASK_GROUP);
  tbb::task::spawn_root_and_wait(initial_partitioner);

  for ( const HypernodeID& hn : this->partitioned_hypergraph.nodes() ) {
    ASSERT_NE(this->partitioned_hypergraph.partID(hn), -1);
  }
}

TYPED_TEST(APoolInitialPartitionerTest, HasNoSignificantLowPartitionWeights) {
  PoolInitialPartitioner& initial_partitioner = *new(tbb::task::allocate_root())
    PoolInitialPartitioner(this->partitioned_hypergraph, this->context, TBB::GLOBAL_TASK_GROUP);
  tbb::task::spawn_root_and_wait(initial_partitioner);

  // Each block should have a weight greater or equal than 20% of the average
  // block weight.
  for ( PartitionID block = 0; block < this->context.partition.k; ++block ) {
    ASSERT_GE(this->partitioned_hypergraph.partWeight(block),
              this->context.partition.perfect_balance_part_weights[block] / 5);
  }
}


}  // namespace mt_kahypar
