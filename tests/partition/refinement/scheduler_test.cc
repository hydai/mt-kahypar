/*******************************************************************************
 * This file is part of KaHyPar.
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

#include "gmock/gmock.h"

#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/partition/refinement/policies/gain_policy.h"
#include "mt-kahypar/partition/refinement/advanced/scheduler.h"
#include "tests/partition/refinement/advanced_refiner_mock.h"

using ::testing::Test;

#define MOVE(HN, FROM, TO) Move { FROM, TO, HN, 0 }

namespace mt_kahypar {

class AAdvancedRefinementScheduler : public Test {
 public:
  AAdvancedRefinementScheduler() :
    hg(HypergraphFactory::construct(TBBNumaArena::GLOBAL_TASK_GROUP,
      7 , 4, { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} }, nullptr, nullptr, true)),
    phg(2, TBBNumaArena::GLOBAL_TASK_GROUP, hg),
    context() {
    context.partition.k = 2;
    context.partition.perfect_balance_part_weights.assign(2, 3);
    context.partition.max_part_weights.assign(2, 4);
    context.partition.objective = kahypar::Objective::km1;

    context.shared_memory.num_threads = 2;
    context.refinement.advanced.algorithm = AdvancedRefinementAlgorithm::mock;
    context.refinement.advanced.num_threads_per_search = 1;
    context.refinement.advanced.num_cut_edges_per_block_pair = 50;
    context.refinement.advanced.max_bfs_distance = 2;

    phg.setOnlyNodePart(0, 0);
    phg.setOnlyNodePart(1, 0);
    phg.setOnlyNodePart(2, 0);
    phg.setOnlyNodePart(3, 0);
    phg.setOnlyNodePart(4, 1);
    phg.setOnlyNodePart(5, 1);
    phg.setOnlyNodePart(6, 1);
    phg.initializePartition(TBBNumaArena::GLOBAL_TASK_GROUP);
  }

  Hypergraph hg;
  PartitionedHypergraph phg;
  Context context;
};

template <class F, class K>
void executeConcurrent(F f1, K f2) {
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

void verifyPartWeights(const vec<HypernodeWeight> actual_weights,
                       const vec<HypernodeWeight> expected_weights) {
  ASSERT_EQ(actual_weights.size(), expected_weights.size());
  for ( size_t i = 0; i < actual_weights.size(); ++i ) {
    ASSERT_EQ(actual_weights[i], expected_weights[i]);
  }
}

TEST_F(AAdvancedRefinementScheduler, MovesOneVertex) {
  AdvancedRefinementScheduler refiner(hg, context, TBBNumaArena::GLOBAL_TASK_GROUP);
  refiner.initialize(phg);
  MoveSequence sequence { { MOVE(3, 0, 1) }, 1 };

  const HyperedgeWeight improvement = refiner.applyMoves(sequence);
  ASSERT_EQ(sequence.state, MoveSequenceState::SUCCESS);
  ASSERT_EQ(improvement, sequence.expected_improvement);
  ASSERT_EQ(1, phg.partID(3));
  verifyPartWeights(refiner.partWeights(), { 3, 4 });
}

TEST_F(AAdvancedRefinementScheduler, MovesVerticesWithIntermediateBalanceViolation) {
  AdvancedRefinementScheduler refiner(hg, context, TBBNumaArena::GLOBAL_TASK_GROUP);
  refiner.initialize(phg);
  MoveSequence sequence { { MOVE(5, 1, 0), MOVE(1, 0, 1), MOVE(3, 0, 1) }, 1 };

  const HyperedgeWeight improvement = refiner.applyMoves(sequence);
  ASSERT_EQ(sequence.state, MoveSequenceState::SUCCESS);
  ASSERT_EQ(improvement, sequence.expected_improvement);
  ASSERT_EQ(1, phg.partID(1));
  ASSERT_EQ(1, phg.partID(3));
  ASSERT_EQ(0, phg.partID(5));
  verifyPartWeights(refiner.partWeights(), { 3, 4 });
}

TEST_F(AAdvancedRefinementScheduler, MovesAVertexThatWorsenSolutionQuality) {
  AdvancedRefinementScheduler refiner(hg, context, TBBNumaArena::GLOBAL_TASK_GROUP);
  refiner.initialize(phg);
  MoveSequence sequence { { MOVE(0, 0, 1) }, 1 };

  const HyperedgeWeight improvement = refiner.applyMoves(sequence);
  ASSERT_EQ(sequence.state, MoveSequenceState::WORSEN_SOLUTION_QUALITY);
  ASSERT_EQ(improvement, 0);
  ASSERT_EQ(0, phg.partID(0));
  verifyPartWeights(refiner.partWeights(), { 4, 3 });
}

TEST_F(AAdvancedRefinementScheduler, MovesAVertexThatViolatesBalanceConstraint) {
  AdvancedRefinementScheduler refiner(hg, context, TBBNumaArena::GLOBAL_TASK_GROUP);
  refiner.initialize(phg);
  MoveSequence sequence { { MOVE(4, 1, 0) }, 1 };

  const HyperedgeWeight improvement = refiner.applyMoves(sequence);
  ASSERT_EQ(sequence.state, MoveSequenceState::VIOLATES_BALANCE_CONSTRAINT);
  ASSERT_EQ(improvement, 0);
  ASSERT_EQ(1, phg.partID(4));
  verifyPartWeights(refiner.partWeights(), { 4, 3 });
}

TEST_F(AAdvancedRefinementScheduler, MovesTwoVerticesConcurrently) {
  context.partition.max_part_weights.assign(2, 5);
  AdvancedRefinementScheduler refiner(hg, context, TBBNumaArena::GLOBAL_TASK_GROUP);
  refiner.initialize(phg);

  MoveSequence sequence_1 { { MOVE(3, 0, 1) }, 1 };
  MoveSequence sequence_2 { { MOVE(5, 1, 0) }, 0 };
  HypernodeWeight improvement_1 = 0, improvement_2 = 0;
  executeConcurrent([&] {
    improvement_1 = refiner.applyMoves(sequence_1);
    ASSERT_EQ(sequence_1.state, MoveSequenceState::SUCCESS);
    ASSERT_EQ(improvement_1, sequence_1.expected_improvement);
    ASSERT_EQ(1, phg.partID(3));
  }, [&] {
    improvement_2 = refiner.applyMoves(sequence_2);
    ASSERT_EQ(sequence_2.state, MoveSequenceState::SUCCESS);
    ASSERT_EQ(improvement_2, sequence_2.expected_improvement);
    ASSERT_EQ(0, phg.partID(5));
  });

  verifyPartWeights(refiner.partWeights(), { 4, 3 });
}

TEST_F(AAdvancedRefinementScheduler, MovesTwoVerticesConcurrentlyWhereOneViolateBalanceConstraint) {
  AdvancedRefinementScheduler refiner(hg, context, TBBNumaArena::GLOBAL_TASK_GROUP);
  refiner.initialize(phg);

  MoveSequence sequence_1 { { MOVE(3, 0, 1) }, 1 };
  MoveSequence sequence_2 { { MOVE(1, 0, 1) }, 0 };
  HypernodeWeight improvement_1 = 0, improvement_2 = 0;
  executeConcurrent([&] {
    improvement_1 = refiner.applyMoves(sequence_1);
  }, [&] {
    improvement_2 = refiner.applyMoves(sequence_2);
  });

  ASSERT_TRUE(sequence_1.state == MoveSequenceState::VIOLATES_BALANCE_CONSTRAINT ||
              sequence_2.state == MoveSequenceState::VIOLATES_BALANCE_CONSTRAINT);
  ASSERT_TRUE(sequence_1.state == MoveSequenceState::SUCCESS ||
              sequence_2.state == MoveSequenceState::SUCCESS);
  if ( sequence_1.state == MoveSequenceState::SUCCESS ) {
    ASSERT_EQ(improvement_1, sequence_1.expected_improvement);
    ASSERT_EQ(1, phg.partID(3));
    ASSERT_EQ(improvement_2, 0);
    ASSERT_EQ(0, phg.partID(1));
  } else {
    ASSERT_EQ(improvement_1, 0);
    ASSERT_EQ(0, phg.partID(3));
    ASSERT_EQ(improvement_2, sequence_2.expected_improvement);
    ASSERT_EQ(1, phg.partID(1));
  }
  verifyPartWeights(refiner.partWeights(), { 3, 4 });
}

class AnAdvancedRefinementEndToEnd : public Test {

  using GainCalculator = Km1Policy<PartitionedHypergraph>;

 public:
  AnAdvancedRefinementEndToEnd() :
    hg(),
    phg(),
    context(),
    max_part_weights(8, 200),
    mover(nullptr) {

    context.partition.graph_filename = "../tests/instances/ibm01.hgr";
    context.partition.k = 8;
    context.partition.epsilon = 0.03;
    context.partition.mode = kahypar::Mode::direct_kway;
    context.partition.objective = kahypar::Objective::km1;
    context.shared_memory.num_threads = std::thread::hardware_concurrency();
    context.refinement.advanced.algorithm = AdvancedRefinementAlgorithm::mock;
    context.refinement.advanced.num_threads_per_search = 1;
    context.refinement.advanced.num_cut_edges_per_block_pair = 50;
    context.refinement.advanced.max_bfs_distance = 2;

    // Read hypergraph
    hg = io::readHypergraphFile(
      context.partition.graph_filename, TBBNumaArena::GLOBAL_TASK_GROUP);
    phg = PartitionedHypergraph(
      context.partition.k, TBBNumaArena::GLOBAL_TASK_GROUP, hg);
    context.setupPartWeights(hg.totalWeight());

    // Read Partition
    std::vector<PartitionID> partition;
    io::readPartitionFile("../tests/instances/ibm01.hgr.part8", partition);
    phg.doParallelForAllNodes([&](const HypernodeID& hn) {
      phg.setOnlyNodePart(hn, partition[hn]);
    });
    phg.initializePartition(TBBNumaArena::GLOBAL_TASK_GROUP);

    AdvancedRefinerMockControl::instance().reset();
    // Define maximum problem size function
    AdvancedRefinerMockControl::instance().max_prob_size_func = [&](ProblemStats& stats) {
      bool limit_reached = true;
      for ( const PartitionID block : stats.containedBlocks() ) {
        bool block_limit_reached = stats.nodeWeightOfBlock(block) >= max_part_weights[block];
        if ( block_limit_reached ) stats.lockBlock(block);
        limit_reached &= block_limit_reached;
      }
      return limit_reached;
    };

    mover = std::make_unique<GainCalculator>(context);
    // Refine solution with simple label propagation
    AdvancedRefinerMockControl::instance().refine_func = [&](const PartitionedHypergraph& phg,
                                                             const AdvancedProblem& problem,
                                                             const size_t) {
      MoveSequence sequence { {}, 0 };
      for ( const HypernodeID& hn : problem.nodes ) {
        Move move = mover->computeMaxGainMove(phg, hn);
        ASSERT(move.from == phg.partID(hn));
        if ( move.from != move.to ) {
          sequence.moves.emplace_back(std::move(move));
          sequence.expected_improvement -= move.gain;
        }
      }
      return sequence;
    };

    // Move approx. 0.5% of the vertices randomly to a different block
    double p = 0.05;
    phg.doParallelForAllNodes([&](const HypernodeID& hn) {
      const int rand_int = utils::Randomize::instance().getRandomInt(0, 100, sched_getcpu());
      if ( rand_int <= p * 100 ) {
        const PartitionID from = phg.partID(hn);
        PartitionID to = utils::Randomize::instance().getRandomInt(
            0, context.partition.k - 1, sched_getcpu());
        while ( from == to ) {
          to = utils::Randomize::instance().getRandomInt(
            0, context.partition.k - 1, sched_getcpu());
        }
        phg.changeNodePart(hn, from, to, context.partition.max_part_weights[to], []{ }, NOOP_FUNC);
      }
    });

    utils::Timer::instance().clear();
    utils::Stats::instance().clear();
  }

  Hypergraph hg;
  PartitionedHypergraph phg;
  Context context;
  vec<HypernodeWeight> max_part_weights;
  std::unique_ptr<GainCalculator> mover;
};

TEST_F(AnAdvancedRefinementEndToEnd, SmokeTestWithTwoBlocksPerRefiner) {
  const bool debug = false;
  AdvancedRefinementScheduler scheduler(hg, context, TBBNumaArena::GLOBAL_TASK_GROUP);

  kahypar::Metrics metrics;
  metrics.cut = metrics::hyperedgeCut(phg);
  metrics.km1 = metrics::km1(phg);
  metrics.imbalance = metrics::imbalance(phg, context);

  if ( debug ) {
    LOG << "Start Solution km1 =" << metrics.km1;
  }

  scheduler.initialize(phg);
  scheduler.refine(phg, {}, metrics, 0.0);

  if ( debug ) {
    LOG << "Final Solution km1 =" << metrics.km1;
  }

  if ( debug ) {
    LOG << utils::Timer::instance(true);
    LOG << utils::Stats::instance();
  }

  ASSERT_EQ(metrics::km1(phg), metrics.km1);
  ASSERT_EQ(metrics::imbalance(phg, context), metrics.imbalance);
  for ( PartitionID i = 0; i < context.partition.k; ++i ) {
    ASSERT_LE(phg.partWeight(i), context.partition.max_part_weights[i]);
  }
}

TEST_F(AnAdvancedRefinementEndToEnd, SmokeTestWithFourBlocksPerRefiner) {
  const bool debug = false;
  AdvancedRefinerMockControl::instance().max_num_blocks = 4;
  AdvancedRefinementScheduler scheduler(hg, context, TBBNumaArena::GLOBAL_TASK_GROUP);

  kahypar::Metrics metrics;
  metrics.cut = metrics::hyperedgeCut(phg);
  metrics.km1 = metrics::km1(phg);
  metrics.imbalance = metrics::imbalance(phg, context);

  if ( debug ) {
    LOG << "Start Solution km1 =" << metrics.km1;
  }

  scheduler.initialize(phg);
  scheduler.refine(phg, {}, metrics, 0.0);

  if ( debug ) {
    LOG << "Final Solution km1 =" << metrics.km1;
  }

  if ( debug ) {
    LOG << utils::Timer::instance(true);
    LOG << utils::Stats::instance();
  }

  ASSERT_EQ(metrics::km1(phg), metrics.km1);
  ASSERT_EQ(metrics::imbalance(phg, context), metrics.imbalance);
  for ( PartitionID i = 0; i < context.partition.k; ++i ) {
    ASSERT_LE(phg.partWeight(i), context.partition.max_part_weights[i]);
  }
}


}