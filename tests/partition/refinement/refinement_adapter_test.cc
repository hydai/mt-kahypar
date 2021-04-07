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

#include "tests/partition/refinement/advanced_refiner_mock.h"
#include "mt-kahypar/partition/refinement/advanced/refiner_adapter.h"

using ::testing::Test;

#define MOVE(HN, FROM, TO) Move { FROM, TO, HN, 0 }
#define EMPTY_ADVANCED_PROBLEM AdvancedProblem { {}, ProblemStats(2) }

namespace mt_kahypar {

class AAdvancedRefinementAdapter : public Test {
 public:
  AAdvancedRefinementAdapter() :
    hg(HypergraphFactory::construct(TBBNumaArena::GLOBAL_TASK_GROUP,
      7 , 4, { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} }, nullptr, nullptr, true)),
    phg(2, TBBNumaArena::GLOBAL_TASK_GROUP, hg),
    context() {
    context.partition.k = 2;
    context.partition.perfect_balance_part_weights.assign(2, 3);
    context.partition.max_part_weights.assign(2, 4);
    context.partition.objective = kahypar::Objective::km1;
    context.shared_memory.num_threads = 8;
    context.refinement.advanced.algorithm = AdvancedRefinementAlgorithm::mock;
    context.refinement.advanced.num_threads_per_search = 1;

    AdvancedRefinerMockControl::instance().reset();

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
  std::unique_ptr<AdvancedRefinerAdapter> refiner;
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

TEST_F(AAdvancedRefinementAdapter, FailsToRegisterMoreSearchesIfAllAreUsed) {
  context.refinement.advanced.num_threads_per_search = 4;
  refiner = std::make_unique<AdvancedRefinerAdapter>(
    hg, context, TBBNumaArena::GLOBAL_TASK_GROUP);

  ASSERT_TRUE(refiner->registerNewSearch(0, phg));
  ASSERT_TRUE(refiner->registerNewSearch(1, phg));
  ASSERT_FALSE(refiner->registerNewSearch(2, phg));
}

TEST_F(AAdvancedRefinementAdapter, UseCorrectNumberOfThreadsForSearch1) {
  context.refinement.advanced.num_threads_per_search = 5;
  refiner = std::make_unique<AdvancedRefinerAdapter>(
    hg, context, TBBNumaArena::GLOBAL_TASK_GROUP);
  ASSERT_EQ(2, refiner->numAvailableRefiner());
  ASSERT_EQ(0, refiner->numUsedThreads());

  AdvancedRefinerMockControl::instance().refine_func =
    [&](const PartitionedHypergraph&, const AdvancedProblem&, const size_t num_threads) -> MoveSequence {
      EXPECT_EQ(5, num_threads);
      EXPECT_EQ(5, refiner->numUsedThreads());
      return MoveSequence { {}, 0 };
    };
  ASSERT_TRUE(refiner->registerNewSearch(0, phg));
  refiner->refine(0, phg, EMPTY_ADVANCED_PROBLEM);
  refiner->finalizeSearch(0);
}

TEST_F(AAdvancedRefinementAdapter, UseCorrectNumberOfThreadsForSearch2) {
  context.refinement.advanced.num_threads_per_search = 5;
  refiner = std::make_unique<AdvancedRefinerAdapter>(
    hg, context, TBBNumaArena::GLOBAL_TASK_GROUP);
  ASSERT_EQ(2, refiner->numAvailableRefiner());
  ASSERT_EQ(0, refiner->numUsedThreads());

  std::atomic<size_t> cnt(0);
  AdvancedRefinerMockControl::instance().refine_func =
    [&](const PartitionedHypergraph&, const AdvancedProblem&, const size_t num_threads) -> MoveSequence {
      EXPECT_EQ(5, num_threads);
      EXPECT_EQ(5, refiner->numUsedThreads());
      ++cnt;
      while ( cnt < 2 ) { }
      return MoveSequence { {}, 0 };
    };
  ASSERT_TRUE(refiner->registerNewSearch(0, phg));
  AdvancedRefinerMockControl::instance().refine_func =
    [&](const PartitionedHypergraph&, const AdvancedProblem&, const size_t num_threads) -> MoveSequence {
      EXPECT_EQ(3, num_threads);
      EXPECT_EQ(8, refiner->numUsedThreads());
      ++cnt;
      return MoveSequence { {}, 0 };
    };
  ASSERT_TRUE(refiner->registerNewSearch(1, phg));
  executeConcurrent([&] {
    refiner->refine(0, phg, EMPTY_ADVANCED_PROBLEM);
  }, [&] {
    while ( cnt < 1 ) { }
    refiner->refine(1, phg, EMPTY_ADVANCED_PROBLEM);
  });
  refiner->finalizeSearch(0);
  refiner->finalizeSearch(1);
}

}