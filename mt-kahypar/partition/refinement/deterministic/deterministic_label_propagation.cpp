/*******************************************************************************
 * This file is part of MT-KaHyPar.
 *
 * Copyright (C) 2021 Lars Gottesbüren <lars.gottesbueren@kit.edu>
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

#include "deterministic_label_propagation.h"

#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/parallel/chunking.h"
#include "mt-kahypar/parallel/parallel_counting_sort.h"

#include <tbb/parallel_sort.h>
#include <tbb/parallel_reduce.h>

namespace mt_kahypar {

  bool DeterministicLabelPropagationRefiner::refineImpl(PartitionedHypergraph& phg,
                                                        const vec<HypernodeID>& ,
                                                        kahypar::Metrics& best_metrics,
                                                        const double)  {
    Gain overall_improvement = 0;
    size_t num_sub_rounds = context.refinement.deterministic_refinement.num_sub_rounds_sync_lp;

    for (size_t iter = 0; iter < context.refinement.label_propagation.maximum_iterations; ++iter) {
      moves_back.store(0, std::memory_order_relaxed);



      size_t n;
      if (context.refinement.deterministic_refinement.feistel_shuffling) {
        // get rid of this constant after initial tests, and use std::array in FeistelPermutation to store the keys
        constexpr size_t num_feistel_rounds = 4;
        feistel_permutation.create_permutation(num_feistel_rounds, phg.initialNumNodes(), prng);
        n = feistel_permutation.max_num_entries();
      } else {
        n = phg.initialNumNodes();
        permutation.create_integer_permutation(n, context.shared_memory.num_threads, prng);
      }

      size_t sub_round_size = parallel::chunking::idiv_ceil(n, num_sub_rounds);
      for (size_t sub_round = 0; sub_round < num_sub_rounds; ++sub_round) {
        // calculate moves
        auto [first, last] = parallel::chunking::bounds(sub_round, phg.initialNumNodes(), sub_round_size);
        if (context.refinement.deterministic_refinement.feistel_shuffling) {
          tbb::parallel_for(HypernodeID(first), HypernodeID(last), [&](const HypernodeID cleartext) {
            const HypernodeID ciphertext = feistel_permutation.encrypt(cleartext);
            if (ciphertext < phg.initialNumNodes()) {
              calculateAndSaveBestMove(phg, ciphertext);
            }
          });
        } else {
          tbb::parallel_for(HypernodeID(first), HypernodeID(last), [&](const HypernodeID position) {
            calculateAndSaveBestMove(phg, permutation.at(position));
          });
        }

        // sync. then apply moves
        overall_improvement += applyMovesSortedByGainAndRevertUnbalanced(phg);

      }
    }


    best_metrics.km1 -= overall_improvement;
    best_metrics.imbalance = metrics::imbalance(phg, context);
    return overall_improvement > 0;
  }

/*
 * for configs where we don't know exact gains --> have to trace the overall improvement with attributed gains
 * called from applyAllMoves() for example
*/
  Gain performMoveWithAttributedGain(PartitionedHypergraph& phg, const Move& m) {
    Gain attributed_gain = 0;
    auto objective_delta = [&](HyperedgeID he, HyperedgeWeight edge_weight, HypernodeID edge_size,
                               HypernodeID pin_count_in_from_part_after, HypernodeID pin_count_in_to_part_after) {
      attributed_gain -= km1Delta(he, edge_weight, edge_size, pin_count_in_from_part_after, pin_count_in_to_part_after);
    };
    phg.changeNodePart(m.node, m.from, m.to, objective_delta);
    return attributed_gain;
  }

  Gain applyAllValidMoves(PartitionedHypergraph& phg, const vec<Move>& moves, size_t end) {
    auto range = tbb::blocked_range<size_t>(0UL, end);
    auto accum = [&](const tbb::blocked_range<size_t>& r, const Gain& init) -> Gain {
      Gain my_gain = init;
      for (size_t i = r.begin(); i < r.end(); ++i) {
        if (moves[i].isValid()) {
          my_gain += performMoveWithAttributedGain(phg, moves[i]);
        }
      }
      return my_gain;
    };
    return tbb::parallel_reduce(range, 0, accum, std::plus<Gain>());
  }

  vec<HypernodeWeight> aggregatePartWeightDeltas(PartitionedHypergraph& phg, const vec<Move>& moves) {
    // parallel reduce makes way too many vector copies
    tbb::enumerable_thread_specific<vec<HypernodeWeight>> ets_part_weight_diffs(phg.k(), 0);
    auto accum = [&](const tbb::blocked_range<size_t>& r) {
      auto& part_weights = ets_part_weight_diffs.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        part_weights[moves[i].from] -= phg.nodeWeight(moves[i].node);
        part_weights[moves[i].to] += phg.nodeWeight(moves[i].node);
      }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0UL, moves.size()), accum);
    vec<HypernodeWeight> res(phg.k(), 0);
    auto combine = [&](const vec<HypernodeWeight>& a) {
      for (size_t i = 0; i < res.size(); ++i) {
        res[i] += a[i];
      }
    };
    ets_part_weight_diffs.combine_each(combine);
    return res;
  }

  Gain DeterministicLabelPropagationRefiner::applyMovesSortedByGainAndRevertUnbalanced(PartitionedHypergraph& phg) {
    auto comp = [](const Move& m1, const Move& m2) {
      return m1.gain > m2.gain || (m1.gain == m2.gain && m1.node < m2.node);
    };
    tbb::parallel_sort(moves.begin(), moves.begin() + moves_back, comp);

    size_t num_overloaded_blocks = 0;
    vec<HypernodeWeight> part_weights = aggregatePartWeightDeltas(phg, moves);
    for (PartitionID i = 0; i < phg.k(); ++i) {
      part_weights[i] += phg.partWeight(i);
      if (part_weights[i] > context.partition.max_part_weights[i]) {
        num_overloaded_blocks++;
      }
    }

    size_t j = moves_back.load(std::memory_order_relaxed);
    while (num_overloaded_blocks > 0 && j > 0) {
      Move& m = moves[--j];
      if (part_weights[m.to] > context.partition.max_part_weights[m.to]
          && part_weights[m.from] + phg.nodeWeight(m.node) <= context.partition.max_part_weights[m.from]) {
        part_weights[m.to] -= phg.nodeWeight(m.node);
        part_weights[m.from] += phg.nodeWeight(m.node);
        m.invalidate();
        if (part_weights[m.to] <= context.partition.max_part_weights[m.to]) {
          num_overloaded_blocks--;
        }
      }
    }

    return applyAllValidMoves(phg, moves, moves_back.load(std::memory_order_relaxed));
  }

  Gain DeterministicLabelPropagationRefiner::applyMovesByMaximalPrefixesInBlockPairs(PartitionedHypergraph& phg) {
    PartitionID k = phg.k();
    PartitionID max_key = k*k;
    auto index = [&](PartitionID b1, PartitionID b2) {
      return b1 * k + b2;
    };
    auto get_key = [&](const Move& m) {
      return index(m.from, m.to);
    };
    struct MovesWrapper {
      const Move& operator[](size_t i) const { return moves[i]; }
      size_t size() const { return sz; }
      const vec<Move>& moves;
      const size_t sz = 0;
    };

    MovesWrapper moves_wrapper { moves, moves_back.load(std::memory_order_relaxed) };

    // aggregate moves by direction. not in-place because of counting sort.
    // but it gives us the positions of the buckets right away
    auto positions = parallel::counting_sort(moves_wrapper, sorted_moves, max_key, get_key,
                                             context.shared_memory.num_threads);

    vec<std::pair<PartitionID, PartitionID>> relevant_block_pairs;
    vec<size_t> involvements(k, 0);
    for (PartitionID p1 = 0; p1 < k; ++p1) {
      for (PartitionID p2 = p1 + 1; p2 < k; ++p2) {
        if (positions[index(p1, p2) + 1] != positions[index(p1, p2)]
            && positions[index(p2, p1) + 1] != positions[index(p2, p1)]) { // neither direction (i,j) nor (j,i) empty
          relevant_block_pairs.emplace_back(p1, p2);
          involvements[p1]++;
          involvements[p2]++;
        }
      }
    }

    vec<std::pair<size_t, size_t>> swap_prefixes(relevant_block_pairs.size());

    tbb::parallel_for(0UL, relevant_block_pairs.size(), [&](size_t bp) {
      // sort both directions by gain (alternative: gain / weight?)
      auto [p1, p2] = relevant_block_pairs[bp];
      auto comp = [&](const Move& m1, const Move& m2) {
        return m1.gain > m2.gain || (m1.gain == m2.gain && m1.node < m2.node);
      };
      const auto b = sorted_moves.begin();
      size_t  i = positions[index(p1, p2)], i_last = positions[index(p1, p2) + 1],
              j = positions[index(p2, p1)], j_last = positions[index(p2, p1) + 1];
      std::sort(b + i, b + i_last, comp);
      std::sort(b + j, b + j_last, comp);


      // get balanced swap prefix
      HypernodeWeight budget_p1 = context.partition.max_part_weights[p1] - phg.partWeight(p1),
                      budget_p2 = context.partition.max_part_weights[p2] - phg.partWeight(p2);
      HypernodeWeight slack_p1 = budget_p1 / involvements[p1],
                      slack_p2 = budget_p2 / involvements[p2];
      std::pair<size_t, size_t> best {0,0};

      int64_t balance = 0;

      /*
       * this can be parallelized as follows.
       * 1. prefix sums of node weights over both move sequences
       * 2. pick middle of larger sequence, binary search for its prefix sum in the smaller sequence
       * 3. search for prefixes independently in both halves, and pick the better one
       *
       * in most cases we're expecting to take roughly as many moves as the size of the shorter sequence, from each of the sequences.
       * can we bias the search towards that?
       */

      // gain > 0 first. alternate depending on balance
      while (i < i_last && sorted_moves[i].gain > 0 && j < j_last && sorted_moves[j].gain > 0) {
        if (balance < 0 || (balance == 0 && sorted_moves[i].gain > sorted_moves[j].gain)) {
          // perform next move from p1 to p2
          balance += phg.nodeWeight(sorted_moves[i++].node);
        } else {
          // perform next move from p2 to p1
          balance -= phg.nodeWeight(sorted_moves[j++].node);
        }

        if (-balance <= slack_p1 && balance <= slack_p2) {
          best = {i,j};
        }
      }

      // if one sequence is depleted or gain == 0. only do rebalancing in the other direction
      if (j == j_last || sorted_moves[j].gain == 0) {
        while (i < i_last && balance <= slack_p2 && (balance < 0 || sorted_moves[i].gain > 0)) {
          balance+= phg.nodeWeight(sorted_moves[i++].node);
          if (-balance <= slack_p1 && balance <= slack_p2) {
            best = {i,j};
          }
        }
      } else if (i == i_last || sorted_moves[i].gain == 0) {
        while (j < j_last && -balance <= slack_p1 && (balance > 0 || sorted_moves[j].gain > 0)) {
          balance -= phg.nodeWeight(sorted_moves[j++].node);
          if (-balance <= slack_p1 && balance <= slack_p2) {
            best = {i,j};
          }
        }
      }

      swap_prefixes[bp] = best;
    });


    // TODO simple greedy combine after the swaps


    Gain gain = 0;

    return gain;
  }

  vec<size_t> DeterministicLabelPropagationRefiner::aggregateDirectionBucketsInplace() {
    // this can be done more efficiently, i.e. in linear time with counting sort. leave as is for now, for simplicity
    tbb::parallel_sort(moves.begin(),
                       moves.begin() + moves_back.load(std::memory_order_relaxed),
                       [](const Move& m1, const Move& m2) { return std::tie(m1.from, m1.to) < std::tie(m2.from, m2.to); }
    );

    PartitionID k = context.partition.k;
    auto index = [&](PartitionID b1, PartitionID b2) {
      return b1 * k + b2;
    };

    vec<size_t> positions(k*k + 1, 0);
    size_t pos = 0;
    for (PartitionID i = 0; i < k; ++i) {
      for (PartitionID j = 0; j < k; ++j) {
        while (pos < moves.size() && moves[pos].from == i && moves[pos].to == j) {
          ++pos;
        }
        positions[index(i,j) + 1] = pos;
      }
    }
    return positions;
  }


}