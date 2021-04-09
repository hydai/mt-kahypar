/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
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

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/refinement/i_refiner.h"
#include "mt-kahypar/partition/refinement/advanced/quotient_graph.h"
#include "mt-kahypar/partition/refinement/advanced/refiner_adapter.h"
#include "mt-kahypar/partition/refinement/advanced/problem_construction.h"
#include "mt-kahypar/parallel/atomic_wrapper.h"

namespace mt_kahypar {

class AdvancedRefinementScheduler final : public IRefiner {

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

  struct RefinementStats {
    RefinementStats() :
      num_refinements(0),
      num_improvements(0),
      correct_expected_improvement(0),
      failed_updates_due_to_conflicting_moves(0),
      failed_updates_due_to_conflicting_moves_without_rollback(0),
      failed_updates_due_to_balance_constraint(0),
      total_improvement(0) { }

    void reset() {
      num_refinements.store(0);
      num_improvements.store(0);
      correct_expected_improvement.store(0);
      failed_updates_due_to_conflicting_moves.store(0);
      failed_updates_due_to_conflicting_moves_without_rollback.store(0);
      failed_updates_due_to_balance_constraint.store(0);
      total_improvement.store(0);
    }

    void update_global_stats();

    CAtomic<int64_t> num_refinements;
    CAtomic<int64_t> num_improvements;
    CAtomic<int64_t> correct_expected_improvement;
    CAtomic<int64_t> failed_updates_due_to_conflicting_moves;
    CAtomic<int64_t> failed_updates_due_to_conflicting_moves_without_rollback;
    CAtomic<int64_t> failed_updates_due_to_balance_constraint;
    CAtomic<HyperedgeWeight> total_improvement;
  };

public:
  explicit AdvancedRefinementScheduler(const Hypergraph& hg,
                                       const Context& context,
                                       const TaskGroupID task_group_id) :
    _phg(nullptr),
    _context(context),
    _quotient_graph(hg, context),
    _refiner(hg, context, task_group_id),
    _constructor(hg, context),
    _was_moved(hg.initialNumNodes(), uint8_t(false)),
    _part_weights_lock(),
    _part_weights(context.partition.k, 0),
    _max_part_weights(context.partition.k, 0),
    _stats() { }

  AdvancedRefinementScheduler(const AdvancedRefinementScheduler&) = delete;
  AdvancedRefinementScheduler(AdvancedRefinementScheduler&&) = delete;

  AdvancedRefinementScheduler & operator= (const AdvancedRefinementScheduler &) = delete;
  AdvancedRefinementScheduler & operator= (AdvancedRefinementScheduler &&) = delete;

  /**
   * Applies the sequence of vertex moves to the partitioned hypergraph.
   * The method ensures that the move sequence does not violate
   * the balance constaint and not worsen solution quality.
   * Returns, improvement in solution quality.
   */
  HyperedgeWeight applyMoves(MoveSequence& sequence);

  /**
   * Returns the current weight of each block.
   * Note, we do not want that the underlying refiner (ILP and Flows)
   * see partially updated part weight information. Thus, we perform
   * part weight updates for a move sequence as a transaction, which
   * we protect with a spin lock.
   */
  vec<HypernodeWeight> partWeights() {
    _part_weights_lock.lock();
    vec<HypernodeWeight> _copy_part_weights(_part_weights);
    _part_weights_lock.unlock();
    return _copy_part_weights;
  }

private:
  bool refineImpl(PartitionedHypergraph& phg,
                  const vec<HypernodeID>& refinement_nodes,
                  kahypar::Metrics& metrics,
                  double time_limit) final;

  void initializeImpl(PartitionedHypergraph& phg) final;

  bool partWeightUpdate(const vec<HypernodeWeight>& part_weight_deltas,
                        const bool rollback);

  PartitionedHypergraph* _phg;
  const Context& _context;

  // ! Contains information of all cut hyperedges between the
  // ! blocks of the partition
  QuotientGraph _quotient_graph;

  // ! Maintains the advanced refiner instances
  AdvancedRefinerAdapter _refiner;

  // ! Responsible for construction of an advanced refinement problem
  ProblemConstruction _constructor;

  // ! For each vertex it store wheather the corresponding vertex
  // ! was moved or not
  vec<uint8_t> _was_moved;

  // ! Maintains the part weights of each block
  SpinLock _part_weights_lock;
  vec<HypernodeWeight> _part_weights;
  vec<HypernodeWeight> _max_part_weights;

  // ! Contains refinement statistics
  RefinementStats _stats;
};

}  // namespace kahypar
