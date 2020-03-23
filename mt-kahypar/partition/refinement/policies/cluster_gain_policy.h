/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2018 Sebastian Schlag <sebastian.schlag@kit.edu>
 * Copyright (C) 2018 Tobias Heuer <tobias.heuer@live.com>
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

#include "kahypar/meta/policy_registry.h"
#include "kahypar/meta/typelist.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/parallel/stl/scalable_vector.h"
#include "mt-kahypar/partition/context.h"

namespace mt_kahypar {
template <class Derived = Mandatory,
          class HyperGraph = Mandatory>
class ClusterGainPolicy : public kahypar::meta::PolicyBase {
  using TmpScores = parallel::scalable_vector<Gain>;
  using ClusterPinCountInPart = parallel::scalable_vector<parallel::scalable_vector<HypernodeID>>;

 public:
  struct ClusterMoveScore {
    ClusterMoveScore() :
      cluster(),
      best_gain(0),
      best_size(0),
      best_block(kInvalidPartition) { }

    void reset() {
      cluster.clear();
      best_gain = 0;
      best_size = 0;
      best_block = kInvalidPartition;
    }

    parallel::scalable_vector<HypernodeID> cluster;
    Gain best_gain;
    size_t best_size;
    PartitionID best_block;
  };

  ClusterGainPolicy(const Context& context) :
    _context(context),
    _tmp_gain(context.partition.k, 0),
    _cluster_score(),
    _cluster_block_weight(context.partition.k, 0),
    _cluster_weight(0) { }

  void updateGainOfCluster(const HyperGraph& hypergraph,
                           const HypernodeID hn,
                           const parallel::scalable_vector<HypernodeID>& cluster_pins_in_he,
                           const ClusterPinCountInPart& cluster_pin_count_in_part) {
    // Update scores of all blocks
    const PartitionID from = hypergraph.partID(hn);
    const HyperedgeWeight overall_delta =
      static_cast<Derived*>(this)->updateGainOfClusterImpl(
        hypergraph, hn, from, cluster_pins_in_he, cluster_pin_count_in_part);

    const HypernodeWeight hn_weight = hypergraph.nodeWeight(hn);
    for ( PartitionID to = 0; to < _context.partition.k; ++to ) {
      if ( from != to ) {
        _tmp_gain[to] += overall_delta;

        // Update best move
        // A move is better than the current best move if it
        // has a better gain or equal gain and the target block
        // is underloaded. In all cases, the move must satisfy the
        // balance constraints
        const Gain gain = _tmp_gain[to];
        const bool is_balanced =
          hypergraph.partWeight(to) + _cluster_weight - _cluster_block_weight[to] + hn_weight <=
          _context.partition.max_part_weights[to];
        const bool improves_balance = hypergraph.partWeight(to) <=
          _context.partition.perfect_balance_part_weights[to];
        if ( ( gain < _cluster_score.best_gain && is_balanced ) ||
             ( _context.refinement.cluster_label_propagation.rebalancing &&
               gain == _cluster_score.best_gain && is_balanced && improves_balance ) ) {
          _cluster_score.best_gain = gain;
          _cluster_score.best_size = _cluster_score.cluster.size() + 1;
          _cluster_score.best_block = to;
        }
      }
    }
  }

  void addVertexToCluster(HyperGraph& hypergraph,
                          const HypernodeID hn) {
    const PartitionID from = hypergraph.partID(hn);
    const HypernodeWeight hn_weight = hypergraph.nodeWeight(hn);
    _cluster_block_weight[from] += hn_weight;
    _cluster_weight += hn_weight;
    _cluster_score.cluster.push_back(hn);
  }

  ClusterMoveScore& clusterMoveScore() {
    return _cluster_score;
  }

  void reset() {
    _cluster_score.reset();
    for ( PartitionID block = 0; block < _context.partition.k; ++block ) {
      _cluster_block_weight[block] = 0;
      _tmp_gain[block] = 0;
    }
    _cluster_weight = 0;
  }

 protected:
  const Context& _context;
  TmpScores _tmp_gain;
  ClusterMoveScore _cluster_score;
  parallel::scalable_vector<HypernodeWeight> _cluster_block_weight;
  HypernodeWeight _cluster_weight;
};

template <class HyperGraph = Mandatory>
class ClusterKm1Policy : public ClusterGainPolicy<ClusterKm1Policy<HyperGraph>, HyperGraph> {
  using Base = ClusterGainPolicy<ClusterKm1Policy<HyperGraph>, HyperGraph>;
  using ClusterPinCountInPart = parallel::scalable_vector<parallel::scalable_vector<HypernodeID>>;

  static constexpr bool enable_heavy_assert = false;

 public:
  ClusterKm1Policy(const Context& context) :
    Base(context) { }

  HyperedgeWeight updateGainOfClusterImpl(const HyperGraph& hypergraph,
                                          const HypernodeID hn,
                                          const PartitionID from,
                                          const parallel::scalable_vector<HypernodeID>& cluster_pins_in_he,
                                          const ClusterPinCountInPart& cluster_pin_count_in_part) {
    ASSERT(from == hypergraph.partID(hn));

    HyperedgeWeight overall_delta = 0;
    for ( const HyperedgeID& he : hypergraph.incidentEdges(hn) ) {
      const HyperedgeID original_he_id = hypergraph.originalEdgeID(he);
      const HypernodeID cluster_pin_count_in_from_part = cluster_pin_count_in_part[original_he_id][from];
      const HypernodeID pin_count_in_from_part = hypergraph.pinCountInPart(he, from);
      const HypernodeID current_pin_count_in_from_part = pin_count_in_from_part - cluster_pin_count_in_from_part;
      const bool contains_at_least_one_pin_in_cluster = cluster_pins_in_he[original_he_id] > 0;
      const HyperedgeWeight he_weight = hypergraph.edgeWeight(he);

      bool decrease_connectivity = false;
      if ( current_pin_count_in_from_part > 1) {
        if ( !contains_at_least_one_pin_in_cluster ) {
          // If the vertex is the first vertex of the cluster in that
          // hyperedge and there are still more than one pins in that block,
          // moving that vertex to an other block would increase
          // the connectivity, except for those in the connectivity set.
          // Note, the gain those not change if we add an second vertex
          // from that hyperedge to the cluster from the same block. In case
          // we add a vertex from an other block the gain does also not
          // change, because moving both vertices to one block which
          // is not in the connectivity set would increase connectivity,
          // but we had considered that gain change already when we update
          // gain of the first vertex.
          overall_delta += he_weight;
          decrease_connectivity = true;
        }
      } else if ( contains_at_least_one_pin_in_cluster ) {
        // Since, all pins of the block are now contained in the cluster, moving
        // the whole cluster to an abitrary block will not increase connectivity any more.
        overall_delta -= he_weight;
      } else {
        // Special case: In case the vertex is the first pin of the hyperedge in the
        // cluster and the only pin in from part, the move will decrease connectivity if
        // we move it to a block in the connectivity set.
        decrease_connectivity = true;
      }

      if ( decrease_connectivity ) {
        for ( const PartitionID& to : hypergraph.connectivitySet(he) ) {
          if ( from != to ) {
            _tmp_gain[to] -= he_weight;
          }
        }
      }
    }

    return overall_delta;
  }

  using Base::_context;
  using Base::_tmp_gain;
};

}  // namespace mt_kahypar
