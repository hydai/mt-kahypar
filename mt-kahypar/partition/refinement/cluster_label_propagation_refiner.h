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

#pragma once

#include "kahypar/datastructure/fast_reset_flag_array.h"
#include "kahypar/datastructure/binary_heap.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/parallel/stl/scalable_vector.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/refinement/i_refiner.h"
#include "mt-kahypar/utils/randomize.h"

namespace mt_kahypar {
template <typename TypeTraits>
class ClusterLabelPropagationRefinerT final : public IRefinerT<TypeTraits> {
 private:
  using HyperGraph = typename TypeTraits::template PartitionedHyperGraph<>;
  using ClusterPinCountInPart = parallel::scalable_vector<parallel::scalable_vector<HypernodeID>>;
  using TmpScores = parallel::scalable_vector<Gain>;
  using PriorityQueue = kahypar::ds::BinaryMaxHeap<HypernodeID, HypernodeID>;

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

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

 public:
  explicit ClusterLabelPropagationRefinerT(HyperGraph&,
                                    const Context& context,
                                    const TaskGroupID task_group_id) :
    _context(context),
    _task_group_id(task_group_id),
    _nodes(),
    _marked(),
    _in_queue_he(1),
    _cluster_pin_count_in_part(),
    _cluster_pins_in_he(),
    _tmp_scores(context.partition.k, 0) { }

  ClusterLabelPropagationRefinerT(const ClusterLabelPropagationRefinerT&) = delete;
  ClusterLabelPropagationRefinerT(ClusterLabelPropagationRefinerT&&) = delete;

  ClusterLabelPropagationRefinerT & operator= (const ClusterLabelPropagationRefinerT &) = delete;
  ClusterLabelPropagationRefinerT & operator= (ClusterLabelPropagationRefinerT &&) = delete;

 private:
  bool refineImpl(HyperGraph& hypergraph,
                  const parallel::scalable_vector<HypernodeID>&,
                  kahypar::Metrics& best_metrics) override final {

    HyperedgeWeight total_gain = 0;
    for ( size_t it = 0; it < _context.refinement.cluster_label_propagation.max_iterations; ++it ) {
      utils::Randomize::instance().shuffleVector(_nodes, _nodes.size(), sched_getcpu());

      ClusterMoveScore c_score;
      PriorityQueue pq(hypergraph.initialNumNodes());
      parallel::scalable_vector<HypernodeWeight> cluster_weight(_context.partition.k, 0);
      HypernodeWeight total_cluster_weight = 0;
      for ( const HypernodeID& hn : _nodes ) {
        const HypernodeID original_id = hypergraph.originalNodeID(hn);
        if ( !_marked[original_id] ) {
          ASSERT(c_score.cluster.size() == 0);
          ASSERT(total_cluster_weight == 0);
          ASSERT(std::accumulate(cluster_weight.begin(), cluster_weight.end(), 0) == 0);
          ASSERT(std::accumulate(_tmp_scores.begin(), _tmp_scores.end(), 0) == 0);

          // Start cluster growing
          pq.push(original_id, 0);
          // We grow as long as there are vertices in the queue or the cluster
          // reach the maximum allowed cluster size
          while(!pq.empty() && c_score.cluster.size() < _context.refinement.cluster_label_propagation.max_cluster_size) {
            const HypernodeID original_hn_id = pq.top();
            const HypernodeID current_hn = hypergraph.globalNodeID(original_hn_id);
            pq.pop();

            // Compute gain of cluster to all blocks of the partition, if we
            // add the current vertex to cluster
            updateGainOfCluster(hypergraph, current_hn);

            // Add vertex to cluster
            c_score.cluster.push_back(current_hn);
            _marked[original_hn_id] = true;
            PartitionID from = hypergraph.partID(current_hn);
            cluster_weight[from] += hypergraph.nodeWeight(current_hn);
            total_cluster_weight += hypergraph.nodeWeight(current_hn);

            // Update cluster pin counts and insert all adjacent vertices into queue
            for ( const HyperedgeID& he : hypergraph.incidentEdges(current_hn) ) {
              const HyperedgeID original_he_id = hypergraph.originalEdgeID(he);
              ++_cluster_pin_count_in_part[original_he_id][from];
              ++_cluster_pins_in_he[original_he_id];
              if ( !_in_queue_he[original_he_id] && hypergraph.edgeSize(he) < _context.partition.hyperedge_size_threshold ) {
                for ( const HypernodeID& pin : hypergraph.pins(he) ) {
                  const HypernodeID original_pin_id = hypergraph.originalNodeID(pin);
                  const bool pq_contains_pin = pq.contains(original_pin_id);
                  if ( !pq_contains_pin &&
                      !_marked[original_pin_id] &&
                      hypergraph.isBorderNode(pin) ) {
                    pq.push(original_pin_id, 1);
                  } else if ( pq_contains_pin ) {
                    pq.updateKeyBy(original_pin_id, 1);
                  }
                }
                _in_queue_he.set(original_he_id, true);
              }
            }

            // Check if moving the cluster to an other block has a better gain
            // than the current best move under the restriction that the balance
            // constraint is not violated.
            for ( PartitionID to = 0; to < _context.partition.k; ++to ) {
              if ( _tmp_scores[to] < c_score.best_gain &&
                  hypergraph.partWeight(to) + total_cluster_weight - cluster_weight[to] <=
                  _context.partition.max_part_weights[to] ) {
                c_score.best_gain = _tmp_scores[to];
                c_score.best_size = c_score.cluster.size();
                c_score.best_block = to;
              } else if ( _tmp_scores[to] == c_score.best_gain &&
                  hypergraph.partWeight(to) + total_cluster_weight - cluster_weight[to] <=
                  _context.partition.max_part_weights[to] &&
                  hypergraph.partWeight(to) < _context.partition.perfect_balance_part_weights[to] ) {
                c_score.best_gain = _tmp_scores[to];
                c_score.best_size = c_score.cluster.size();
                c_score.best_block = to;
              }
            }
          }

          if ( c_score.best_block != kInvalidPartition ) {
            // In case move of cluster increase objective, we
            // apply the move
            applyClusterMove(hypergraph, c_score);
            total_gain += c_score.best_gain;
          } else {
            // Otherwise, we only reset cluster pin counts
            for ( const HypernodeID& hn : c_score.cluster ) {
              resetClusterPinCount(hypergraph, hn);
            }
          }

          // Reset internal data structures
          c_score.reset();
          total_cluster_weight = 0;
          for (PartitionID to = 0; to < _context.partition.k; ++to) {
            _tmp_scores[to] = 0;
            cluster_weight[to] = 0;
          }
          _in_queue_he.reset();
          pq.clear();
        }
      }

      for ( size_t i = 0; i < _marked.size(); ++i ) {
        _marked[i] = false;
      }
      LOG << "Total Improvement #" << it << " =" << total_gain;
    }

    HyperedgeWeight current_metric = best_metrics.getMetric(
      kahypar::Mode::direct_kway, _context.partition.objective);
    best_metrics.updateMetric(current_metric + total_gain,
      kahypar::Mode::direct_kway, _context.partition.objective);
    ASSERT(best_metrics.getMetric(kahypar::Mode::direct_kway, _context.partition.objective) ==
      metrics::objective(hypergraph, _context.partition.objective),
      V(best_metrics.getMetric(kahypar::Mode::direct_kway, _context.partition.objective)) <<
      V(metrics::objective(hypergraph, _context.partition.objective)));
    LOG << "Total Improvement =" << total_gain;
    return true;
  }

  void updateGainOfCluster(const HyperGraph& hypergraph,
                   const HypernodeID hn) {
    const PartitionID from = hypergraph.partID(hn);
    Gain internal_weight = 0;
    for ( const HyperedgeID& he : hypergraph.incidentEdges(hn) ) {
      const HyperedgeID original_he_id = hypergraph.originalEdgeID(he);
      const HypernodeID cluster_pin_count_in_from_part = _cluster_pin_count_in_part[original_he_id][from];
      const HypernodeID pin_count_in_from_part = hypergraph.pinCountInPart(he, from);
      const HypernodeID current_pin_count_in_from_part = pin_count_in_from_part - cluster_pin_count_in_from_part;
      const HyperedgeWeight he_weight = hypergraph.edgeWeight(he);

      if ( current_pin_count_in_from_part > 1 ) {
        if ( _cluster_pins_in_he[original_he_id] == 0 ) {
          internal_weight += he_weight;
          for ( const PartitionID& to : hypergraph.connectivitySet(he) ) {
            if ( from != to ) {
              _tmp_scores[to] -= he_weight;
            }
          }
        }
      } else if ( ( pin_count_in_from_part > 1 || _cluster_pins_in_he[original_he_id] > 0 ) &&
                    pin_count_in_from_part == cluster_pin_count_in_from_part + 1 ) {
        for (PartitionID to = 0; to < _context.partition.k; ++to) {
          if ( from != to ) {
            _tmp_scores[to] -= he_weight;
          }
        }
      } else {
        for ( const PartitionID& to : hypergraph.connectivitySet(he) ) {
          if ( from != to ) {
            _tmp_scores[to] -= he_weight;
          }
        }
      }
    }

    for (PartitionID to = 0; to < _context.partition.k; ++to) {
      if ( from != to ) {
        _tmp_scores[to] += internal_weight;
      }
    }
  }

  void applyClusterMove(HyperGraph& hypergraph,
                        ClusterMoveScore& c_score) {
    ASSERT(c_score.best_block != kInvalidPartition);
    ASSERT(c_score.best_gain < 0);
    ASSERT(c_score.best_size <= c_score.cluster.size());
    DBG << "Move cluster of size" << c_score.best_size << "to block"
        << c_score.best_block << "with gain" << c_score.best_gain;

    // Remove all vertices from cluster that are not contained
    // in cluster with best gain
    parallel::scalable_vector<HypernodeID>& cluster = c_score.cluster;
    while(cluster.size() > c_score.best_size) {
      const HypernodeID hn = cluster.back();
      resetClusterPinCount(hypergraph, hn);
      cluster.pop_back();
    }

    // Move cluster to block with best gain
    const PartitionID to = c_score.best_block;
    for ( const HypernodeID& hn : cluster ) {
      const PartitionID from = hypergraph.partID(hn);
      resetClusterPinCount(hypergraph, hn);
      if ( from != to ) {
        hypergraph.changeNodePart(hn, from, to);
      }
    }
  }

  void resetClusterPinCount(const HyperGraph& hypergraph,
                            const HypernodeID hn) {
    const PartitionID from = hypergraph.partID(hn);
    for ( const HyperedgeID& he : hypergraph.incidentEdges(hn) ) {
      const HyperedgeID original_he_id = hypergraph.originalEdgeID(he);
      ASSERT(_cluster_pin_count_in_part[original_he_id][from] > 0);
      ASSERT(_cluster_pins_in_he[original_he_id] > 0);
      --_cluster_pin_count_in_part[original_he_id][from];
      --_cluster_pins_in_he[original_he_id];
    }
  }

  void initializeImpl(HyperGraph& hypergraph) override final {
    _nodes.clear();
    _marked.assign(hypergraph.initialNumNodes(), false);
    kahypar::ds::FastResetFlagArray<> tmp_in_queue_hn(hypergraph.initialNumNodes());
    kahypar::ds::FastResetFlagArray<> tmp_in_queue_he(hypergraph.initialNumEdges());
    _in_queue_he.swap(tmp_in_queue_he);
    for ( const HypernodeID& hn : hypergraph.nodes() ) {
      if ( hypergraph.isBorderNode(hn) ) {
        _nodes.push_back(hn);
      }
    }

    _cluster_pins_in_he.assign(hypergraph.initialNumEdges(), 0);
    _cluster_pin_count_in_part.resize(hypergraph.initialNumEdges());
    tbb::parallel_for(ID(0), hypergraph.initialNumEdges(), [&](const HyperedgeID id) {
      parallel::scalable_vector<HypernodeID> tmp_pin_count_in_part(_context.partition.k, 0);
      _cluster_pin_count_in_part[id] = std::move(tmp_pin_count_in_part);
    });
  }

  const Context& _context;
  const TaskGroupID _task_group_id;

  parallel::scalable_vector<HypernodeID> _nodes;
  parallel::scalable_vector<uint8_t> _marked;
  kahypar::ds::FastResetFlagArray<> _in_queue_he;
  ClusterPinCountInPart _cluster_pin_count_in_part;
  parallel::scalable_vector<HypernodeID> _cluster_pins_in_he;
  TmpScores _tmp_scores;
};

using ClusterLabelPropagationRefiner = ClusterLabelPropagationRefinerT<GlobalTypeTraits>;
}  // namespace kahypar
