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
#include "mt-kahypar/partition/refinement/policies/cluster_gain_policy.h"
#include "mt-kahypar/utils/randomize.h"

namespace mt_kahypar {
template <typename TypeTraits,
          template <typename> class GainPolicy>
class ClusterLabelPropagationRefinerT final : public IRefinerT<TypeTraits> {
 private:
  using HyperGraph = typename TypeTraits::template PartitionedHyperGraph<>;
  using GainCalculator = GainPolicy<HyperGraph>;
  using ClusterMoveScore = typename GainCalculator::ClusterMoveScore;
  using ClusterPinCountInPart = parallel::scalable_vector<parallel::scalable_vector<HypernodeID>>;
  using PriorityQueue = kahypar::ds::BinaryMaxHeap<HypernodeID, HypernodeID>;

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

 public:
  explicit ClusterLabelPropagationRefinerT(HyperGraph&,
                                    const Context& context,
                                    const TaskGroupID task_group_id) :
    _context(context),
    _task_group_id(task_group_id),
    _nodes(),
    _gain(context),
    _marked(),
    _in_queue_he(1),
    _cluster_pin_count_in_part(),
    _cluster_pins_in_he(){ }

  ClusterLabelPropagationRefinerT(const ClusterLabelPropagationRefinerT&) = delete;
  ClusterLabelPropagationRefinerT(ClusterLabelPropagationRefinerT&&) = delete;

  ClusterLabelPropagationRefinerT & operator= (const ClusterLabelPropagationRefinerT &) = delete;
  ClusterLabelPropagationRefinerT & operator= (ClusterLabelPropagationRefinerT &&) = delete;

 private:
  bool refineImpl(HyperGraph& hypergraph,
                  const parallel::scalable_vector<HypernodeID>&,
                  kahypar::Metrics& best_metrics) override final {

    HyperedgeWeight total_gain = 0;
    // todo(heuer):
    // 1.) How many iterations are required until we can terminate CLP?
    //     a.) Fix number of iterations?
    //     b.) Early stopping criteria based on active nodes ord improvement?
    // 2.) Which nodes should be considered for current round?
    for ( size_t it = 0; it < _context.refinement.cluster_label_propagation.max_iterations; ++it ) {
      // todo(heuer):
      // 3.) How should we iterate over the set of active vertices?
      //     a.) Random or sorted in decreasing order of their node degree or number
      //         of incident cut hyperedges?
      utils::Randomize::instance().shuffleVector(_nodes, _nodes.size(), sched_getcpu());

      PriorityQueue pq(hypergraph.initialNumNodes());
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
          // todo(heuer):
          // 4.) What should be the maximum cluster size?
          //     a.) Fixed or dynamic based on instance (medium hyperedge size)?
          //     b.) Is there an early stopping criteria when we should stop growing the cluster?
          size_t cluster_size = 0;
          while(!pq.empty() && cluster_size < _context.refinement.cluster_label_propagation.max_cluster_size) {
            const HypernodeID original_hn_id = pq.top();
            const HypernodeID current_hn = hypergraph.globalNodeID(original_hn_id);
            pq.pop();

            // Compute gain of cluster to all blocks of the partition, if we
            // add the current vertex to cluster
            _gain.updateGainOfCluster(hypergraph, current_hn,
              _cluster_pins_in_he, _cluster_pin_count_in_part);
            _gain.addVertexToCluster(hypergraph, current_hn);
            ++cluster_size;

            // Update cluster pin counts and insert all adjacent vertices into queue
            // todo(heuer):
            // 5.) How to grow the cluster?
            //     a.) BFS-style
            //     b.) Number of incident hyperedges to cluster
            //     c.) Number of incident cut hyperedges
            _marked[original_hn_id] = true;
            PartitionID from = hypergraph.partID(current_hn);
            for ( const HyperedgeID& he : hypergraph.incidentEdges(current_hn) ) {
              const HyperedgeID original_he_id = hypergraph.originalEdgeID(he);
              ++_cluster_pin_count_in_part[original_he_id][from];
              ++_cluster_pins_in_he[original_he_id];
              if ( !_in_queue_he[original_he_id] && hypergraph.edgeSize(he) < _context.partition.hyperedge_size_threshold ) {
                for ( const HypernodeID& pin : hypergraph.pins(he) ) {
                  const HypernodeID original_pin_id = hypergraph.originalNodeID(pin);
                  const bool pq_contains_pin = pq.contains(original_pin_id);
                  // todo(heuer):
                  // 6.) Should non-border hypernodes be included in a cluster?
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
          }

          ClusterMoveScore& c_score = _gain.clusterMoveScore();
          if ( c_score.best_block != kInvalidPartition ) {
            // In case move of cluster decrease objective, we
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
          _gain.reset();
          _in_queue_he.reset();
          pq.clear();
        }
      }

      for ( size_t i = 0; i < _marked.size(); ++i ) {
        _marked[i] = false;
      }
      DBG << "Total Improvement #" << it << " =" << total_gain;
    }

    HyperedgeWeight current_metric = best_metrics.getMetric(
      kahypar::Mode::direct_kway, _context.partition.objective);
    best_metrics.updateMetric(current_metric + total_gain,
      kahypar::Mode::direct_kway, _context.partition.objective);
    ASSERT(best_metrics.getMetric(kahypar::Mode::direct_kway, _context.partition.objective) ==
      metrics::objective(hypergraph, _context.partition.objective),
      V(best_metrics.getMetric(kahypar::Mode::direct_kway, _context.partition.objective)) <<
      V(metrics::objective(hypergraph, _context.partition.objective)));
    DBG << "Total Improvement =" << total_gain;
    return true;
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
  GainCalculator _gain;
  parallel::scalable_vector<uint8_t> _marked;
  kahypar::ds::FastResetFlagArray<> _in_queue_he;
  ClusterPinCountInPart _cluster_pin_count_in_part;
  parallel::scalable_vector<HypernodeID> _cluster_pins_in_he;
};

using ClusterLabelPropagationRefiner = ClusterLabelPropagationRefinerT<GlobalTypeTraits, ClusterKm1Policy>;
}  // namespace kahypar
