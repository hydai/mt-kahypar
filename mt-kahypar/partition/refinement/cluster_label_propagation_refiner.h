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

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

 public:
  explicit ClusterLabelPropagationRefinerT(HyperGraph&,
                                    const Context& context,
                                    const TaskGroupID task_group_id) :
    _context(context),
    _task_group_id(task_group_id),
    _nodes(),
    _marked(),
    _in_queue_hn(1),
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

    utils::Randomize::instance().shuffleVector(_nodes, _nodes.size(), sched_getcpu());

    HyperedgeWeight total_gain = 0;
    for ( const HypernodeID& hn : _nodes ) {
      const HypernodeID original_id = hypergraph.originalNodeID(hn);
      if ( !_marked[original_id] ) {
        parallel::scalable_vector<HypernodeID> cluster;
        parallel::scalable_vector<HypernodeWeight> cluster_weight(_context.partition.k, 0);
        HypernodeWeight total_cluster_weight = 0;
        parallel::scalable_queue<HypernodeID> q;
        q.push(hn);
        _in_queue_hn.set(original_id, true);
        Gain best_gain = 0;
        size_t best_size = 0;
        PartitionID best_block = kInvalidPartition;
        while(!q.empty() && cluster.size() < _context.refinement.cluster_label_propagation.max_cluster_size) {
          const HypernodeID current_hn = q.front();
          const HypernodeID original_hn_id = hypergraph.originalNodeID(current_hn);
          q.pop();

          updateScore(hypergraph, current_hn);
          cluster.push_back(current_hn);
          _marked[original_hn_id] = true;
          PartitionID from = hypergraph.partID(current_hn);
          cluster_weight[from] += hypergraph.nodeWeight(current_hn);
          total_cluster_weight += hypergraph.nodeWeight(current_hn);
          for ( const HyperedgeID& he : hypergraph.incidentEdges(current_hn) ) {
            const HyperedgeID original_he_id = hypergraph.originalEdgeID(he);
            ++_cluster_pin_count_in_part[original_he_id][from];
            ++_cluster_pins_in_he[original_he_id];
            if ( !_in_queue_he[original_he_id] ) {
              for ( const HypernodeID& pin : hypergraph.pins(he) ) {
                const HypernodeID original_pin_id = hypergraph.originalNodeID(pin);
                if ( !_in_queue_hn[original_pin_id] && !_marked[original_pin_id] && hypergraph.isBorderNode(pin) ) {
                  q.push(pin);
                  _in_queue_hn.set(original_pin_id, true);
                }
              }
              _in_queue_he.set(original_he_id, true);
            }
          }

          for ( PartitionID to = 0; to < _context.partition.k; ++to ) {
            if ( _tmp_scores[to] < best_gain &&
                 hypergraph.partWeight(to) + total_cluster_weight - cluster_weight[to] <=
                 _context.partition.max_part_weights[to] ) {
              best_gain = _tmp_scores[to];
              best_size = cluster.size();
              best_block = to;
            }
          }
        }

        if ( best_block != kInvalidPartition ) {
          while(cluster.size() > best_size) {
            const HypernodeID hn = cluster.back();
            resetClusterPinCount(hypergraph, hn);
            cluster.pop_back();
          }

          for ( const HypernodeID& hn : cluster ) {
            const PartitionID from = hypergraph.partID(hn);
            resetClusterPinCount(hypergraph, hn);
            if ( from != best_block ) {
              hypergraph.changeNodePart(hn, from, best_block);
            }
          }
          total_gain += best_gain;
        } else {
          for ( const HypernodeID& hn : cluster ) {
            resetClusterPinCount(hypergraph, hn);
          }
        }

        for (PartitionID to = 0; to < _context.partition.k; ++to) {
          _tmp_scores[to] = 0;
        }
        _in_queue_hn.reset();
        _in_queue_he.reset();
      }
    }


    HyperedgeWeight current_metric = best_metrics.getMetric(
      kahypar::Mode::direct_kway, _context.partition.objective);
    best_metrics.updateMetric(current_metric + total_gain,
      kahypar::Mode::direct_kway, _context.partition.objective);
    return false;
  }

  void updateScore(const HyperGraph& hypergraph,
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
      } else if ( pin_count_in_from_part > 1 && pin_count_in_from_part == cluster_pin_count_in_from_part + 1 ) {
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

  void resetClusterPinCount(const HyperGraph& hypergraph,
                            const HypernodeID hn) {
    const PartitionID from = hypergraph.partID(hn);
    for ( const HyperedgeID& he : hypergraph.incidentEdges(hn) ) {
      const HyperedgeID original_he_id = hypergraph.originalEdgeID(he);
      --_cluster_pin_count_in_part[original_he_id][from];
      --_cluster_pins_in_he[original_he_id];
    }
  }

  void initializeImpl(HyperGraph& hypergraph) override final {
    _nodes.clear();
    _marked.assign(hypergraph.initialNumNodes(), false);
    _in_queue_hn.setSize(hypergraph.initialNumNodes());
    _in_queue_he.setSize(hypergraph.initialNumEdges());
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
  kahypar::ds::FastResetFlagArray<> _in_queue_hn;
  kahypar::ds::FastResetFlagArray<> _in_queue_he;
  ClusterPinCountInPart _cluster_pin_count_in_part;
  parallel::scalable_vector<HypernodeID> _cluster_pins_in_he;
  TmpScores _tmp_scores;
};

using ClusterLabelPropagationRefiner = ClusterLabelPropagationRefinerT<GlobalTypeTraits>;
}  // namespace kahypar
