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

#include "static_graph.h"

#include "mt-kahypar/parallel/parallel_prefix_sum.h"
#include "mt-kahypar/datastructures/concurrent_bucket_map.h"
#include "mt-kahypar/utils/timer.h"
#include "mt-kahypar/utils/memory_tree.h"

#include <tbb/parallel_reduce.h>


namespace mt_kahypar::ds {

  // TODO split contraction into multiple functions!


  /*!
  * This struct is used during multilevel coarsening to efficiently
  * detect parallel hyperedges.
  */
  struct ContractedHyperedgeInformation {
    HyperedgeID he = kInvalidHyperedge;
    size_t hash = kEdgeHashSeed;
    size_t size = std::numeric_limits<size_t>::max();
    bool valid = false;
  };

  /*!
   * Contracts a given community structure. All vertices with the same label
   * are collapsed into the same vertex. The resulting single-pin and parallel
   * hyperedges are removed from the contracted graph. The function returns
   * the contracted hypergraph and a mapping which specifies a mapping from
   * community label (given in 'communities') to a vertex in the coarse hypergraph.
   *
   * \param communities Community structure that should be contracted
   * \param task_group_id Task Group ID
   */
  StaticGraph StaticGraph::contract(
          parallel::scalable_vector<HypernodeID>& communities,
          const TaskGroupID /* task_group_id */) {
    ASSERT(communities.size() == _num_nodes);

    if ( !_tmp_contraction_buffer ) {
      allocateTmpContractionBuffer();
    }

    // AUXILLIARY BUFFERS - Reused during multilevel hierarchy to prevent expensive allocations
    Array<HypernodeID>& mapping = _tmp_contraction_buffer->mapping;
    Array<Node>& tmp_nodes = _tmp_contraction_buffer->tmp_nodes;
    Array<HyperedgeID>& node_sizes = _tmp_contraction_buffer->node_sizes;
    Array<parallel::IntegralAtomicWrapper<HyperedgeID>>& tmp_num_incident_edges =
            _tmp_contraction_buffer->tmp_num_incident_edges;
    Array<parallel::IntegralAtomicWrapper<HypernodeWeight>>& node_weights =
            _tmp_contraction_buffer->node_weights;
    Array<TmpEdgeInformation>& tmp_edges = _tmp_contraction_buffer->tmp_edges;

    ASSERT(static_cast<size_t>(_num_nodes) <= mapping.size());
    ASSERT(static_cast<size_t>(_num_nodes) <= tmp_nodes.size());
    ASSERT(static_cast<size_t>(_num_nodes) <= node_sizes.size());
    ASSERT(static_cast<size_t>(_num_nodes) <= tmp_num_incident_edges.size());
    ASSERT(static_cast<size_t>(_num_nodes) <= node_weights.size());
    ASSERT(static_cast<size_t>(_num_edges) <= tmp_edges.size());


    // #################### STAGE 1 ####################
    // Compute vertex ids of coarse graph with a parallel prefix sum
    utils::Timer::instance().start_timer("preprocess_contractions", "Preprocess Contractions");
    mapping.assign(_num_nodes, 0);

    doParallelForAllNodes([&](const HypernodeID& node) {
      ASSERT(static_cast<size_t>(communities[node]) < mapping.size());
      mapping[communities[node]] = 1UL;
    });

    // Prefix sum determines vertex ids in coarse graph
    parallel::TBBPrefixSum<HyperedgeID, Array> mapping_prefix_sum(mapping);
    tbb::parallel_scan(tbb::blocked_range<size_t>(0UL, _num_nodes), mapping_prefix_sum);
    HypernodeID coarsened_num_nodes = mapping_prefix_sum.total_sum();

    // Remap community ids
    tbb::parallel_for(ID(0), _num_nodes, [&](const HypernodeID& node) {
      if ( nodeIsEnabled(node) ) {
        communities[node] = mapping_prefix_sum[communities[node]];
      } else {
        communities[node] = kInvalidHypernode;
      }

      // Reset tmp contraction buffer
      if ( node < coarsened_num_nodes ) {
        node_weights[node] = 0;
        tmp_nodes[node] = Node(true);
        node_sizes[node] = 0;
        tmp_num_incident_edges[node] = 0;
      }
    });

    // Mapping from a vertex id of the current hypergraph to its
    // id in the coarse hypergraph
    auto map_to_coarse_graph = [&](const HypernodeID node) {
      ASSERT(node < communities.size());
      return communities[node];
    };


    doParallelForAllNodes([&](const HypernodeID& node) {
      const HypernodeID coarse_node = map_to_coarse_graph(node);
      ASSERT(coarse_node < coarsened_num_nodes, V(coarse_node) << V(coarsened_num_nodes));
      // Weight vector is atomic => thread-safe
      node_weights[coarse_node] += nodeWeight(node);
      // In case community detection is enabled all vertices matched to one vertex
      // in the contracted hypergraph belong to same community. Otherwise, all communities
      // are default assigned to community 0
      // Aggregate upper bound for number of incident nets of the contracted vertex
      tmp_num_incident_edges[coarse_node] += nodeDegree(node);
    });
    utils::Timer::instance().stop_timer("preprocess_contractions");

    // #################### STAGE 2 ####################
    // In this step the incident edges of vertices are processed and stored inside the temporary
    // buffer. The vertex ids of the targets are remapped and edges that are contained inside
    // one community after contraction are marked as invalid. Note that parallel edges are not
    // invalidated yet.
    utils::Timer::instance().start_timer("tmp_copy_incident_edges", "Tmp Copy Incident Edges", true);

    // Compute start position the incident nets of a coarse vertex in the
    // temporary incident nets array with a parallel prefix sum
    parallel::scalable_vector<parallel::IntegralAtomicWrapper<HyperedgeID>> tmp_incident_edges_pos;
    parallel::TBBPrefixSum<parallel::IntegralAtomicWrapper<HyperedgeID>, Array>
            tmp_incident_edges_prefix_sum(tmp_num_incident_edges);
    tbb::parallel_invoke([&] {
      tbb::parallel_scan(tbb::blocked_range<size_t>(
              0UL, static_cast<size_t>(coarsened_num_nodes)), tmp_incident_edges_prefix_sum);
    }, [&] {
      tmp_incident_edges_pos.assign(coarsened_num_nodes, parallel::IntegralAtomicWrapper<HyperedgeID>(0));
    });

    // Write the incident edges of each contracted vertex to the temporary edge array
    doParallelForAllNodes([&](const HypernodeID& node) {
      const HypernodeID coarse_node = map_to_coarse_graph(node);
      const HyperedgeID node_degree = nodeDegree(node);
      const size_t coarse_edges_pos = tmp_incident_edges_prefix_sum[coarse_node] +
                                      tmp_incident_edges_pos[coarse_node].fetch_add(node_degree);
      const size_t edges_pos = _nodes[node].firstEntry();
      ASSERT(coarse_edges_pos + node_degree <= tmp_incident_edges_prefix_sum[coarse_node + 1]);
      ASSERT(edges_pos + node_degree <= _edges.size());
      for (size_t i = 0; i < static_cast<size_t>(node_degree); ++i) {
        const Edge& edge = _edges[edges_pos + i];
        const HypernodeID target = map_to_coarse_graph(edge.target());
        const bool is_valid = target != coarse_node;
        if (is_valid) {
          tmp_edges[coarse_edges_pos + i] = TmpEdgeInformation(edge.target(), edge.weight());
        } else {
          tmp_edges[coarse_edges_pos + i] = TmpEdgeInformation();
        }
      }
    });

    utils::Timer::instance().stop_timer("tmp_copy_incident_edges");




    StaticGraph hypergraph;








    return hypergraph;
  }


  // ! Copy static hypergraph in parallel
  StaticGraph StaticGraph::copy(const TaskGroupID /* task_group_id */) {
    StaticGraph hypergraph;

    hypergraph._num_hypernodes = _num_hypernodes;
    hypergraph._num_removed_hypernodes = _num_removed_hypernodes;
    hypergraph._num_hyperedges = _num_hyperedges;
    hypergraph._total_weight = _total_weight;

    tbb::parallel_invoke([&] {
      hypergraph._hypernodes.resize(_hypernodes.size());
      memcpy(hypergraph._hypernodes.data(), _hypernodes.data(),
             sizeof(Hypernode) * _hypernodes.size());
    }, [&] {
      hypergraph._incident_nets.resize(_incident_nets.size());
      memcpy(hypergraph._incident_nets.data(), _incident_nets.data(),
             sizeof(HyperedgeID) * _incident_nets.size());
    }, [&] {
      hypergraph._hyperedges.resize(_hyperedges.size());
      memcpy(hypergraph._hyperedges.data(), _hyperedges.data(),
             sizeof(Hyperedge) * _hyperedges.size());
    }, [&] {
      hypergraph._incidence_array.resize(_incidence_array.size());
      memcpy(hypergraph._incidence_array.data(), _incidence_array.data(),
             sizeof(HypernodeID) * _incidence_array.size());
    }, [&] {
      hypergraph._community_ids = _community_ids;
    });
    return hypergraph;
  }

  // ! Copy static hypergraph sequential
  StaticGraph StaticGraph::copy() {
    StaticGraph hypergraph;

    hypergraph._num_hypernodes = _num_hypernodes;
    hypergraph._num_removed_hypernodes = _num_removed_hypernodes;
    hypergraph._num_hyperedges = _num_hyperedges;
    hypergraph._total_weight = _total_weight;

    hypergraph._hypernodes.resize(_hypernodes.size());
    memcpy(hypergraph._hypernodes.data(), _hypernodes.data(),
           sizeof(Hypernode) * _hypernodes.size());
    hypergraph._incident_nets.resize(_incident_nets.size());
    memcpy(hypergraph._incident_nets.data(), _incident_nets.data(),
           sizeof(HyperedgeID) * _incident_nets.size());

    hypergraph._hyperedges.resize(_hyperedges.size());
    memcpy(hypergraph._hyperedges.data(), _hyperedges.data(),
           sizeof(Hyperedge) * _hyperedges.size());
    hypergraph._incidence_array.resize(_incidence_array.size());
    memcpy(hypergraph._incidence_array.data(), _incidence_array.data(),
           sizeof(HypernodeID) * _incidence_array.size());

    hypergraph._community_ids = _community_ids;

    return hypergraph;
  }




  void StaticGraph::memoryConsumption(utils::MemoryTreeNode* parent) const {
    ASSERT(parent);
    parent->addChild("Hypernodes", sizeof(Hypernode) * _hypernodes.size());
    parent->addChild("Incident Nets", sizeof(HyperedgeID) * _incident_nets.size());
    parent->addChild("Hyperedges", sizeof(Hyperedge) * _hyperedges.size());
    parent->addChild("Incidence Array", sizeof(HypernodeID) * _incidence_array.size());
    parent->addChild("Communities", sizeof(PartitionID) * _community_ids.capacity());
  }

  // ! Computes the total node weight of the hypergraph
  void StaticGraph::computeAndSetTotalNodeWeight(const TaskGroupID) {
    _total_weight = tbb::parallel_reduce(tbb::blocked_range<HypernodeID>(ID(0), _num_hypernodes), 0,
                                         [this](const tbb::blocked_range<HypernodeID>& range, HypernodeWeight init) {
                                           HypernodeWeight weight = init;
                                           for (HypernodeID hn = range.begin(); hn < range.end(); ++hn) {
                                             if (nodeIsEnabled(hn)) {
                                               weight += this->_hypernodes[hn].weight();
                                             }
                                           }
                                           return weight;
                                         }, std::plus<>());
  }

} // namespace