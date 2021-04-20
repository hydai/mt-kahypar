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
    Array<HyperedgeID>& edge_id_mapping = _tmp_contraction_buffer->edge_id_mapping;

    ASSERT(static_cast<size_t>(_num_nodes) <= mapping.size());
    ASSERT(static_cast<size_t>(_num_nodes) <= tmp_nodes.size());
    ASSERT(static_cast<size_t>(_num_nodes) <= node_sizes.size());
    ASSERT(static_cast<size_t>(_num_nodes) <= tmp_num_incident_edges.size());
    ASSERT(static_cast<size_t>(_num_nodes) <= node_weights.size());
    ASSERT(static_cast<size_t>(_num_edges) <= tmp_edges.size());
    ASSERT(static_cast<size_t>(_num_edges / 2) <= edge_id_mapping.size());


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
    tbb::parallel_scan(tbb::blocked_range<size_t>(ID(0), _num_nodes), mapping_prefix_sum);
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
              ID(0), static_cast<size_t>(coarsened_num_nodes)), tmp_incident_edges_prefix_sum);
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
          tmp_edges[coarse_edges_pos + i] = TmpEdgeInformation(target, edge.weight(), edge.uniqueID());
        } else {
          tmp_edges[coarse_edges_pos + i] = TmpEdgeInformation();
        }
      }
    });

    utils::Timer::instance().stop_timer("tmp_copy_incident_edges");

    // #################### STAGE 3 ####################
    // In this step, we deduplicate parallel edges. To this end, the incident edges
    // of each vertex are sorted and aggregated. However, there is a special treatment
    // for vertices with extremely high degree, as they might become a bottleneck
    // otherwise. Afterwards, for all parallel edges all but one are invalidated and
    // the weight of the remaining edge is set to the sum of the weights.
    utils::Timer::instance().start_timer("remove_parallel_edges", "Remove Parallel Edges", true);

    // A list of high degree vertices that are processed afterwards
    parallel::scalable_vector<HypernodeID> high_degree_vertices;
    std::mutex high_degree_vertex_mutex;
    tbb::parallel_for(ID(0), coarsened_num_nodes, [&](const HypernodeID& coarse_node) {
      // Remove duplicates
      const size_t incident_edges_start = tmp_incident_edges_prefix_sum[coarse_node];
      const size_t incident_edges_end = tmp_incident_edges_prefix_sum[coarse_node + 1];
      const size_t tmp_degree = incident_edges_end - incident_edges_start;
      // TODO(maas): is coarsened_num_nodes a sensible threshold?
      if ( tmp_degree <= std::max(coarsened_num_nodes, HIGH_DEGREE_CONTRACTION_THRESHOLD) ) {
        std::sort(tmp_edges.begin() + incident_edges_start, tmp_edges.begin() + incident_edges_end,
                  [](const TmpEdgeInformation& e1, const TmpEdgeInformation& e2) {
                    return e1._target < e2._target;
                  });

        // Deduplicate, aggregate weights and calculate minimum unique id
        //
        // <-- deduplicated --> <-- already processed --> <-- to be processed --> <-- invalid edges -->
        //                    ^                         ^
        // valid_edge_index ---        tmp_edge_index ---
        size_t valid_edge_index = incident_edges_start;
        size_t tmp_edge_index = incident_edges_start + 1;
        while (tmp_edge_index < incident_edges_end && tmp_edges[tmp_edge_index].isValid()) {
          HEAVY_COARSENING_ASSERT(
            [&](){
              size_t i = incident_edges_start;
              for (; i <= valid_edge_index; ++i) {
                if (!tmp_edges[i].isValid()) {
                  return false;
                } else if ((i + 1 <= valid_edge_index) &&
                  tmp_edges[i].getTarget() >= tmp_edges[i + 1].getTarget()) {
                  return false;
                }
              }
              for (; i < tmp_edge_index; ++i) {
                if (tmp_edges[i].isValid()) {
                  return false;
                }
              }
              return true;
            }(),
            "Invariant violated while deduplicating incident edges!"
          );

          TmpEdgeInformation& valid_edge = tmp_edges[valid_edge_index];
          TmpEdgeInformation& next_edge = tmp_edges[tmp_edge_index];
          if (next_edge.isValid()) {
            if (valid_edge.getTarget() == next_edge.getTarget()) {
              valid_edge.addWeight(next_edge.getWeight());
              valid_edge.updateID(next_edge.getID());
              next_edge.invalidate();
            } else {
              ++valid_edge_index;
              std::swap(tmp_edges[valid_edge_index], next_edge);
            }
            ++tmp_edge_index;
          }
        }
        const HyperedgeID contracted_size = tmp_edges[valid_edge_index].isValid() ? 
                                            (valid_edge_index - incident_edges_start + 1) : 0;
        node_sizes[coarse_node] = contracted_size;
      } else {
        std::lock_guard<std::mutex> lock(high_degree_vertex_mutex);
        high_degree_vertices.push_back(coarse_node);
      }
      tmp_nodes[coarse_node].setWeight(node_weights[coarse_node]);
      tmp_nodes[coarse_node].setFirstEntry(incident_edges_start);
    });

    if ( !high_degree_vertices.empty() ) {
      // High degree vertices are treated special, because sorting and afterwards
      // removing duplicates can become a major sequential bottleneck. Therefore,
      // we sum the parallel incident edges of a high degree vertex using an atomic
      // vector. Then, the index is calculated with a prefix sum.
      parallel::scalable_vector<parallel::IntegralAtomicWrapper<HyperedgeWeight>> summed_edge_weights_for_target;
      summed_edge_weights_for_target.assign(coarsened_num_nodes, parallel::IntegralAtomicWrapper<HyperedgeWeight>(0));
      parallel::scalable_vector<parallel::IntegralAtomicWrapper<HyperedgeID>> min_edge_id_for_target;
      min_edge_id_for_target.assign(coarsened_num_nodes,
        parallel::IntegralAtomicWrapper<HyperedgeID>(std::numeric_limits<HyperedgeID>::max()));
      parallel::scalable_vector<HyperedgeID> incident_edges_inclusion;
      incident_edges_inclusion.assign(coarsened_num_nodes, 0);
      for ( const HypernodeID& coarse_node : high_degree_vertices ) {
        const size_t incident_edges_start = tmp_incident_edges_prefix_sum[coarse_node];
        const size_t incident_edges_end = tmp_incident_edges_prefix_sum[coarse_node + 1];

        // sum edge weights for each target node and calculate id
        tbb::parallel_for(incident_edges_start, incident_edges_end, [&](const size_t pos) {
          TmpEdgeInformation& edge = tmp_edges[pos];
          if (edge.isValid()) {
            const HyperedgeID target = edge.getTarget();
            const HyperedgeID id = edge.getID();
            summed_edge_weights_for_target[target].fetch_add(edge.getWeight());
            HyperedgeID expected = min_edge_id_for_target[target].load();
            while (id < expected) {
              min_edge_id_for_target[target].compare_exchange_weak(expected, id);
            }
          }
        });

        // each edge with weight greater than zero is included
        tbb::parallel_for(ID(0), coarsened_num_nodes, [&](const size_t target) {
          bool include = summed_edge_weights_for_target[target].load() > 0;
          incident_edges_inclusion[target] = include ? 1 : 0;
        });

        // calculate relative index of edges via prefix sum
        parallel::TBBPrefixSum<HyperedgeID, parallel::scalable_vector> incident_edges_pos(incident_edges_inclusion);
        tbb::parallel_scan(tbb::blocked_range<size_t>(ID(0), static_cast<size_t>(coarsened_num_nodes)), incident_edges_pos);

        // insert edges
        tbb::parallel_for(ID(0), coarsened_num_nodes, [&](const size_t target) {
          const HyperedgeWeight weight = summed_edge_weights_for_target[target].load();
          if (weight > 0) {
            const HyperedgeID id = min_edge_id_for_target[target].load();
            tmp_edges[incident_edges_start + incident_edges_pos[target]] = TmpEdgeInformation(target, weight, id);
            summed_edge_weights_for_target[target].store(0);
            min_edge_id_for_target[target].store(std::numeric_limits<HyperedgeID>::max());
          }
        });

        const size_t contracted_size = incident_edges_pos.total_sum();
        node_sizes[coarse_node] = contracted_size;
      }
    }

    utils::Timer::instance().stop_timer("remove_parallel_edges");

    // #################### STAGE 4 ####################
    // Coarsened graph is constructed here by writting data from temporary
    // buffers to corresponding members in coarsened graph. We compute
    // a prefix sum over the vertex sizes to determine the start index
    // of the edges in the edge array, removing all invalid edges.
    // Afterwards, we additionally need to reconstruct the backwards edges
    // (hopefully not necessary anymore in the future).
    utils::Timer::instance().start_timer("contract_hypergraph", "Contract Hypergraph");

    StaticGraph hypergraph;

    // Compute number of edges in coarse graph (those flagged as valid)
    parallel::TBBPrefixSum<HyperedgeID, Array> degree_mapping(node_sizes);
    tbb::parallel_scan(tbb::blocked_range<size_t>(
            ID(0), static_cast<size_t>(coarsened_num_nodes)), degree_mapping);
    const HyperedgeID coarsened_num_edges = degree_mapping.total_sum();
    hypergraph._num_nodes = coarsened_num_nodes;
    hypergraph._num_edges = coarsened_num_edges;

    edge_id_mapping.assign(_num_edges / 2, 0);
    tbb::parallel_invoke([&] {
      utils::Timer::instance().start_timer("setup_edges", "Setup Edges", true);
      // Copy edges
      hypergraph._edges.resize(coarsened_num_edges);
      tbb::parallel_for(ID(0), coarsened_num_nodes, [&](const HyperedgeID& coarse_node) {
        const HyperedgeID tmp_edges_start = tmp_nodes[coarse_node].firstEntry();
        const HyperedgeID edges_start = degree_mapping[coarse_node];
        // TODO(maas): good idea? Better handle high degree nodes separately?
        tbb::parallel_for(ID(0), degree_mapping.value(coarse_node), [&](const HyperedgeID& index) {
          ASSERT(tmp_edges_start + index < tmp_edges.size() && edges_start + index < hypergraph._edges.size());
          const TmpEdgeInformation& tmp_edge = tmp_edges[tmp_edges_start + index];
          Edge& edge = hypergraph.edge(edges_start + index);          
          edge.setTarget(tmp_edge.getTarget());
          edge.setSource(coarse_node);
          edge.setWeight(tmp_edge.getWeight());
          edge.setUniqueID(tmp_edge.getID());
          ASSERT(static_cast<size_t>(tmp_edge.getID()) < edge_id_mapping.size());
          edge_id_mapping[tmp_edge.getID()] = 1UL;
        });
      });
      utils::Timer::instance().stop_timer("setup_edges");
    }, [&] {
      hypergraph._nodes.resize(coarsened_num_nodes + 1);
      tbb::parallel_for(ID(0), coarsened_num_nodes, [&](const HyperedgeID& coarse_node) {
        Node& node = hypergraph.node(coarse_node);
        node.enable();
        node.setFirstEntry(degree_mapping[coarse_node]);
        node.setWeight(tmp_nodes[coarse_node].weight());
      });
      hypergraph._nodes.back() = Node(static_cast<size_t>(coarsened_num_edges));
    }, [&] {
      hypergraph._community_ids.resize(coarsened_num_nodes);
      doParallelForAllNodes([&](HypernodeID fine_node) {
        hypergraph.setCommunityID(map_to_coarse_graph(fine_node), communityID(fine_node));
      });
    });

    utils::Timer::instance().start_timer("remap_edge_ids", "Remap Edge IDs", true);

    // Remap unique edge ids via prefix sum
    parallel::TBBPrefixSum<HyperedgeID, Array> edge_id_prefix_sum(edge_id_mapping);
    tbb::parallel_scan(tbb::blocked_range<size_t>(ID(0), _num_edges / 2), edge_id_prefix_sum);
    ASSERT(edge_id_prefix_sum.total_sum() == coarsened_num_edges / 2);

    tbb::parallel_for(ID(0), coarsened_num_edges, [&](const HyperedgeID& e) {
      Edge& edge = hypergraph.edge(e);
      edge.setUniqueID(edge_id_prefix_sum[edge.uniqueID()]);
    });

    utils::Timer::instance().stop_timer("remap_edge_ids");

    utils::Timer::instance().stop_timer("contract_hypergraph");


    hypergraph._total_weight = _total_weight;
    hypergraph._tmp_contraction_buffer = _tmp_contraction_buffer;
    _tmp_contraction_buffer = nullptr;
    return hypergraph;
  }


  // ! Copy static hypergraph in parallel
  StaticGraph StaticGraph::copy(const TaskGroupID /* task_group_id */) {
    StaticGraph hypergraph;

    hypergraph._num_nodes = _num_nodes;
    hypergraph._num_removed_nodes = _num_removed_nodes;
    hypergraph._num_edges = _num_edges;
    hypergraph._total_weight = _total_weight;

    tbb::parallel_invoke([&] {
      hypergraph._nodes.resize(_nodes.size());
      memcpy(hypergraph._nodes.data(), _nodes.data(),
             sizeof(Node) * _nodes.size());
    }, [&] {
      hypergraph._edges.resize(_edges.size());
      memcpy(hypergraph._edges.data(), _edges.data(),
             sizeof(Edge) * _edges.size());
    }, [&] {
      hypergraph._community_ids = _community_ids;
    });
    return hypergraph;
  }

  // ! Copy static hypergraph sequential
  StaticGraph StaticGraph::copy() {
    StaticGraph hypergraph;

    hypergraph._num_nodes = _num_nodes;
    hypergraph._num_removed_nodes = _num_removed_nodes;
    hypergraph._num_edges = _num_edges;
    hypergraph._total_weight = _total_weight;

    hypergraph._nodes.resize(_nodes.size());
    memcpy(hypergraph._nodes.data(), _nodes.data(),
           sizeof(Node) * _nodes.size());

    hypergraph._edges.resize(_edges.size());
    memcpy(hypergraph._edges.data(), _edges.data(),
           sizeof(Edge) * _edges.size());

    hypergraph._community_ids = _community_ids;

    return hypergraph;
  }




  void StaticGraph::memoryConsumption(utils::MemoryTreeNode* parent) const {
    ASSERT(parent);
    parent->addChild("Hypernodes", sizeof(Node) * _nodes.size());
    parent->addChild("Hyperedges", 2 * sizeof(Edge) * _edges.size());
    parent->addChild("Communities", sizeof(PartitionID) * _community_ids.capacity());
  }

  // ! Computes the total node weight of the hypergraph
  void StaticGraph::computeAndSetTotalNodeWeight(const TaskGroupID) {
    _total_weight = tbb::parallel_reduce(tbb::blocked_range<HypernodeID>(ID(0), _num_nodes), 0,
                                         [this](const tbb::blocked_range<HypernodeID>& range, HypernodeWeight init) {
                                           HypernodeWeight weight = init;
                                           for (HypernodeID hn = range.begin(); hn < range.end(); ++hn) {
                                             if (nodeIsEnabled(hn)) {
                                               weight += this->_nodes[hn].weight();
                                             }
                                           }
                                           return weight;
                                         }, std::plus<>());
  }

} // namespace