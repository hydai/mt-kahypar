/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2019, 2020 Lars Gottesbüren <lars.gottesbueren@kit.edu>
 * Copyright (C) 2020 Tobias Heuer <tobias.heuer@kit.edu>
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

#include "graph.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_invoke.h>
#include <tbb/enumerable_thread_specific.h>

#include "mt-kahypar/datastructures/sparse_map.h"

#include "mt-kahypar/parallel/stl/scalable_vector.h"
#include "mt-kahypar/parallel/parallel_prefix_sum.h"
#include "mt-kahypar/parallel/atomic_wrapper.h"
#include "mt-kahypar/utils/timer.h"
#include "mt-kahypar/parallel/parallel_counting_sort.h"


namespace mt_kahypar::ds {

  Graph::Graph(Hypergraph& hypergraph, const LouvainEdgeWeight edge_weight_type) :
          _num_nodes(0),
          _num_arcs(0),
          _total_volume(0),
          _max_degree(0),
          _indices(),
          _arcs(),
          _node_volumes(),
          _tmp_graph_buffer(nullptr) {

    switch( edge_weight_type ) {
      case LouvainEdgeWeight::uniform:
        construct(hypergraph,
                  [&](const HyperedgeWeight edge_weight,
                      const HypernodeID,
                      const HyperedgeID) {
                    return static_cast<ArcWeight>(edge_weight);
                  });
        break;
      case LouvainEdgeWeight::non_uniform:
        construct(hypergraph,
                  [&](const HyperedgeWeight edge_weight,
                      const HypernodeID edge_size,
                      const HyperedgeID) {
                    return static_cast<ArcWeight>(edge_weight) /
                           static_cast<ArcWeight>(edge_size);
                  });
        break;
      case LouvainEdgeWeight::degree:
        construct(hypergraph,
                  [&](const HyperedgeWeight edge_weight,
                      const HypernodeID edge_size,
                      const HyperedgeID node_degree) {
                    return static_cast<ArcWeight>(edge_weight) *
                           (static_cast<ArcWeight>(node_degree) /
                            static_cast<ArcWeight>(edge_size));
                  });
        break;
      case LouvainEdgeWeight::hybrid:
      case LouvainEdgeWeight::UNDEFINED:
      ERROR("No valid louvain edge weight");
    }
  }

  Graph::Graph(Graph&& other) :
    _num_nodes(other._num_nodes),
    _num_arcs(other._num_arcs),
    _total_volume(other._total_volume),
    _max_degree(other._max_degree),
    _indices(std::move(other._indices)),
    _arcs(std::move(other._arcs)),
    _node_volumes(std::move(other._node_volumes)),
    _tmp_graph_buffer(std::move(other._tmp_graph_buffer)) {
    other._num_nodes = 0;
    other._num_arcs = 0;
    other._total_volume = 0;
    other._max_degree = 0;
    other._tmp_graph_buffer = nullptr;
  }

  Graph& Graph::operator= (Graph&& other) {
    _num_nodes = other._num_nodes;
    _num_arcs = other._num_arcs;
    _total_volume = other._total_volume;
    _max_degree = other._max_degree;
    _indices = std::move(other._indices);
    _arcs = std::move(other._arcs);
    _node_volumes = std::move(other._node_volumes);
    _tmp_graph_buffer = std::move(other._tmp_graph_buffer);
    other._num_nodes = 0;
    other._num_arcs = 0;
    other._total_volume = 0;
    other._max_degree = 0;
    other._tmp_graph_buffer = nullptr;
    return *this;
  }

  Graph::~Graph() {
    if ( _tmp_graph_buffer ) {
      delete(_tmp_graph_buffer);
    }
  }

  Graph Graph::contract_low_memory(Clustering& communities) {
    // map cluster IDs to consecutive range
    vec<NodeID> mapping(numNodes(), 0);   // TODO extract?
    tbb::parallel_for(0UL, numNodes(), [&](NodeID u) { mapping[communities[u]] = 1; });
    parallel_prefix_sum(mapping.begin(), mapping.begin() + numNodes(), mapping.begin(), std::plus<>(), 0);
    NodeID num_coarse_nodes = mapping[numNodes() - 1];
    // apply mapping to cluster IDs. subtract one because prefix sum is inclusive
    tbb::parallel_for(0UL, numNodes(), [&](NodeID u) { communities[u] = mapping[communities[u]] - 1; });

    // sort nodes by cluster
    auto get_cluster = [&](NodeID u) { assert(u < communities.size()); return communities[u]; };
    vec<NodeID> nodes_sorted_by_cluster(std::move(mapping));    // reuse memory from mapping since it's no longer needed
    /*auto cluster_bounds = parallel::counting_sort(nodes(), nodes_sorted_by_cluster, num_coarse_nodes,
                                                  get_cluster, TBBNumaArena::instance().total_number_of_threads());
*/
    Graph coarse_graph;
    coarse_graph._num_nodes = num_coarse_nodes;
    coarse_graph._indices.resize(num_coarse_nodes + 1);
    coarse_graph._node_volumes.resize(num_coarse_nodes);
    coarse_graph._total_volume = totalVolume();

    /*
    // alternative easier counting sort implementation
    vec<uint32_t> cluster_bounds(num_coarse_nodes + 2, 0);
    tbb::parallel_for(0UL, numNodes(), [&](NodeID u) {
      __atomic_fetch_add(&cluster_bounds[get_cluster(u) + 2], 1, __ATOMIC_RELAXED);
    });
    parallel_prefix_sum(cluster_bounds.begin(), cluster_bounds.end(), cluster_bounds.begin(), std::plus<>(), 0);
    tbb::parallel_for(0UL, numNodes(), [&](NodeID u) {
      size_t pos = __atomic_fetch_add(&cluster_bounds[get_cluster(u) + 1], 1, __ATOMIC_RELAXED);
      nodes_sorted_by_cluster[pos] = u;
    });
     */

    vec<uint32_t> cluster_bounds(num_coarse_nodes + 2, 0);
    tbb::parallel_for(0UL, numNodes(), [&](NodeID u) {
      __atomic_fetch_add(&cluster_bounds[get_cluster(u) + 2], 1, __ATOMIC_RELAXED);
    });
    parallel_prefix_sum(cluster_bounds.begin(), cluster_bounds.end(), cluster_bounds.begin(), std::plus<>(), 0);
    tbb::parallel_for(0UL, numNodes(), [&](NodeID u) {
      size_t pos = __atomic_fetch_add(&cluster_bounds[get_cluster(u) + 1], 1, __ATOMIC_RELAXED);
      nodes_sorted_by_cluster[pos] = u;
    });

    // TODO pass map from local moving code?
    struct ClearList {
      vec<NodeID> used;
      vec<ArcWeight> values;

      ClearList(size_t n) : values(n, 0.0) { }
    };
    tbb::enumerable_thread_specific<ClearList> clear_lists(num_coarse_nodes);
    tbb::enumerable_thread_specific<size_t> local_max_degree(0);

    // first pass generating unique coarse arcs to determine coarse node degrees
    tbb::parallel_for(0U, num_coarse_nodes, [&](NodeID cu) {
      auto& clear_list = clear_lists.local();
      ArcWeight volume_cu = 0.0;
      for (auto i = cluster_bounds[cu]; i < cluster_bounds[cu + 1]; ++i) {
        NodeID fu = nodes_sorted_by_cluster[i];
        volume_cu += nodeVolume(fu);
        for (const Arc& arc : arcsOf(fu)) {
          NodeID cv = get_cluster(arc.head);
          if (cv != cu && clear_list.values[cv] == 0.0) {
            clear_list.used.push_back(cv);
            clear_list.values[cv] = 1.0;
          }
        }
      }
      coarse_graph._indices[cu + 1] = clear_list.used.size();
      local_max_degree.local() = std::max(local_max_degree.local(), clear_list.used.size());
      for (const NodeID cv : clear_list.used) {
        clear_list.values[cv] = 0.0;
      }
      clear_list.used.clear();
      coarse_graph._node_volumes[cu] = volume_cu;
    });

    // prefix sum coarse node degrees for offsets to write the coarse arcs in second pass
    parallel_prefix_sum(coarse_graph._indices.begin(), coarse_graph._indices.end(), coarse_graph._indices.begin(), std::plus<>(), 0UL);
    size_t num_coarse_arcs = coarse_graph._indices.back();
    // TODO get this to use reusable memory
    coarse_graph._arcs.resize(num_coarse_arcs);
    coarse_graph._num_arcs = num_coarse_arcs;
    coarse_graph._max_degree = local_max_degree.combine([](size_t lhs, size_t rhs) { return std::max(lhs, rhs); });

    // second pass generating unique coarse arcs
    tbb::parallel_for(0U, num_coarse_nodes, [&](NodeID cu) {
      auto& clear_list = clear_lists.local();
      for (auto i = cluster_bounds[cu]; i < cluster_bounds[cu+1]; ++i) {
        for (const Arc& arc : arcsOf(nodes_sorted_by_cluster[i])) {
          NodeID cv = get_cluster(arc.head);
          if (cv != cu) {
            if (clear_list.values[cv] == 0.0) {
              clear_list.used.push_back(cv);
            }
            clear_list.values[cv] += arc.weight;
          }
        }
      }
      size_t pos = coarse_graph._indices[cu];
      for (const NodeID cv : clear_list.used) {
        coarse_graph._arcs[pos++] = Arc(cv, clear_list.values[cv]);
        clear_list.values[cv] = 0.0;
      }
      clear_list.used.clear();
    });

    coarse_graph._tmp_graph_buffer = _tmp_graph_buffer;
    _tmp_graph_buffer = nullptr;

    return coarse_graph;
  }


  static constexpr std::size_t kChunkSize = (1 << 15);

  using LocalEdgeMemoryChunk = parallel::scalable_vector <Arc>;

  struct LocalEdgeMemory {
    LocalEdgeMemory() { current_chunk.reserve(kChunkSize); }

    parallel::scalable_vector<LocalEdgeMemoryChunk> chunks;
    LocalEdgeMemoryChunk current_chunk;

    std::size_t get_current_position() const { return chunks.size() * kChunkSize + current_chunk.size(); }

    void push(const NodeID c_v, const ArcWeight weight) {
      if (current_chunk.size() == kChunkSize) { flush(); }
      current_chunk.emplace_back(c_v, weight);
    }

    const auto &get(const std::size_t position) const { return chunks[position / kChunkSize][position % kChunkSize]; }

    void flush() {
      chunks.push_back(std::move(current_chunk));
      current_chunk.clear();
      current_chunk.reserve(kChunkSize);
    }
  };

  struct BufferNode {
    NodeID c_u;
    std::size_t position;
    LocalEdgeMemory *chunks;
  };

  /*!
 * Contracts the graph based on the community structure passed as argument.
 * In the first step the community ids are compactified (via parallel prefix sum)
 * which also determines the node ids in the coarse graph. Afterwards, we create
 * a temporary graph which contains all arcs that will not form a selfloop in the
 * coarse graph. Finally, the weights of each multiedge in that temporary graph
 * are aggregated and the result is written to the final contracted graph.
 */
  Graph Graph::contract(Clustering& communities) {
    ASSERT(canBeUsed());
    ASSERT(_num_nodes == communities.size());
    //return contract_low_memory(communities);
    ASSERT(_tmp_graph_buffer);
    Graph coarse_graph;
    coarse_graph._total_volume = _total_volume;

    // Compute node ids of coarse graph with a parallel prefix sum
    utils::Timer::instance().start_timer("compute_cluster_mapping", "Compute Cluster Mapping");
    parallel::scalable_vector<size_t> mapping(_num_nodes);
    ds::Array<parallel::IntegralAtomicWrapper<size_t>> &tmp_indices = _tmp_graph_buffer->tmp_indices;
    ds::Array<parallel::AtomicWrapper<ArcWeight>> &coarse_node_volumes = _tmp_graph_buffer->tmp_node_volumes;
    tbb::parallel_for(0U, static_cast<NodeID>(_num_nodes), [&](const NodeID u) {
      ASSERT(static_cast<size_t>(communities[u]) < _num_nodes);
      mapping[communities[u]] = 1UL;
      //tmp_pos[u] = 0;
      tmp_indices[u] = 0;
      coarse_node_volumes[u].store(0.0);
    });

    // Prefix sum determines node ids in coarse graph
    parallel::TBBPrefixSum<size_t> mapping_prefix_sum(mapping);
    tbb::parallel_scan(tbb::blocked_range<size_t>(0UL, _num_nodes), mapping_prefix_sum);

    // Remap community ids
    coarse_graph._num_nodes = mapping_prefix_sum.total_sum();
    tbb::parallel_for(0U, static_cast<NodeID>(_num_nodes), [&](const NodeID u) {
      communities[u] = mapping_prefix_sum[communities[u]];
    });
    utils::Timer::instance().stop_timer("compute_cluster_mapping");


    //
    // Sort nodes into buckets: place all nodes belonging to coarse node i into the i-th bucket
    //
    // Count the number of nodes in each bucket, then compute the position of the bucket in the global buckets array
    // using a prefix sum

    utils::Timer::instance().start_timer("sort_nodes_into_buckets",
                                         "Sort nodes into buckets belonging to their coarse node");
    parallel::scalable_vector<parallel::IntegralAtomicWrapper<NodeID>> buckets_index(coarse_graph._num_nodes + 1);
    //parallel::scalable_vector<NodeID> buckets(_num_nodes);
    ds::Array<NodeID>& buckets = _tmp_graph_buffer->buckets;
    tbb::parallel_for(static_cast<NodeID>(0), static_cast<NodeID>(_num_nodes),
                      [&](const NodeID u) { buckets_index[communities[u]].fetch_add(1, std::memory_order_relaxed); });

    parallel::TBBPrefixSum<parallel::IntegralAtomicWrapper<NodeID>> buckets_indices_prefix_sum(buckets_index);
    tbb::parallel_scan(tbb::blocked_range<size_t>(0UL, buckets_index.size()), buckets_indices_prefix_sum);

    ASSERT(buckets_indices_prefix_sum.total_sum() <= _num_nodes);
    // Sort nodes into   buckets
    tbb::parallel_for(static_cast<NodeID>(0), static_cast<NodeID>(_num_nodes), [&](const NodeID u) {
      const std::size_t pos = buckets_index[communities[u]].fetch_sub(1, std::memory_order_relaxed) - 1;
      buckets[pos] = u;
    });

    utils::Timer::instance().stop_timer("sort_nodes_into_buckets");

    //
    // Build nodes array of the coarse graph
    // - firstly, we count the degree of each coarse node
    // - secondly, we obtain the nodes array using a prefix sum
    //

    utils::Timer::instance().start_timer("build_indices_array", "Build indices array and node_volume array for the coarse graph");

    // we don't know the number of coarse edges yet, but there are hopefully much fewer than graph.m(); hence, we allocate
    // but don't initialize the memory

    tbb::enumerable_thread_specific<SparseMap<NodeID, ArcWeight>> collector{ [&] { return SparseMap<NodeID, ArcWeight>(coarse_graph._num_nodes); }};
    //tbb::enumerable_thread_specific<std::map<NodeID, ArcWeight>> collector{ [&] { return std::map<NodeID, ArcWeight>(); }};

    //
    // We build the coarse graph in multiple steps:
    // (1) During the first step, we compute
    //     - the node weight of each coarse node
    //     - the degree of each coarse node
    //     We can't build c_edges and c_edge_weights yet, because positioning edges in those arrays depends on c_nodes,
    //     which we only have after computing a prefix sum over all coarse node degrees
    //     Hence, we store edges and edge weights in unsorted auxiliary arrays during the first pass
    // (2) We finalize c_nodes arrays by computing a prefix sum over all coarse node degrees
    // (3) We copy coarse edges and coarse edge weights from the auxiliary arrays to c_edges and c_edge_weights
    //

    tbb::enumerable_thread_specific<size_t> local_max_degree(0);
    tbb::enumerable_thread_specific<LocalEdgeMemory> shared_edge_buffer;
    tbb::enumerable_thread_specific<std::vector<BufferNode>> shared_node_buffer;

    //for (NodeID c_u = 0; c_u < static_cast<NodeID>(coarse_graph._num_nodes); c_u ++) {
    tbb::parallel_for(static_cast<NodeID>(0), static_cast<NodeID>(coarse_graph._num_nodes), [&](const NodeID c_u) {
      auto &local_collector = collector.local();

      const std::size_t first = buckets_indices_prefix_sum[c_u + 1];
      const std::size_t last = buckets_indices_prefix_sum[c_u + 2];

      ASSERT(first <= buckets.size());
      ASSERT(last <= buckets.size());

      // we need an upper bound on the number of coarse edges to choose the right hash map -- sum all node degrees
      size_t upper_bound_degree = 0;
      for (std::size_t i = first; i < last; i++) {
        ASSERT(i <= buckets.size());
        const NodeID u = buckets[i];
        upper_bound_degree += degree(u);
      }
      upper_bound_degree = std::min(upper_bound_degree, coarse_graph._num_nodes);
      local_collector.setMaxSize(upper_bound_degree);

      // second pass over c_u bucket: compute actual degree, node weight, edges, edge weights
      ArcWeight c_u_weight = 0;
      for (std::size_t i = first; i < last; i++) {
        ASSERT(i <= buckets.size());
        const NodeID u = buckets[i];
        ASSERT(communities[u] == c_u);
        ASSERT(u < _num_nodes);

        c_u_weight += nodeVolume(u); // coarse node weight

        // collect coarse edges
        for (Arc e : arcsOf(u)) {
          ASSERT(e.head < _num_nodes);
          const NodeID c_v = communities[e.head];
          if (c_u != c_v) {
            ASSERT(c_v <= coarse_graph.numNodes() && c_v >= 0);
            local_collector[c_v] += e.weight;
          }
        }
      }

      coarse_node_volumes[c_u].store(c_u_weight);          // coarse node weights are done now
      tmp_indices[c_u + 1] = local_collector.size(); // node degree (used to build c_nodes)
      local_max_degree.local() = std::max(local_max_degree.local(), local_collector.size());

      // since we don't know the value of c_nodes[c_u] yet (so far, it only holds the nodes degree), we can't place the
      // edges of c_u in the c_edges and c_edge_weights arrays; hence, we store them in auxiliary arrays and note their
      // position in the auxiliary arrays
      auto &local_edge_buffer = shared_edge_buffer.local();
      auto &local_node_buffer = shared_node_buffer.local();
      const std::size_t position = local_edge_buffer.get_current_position();
      local_node_buffer.emplace_back(BufferNode{c_u, position, &local_edge_buffer});
      for (const auto[c_v, weight] : local_collector) {
        ASSERT(c_v <= coarse_graph.numNodes() && c_v >= 0);
        local_edge_buffer.push(c_v, weight);
      }
      local_collector.clear();
    });
    //}

    coarse_graph._max_degree = local_max_degree.combine(
      [&](const size_t& lhs, const size_t& rhs) {
        return std::max(lhs, rhs);
      });

    parallel::TBBPrefixSum<parallel::IntegralAtomicWrapper<size_t>, ds::Array> indices_prefix_sum(tmp_indices);
    tbb::parallel_scan(tbb::blocked_range<size_t>(0UL, coarse_graph._num_nodes + 1), indices_prefix_sum);

    coarse_graph._num_arcs = indices_prefix_sum.total_sum();

    utils::Timer::instance().stop_timer("build_indices_array");

    //
    // Construct rest of the coarse graph: edges, edge weights
    //

    utils::Timer::instance().start_timer("build_arcs_array", "Build arcs array for the coarse graph");
    parallel::scalable_vector<BufferNode> all_buffered_nodes(coarse_graph._num_nodes);
    parallel::IntegralAtomicWrapper<std::size_t> global_pos(0);

    // Move memory down to coarse graph
    coarse_graph._indices = std::move(_indices);
    coarse_graph._arcs = std::move(_arcs);
    coarse_graph._node_volumes = std::move(_node_volumes);

    tbb::parallel_invoke(
      [&] {
        tbb::parallel_for(shared_edge_buffer.range(), [&](auto &r) {
          for (auto &buffer : r) {
            if (!buffer.current_chunk.empty()) { buffer.flush(); }
          }
        });
      },
      [&] {
        tbb::parallel_for(shared_node_buffer.range(), [&](const auto &r) {
          for (const auto &buffer : r) {
            const std::size_t local_pos = global_pos.fetch_add(buffer.size());
            std::copy(buffer.begin(), buffer.end(), all_buffered_nodes.begin() + local_pos);
          }
        });
      },
      [&] {
        tbb::parallel_for(0U, static_cast<NodeID>(coarse_graph._num_nodes), [&](const NodeID u) {
          const size_t start_index_pos = indices_prefix_sum[u + 1];
          ASSERT(start_index_pos <= coarse_graph._num_arcs);
          coarse_graph._indices[u] = start_index_pos;
          coarse_graph._node_volumes[u] = coarse_node_volumes[u];
        });
      coarse_graph._indices[coarse_graph._num_nodes] = coarse_graph._num_arcs;
    });

    // build coarse graph
    tbb::parallel_for(static_cast<NodeID>(0), static_cast<NodeID>(coarse_graph._num_nodes), [&](const NodeID i) {
      const auto &buffered_node = all_buffered_nodes[i];
      const auto *chunks = buffered_node.chunks;
      const NodeID c_u = buffered_node.c_u;

      const size_t c_u_degree = indices_prefix_sum[c_u + 2] - indices_prefix_sum[c_u + 1];
      const NodeID first_target_index = indices_prefix_sum[c_u + 1];
      const NodeID first_source_index = buffered_node.position;

      for (std::size_t j = 0; j < c_u_degree; ++j) {
        const auto to = first_target_index + j;
        const auto [c_v, weight] = chunks->get(first_source_index + j);
        ASSERT(c_v <= coarse_graph.numNodes() && c_v >= 0);
        coarse_graph._arcs[to] = Arc(c_v, weight);
      }
    });

    coarse_graph._tmp_graph_buffer = _tmp_graph_buffer;
    _tmp_graph_buffer = nullptr;

    utils::Timer::instance().stop_timer("build_arcs_array");

    return coarse_graph;
  }

  Graph::Graph() :
          _num_nodes(0),
          _num_arcs(0),
          _total_volume(0),
          _max_degree(0),
          _indices(),
          _arcs(),
          _node_volumes(),
          _tmp_graph_buffer(nullptr) {

  }




  /*!
   * Constructs a graph from a given hypergraph.
   */
  template<typename F>
  void Graph::construct(const Hypergraph& hypergraph,
                 const F& edge_weight_func) {
    // Test, if hypergraph is actually a graph
    const bool is_graph = tbb::parallel_reduce(tbb::blocked_range<HyperedgeID>(
            ID(0), hypergraph.initialNumEdges()), true, [&](const tbb::blocked_range<HyperedgeID>& range, bool isGraph) {
      if ( isGraph ) {
        bool tmp_is_graph = isGraph;
        for (HyperedgeID he = range.begin(); he < range.end(); ++he) {
          if ( hypergraph.edgeIsEnabled(he) ) {
            tmp_is_graph &= (hypergraph.edgeSize(he) == 2);
          }
        }
        return tmp_is_graph;
      }
      return false;
    }, [&](const bool lhs, const bool rhs) {
      return lhs && rhs;
    });

    if ( is_graph ) {
      _num_nodes = hypergraph.initialNumNodes();
      _num_arcs = 2 * hypergraph.initialNumEdges();
      constructGraph(hypergraph, edge_weight_func);
    } else {
      _num_nodes = hypergraph.initialNumNodes() + hypergraph.initialNumEdges();
      _num_arcs = 2 * hypergraph.initialNumPins();
      constructBipartiteGraph(hypergraph, edge_weight_func);
    }

    // Compute node volumes and total volume
    utils::Timer::instance().start_timer("compute_node_volumes", "Compute Node Volumes");
    _total_volume = 0.0;
    tbb::enumerable_thread_specific<ArcWeight> local_total_volume(0.0);
    tbb::parallel_for(0U, static_cast<NodeID>(numNodes()), [&](const NodeID u) {
      local_total_volume.local() += computeNodeVolume(u);
    });
    _total_volume = local_total_volume.combine(std::plus<ArcWeight>());
    utils::Timer::instance().stop_timer("compute_node_volumes");
  }

  template<typename F>
  void Graph::constructBipartiteGraph(const Hypergraph& hypergraph,
                               F& edge_weight_func) {
    _indices.resize("Preprocessing", "indices", _num_nodes + 1);
    _arcs.resize("Preprocessing", "arcs", _num_arcs);
    _node_volumes.resize("Preprocessing", "node_volumes", _num_nodes);
    _tmp_graph_buffer = new TmpGraphBuffer(_num_nodes, _num_arcs);

    // Initialize data structure
    utils::Timer::instance().start_timer("compute_node_degrees", "Compute Node Degrees");
    const HypernodeID num_hypernodes = hypergraph.initialNumNodes();
    const HypernodeID num_hyperedges = hypergraph.initialNumEdges();
    tbb::parallel_invoke([&] {
      tbb::parallel_for(ID(0), num_hypernodes, [&](const HypernodeID u) {
        ASSERT(u + 1 < _indices.size());
        _indices[u + 1] = hypergraph.nodeDegree(u);
      });
    }, [&] {
      tbb::parallel_for(num_hypernodes, num_hypernodes + num_hyperedges, [&](const HyperedgeID u) {
        ASSERT(u + 1 < _indices.size());
        const HyperedgeID he = u - num_hypernodes;
        _indices[u + 1] = hypergraph.edgeSize(he);
      });
    });

    parallel::TBBPrefixSum<size_t, ds::Array> indices_prefix_sum(_indices);
    tbb::parallel_scan(tbb::blocked_range<size_t>(0UL, _indices.size()), indices_prefix_sum);
    utils::Timer::instance().stop_timer("compute_node_degrees");

    utils::Timer::instance().start_timer("construct_arcs", "Construct Arcs");
    tbb::enumerable_thread_specific<size_t> local_max_degree(0);
    tbb::parallel_invoke([&] {
      tbb::parallel_for(ID(0), num_hypernodes, [&](const HypernodeID u) {
        ASSERT(u + 1 < _indices.size());
        size_t pos = _indices[u];
        const HypernodeID hn = u;
        const HyperedgeID node_degree = hypergraph.nodeDegree(hn);
        local_max_degree.local() = std::max(
                local_max_degree.local(), static_cast<size_t>(node_degree));
        for ( const HyperedgeID& he : hypergraph.incidentEdges(hn) ) {
          const NodeID v = he + num_hypernodes;
          const HyperedgeWeight edge_weight = hypergraph.edgeWeight(he);
          const HypernodeID edge_size = hypergraph.edgeSize(he);
          ASSERT(pos < _indices[u + 1]);
          _arcs[pos++] = Arc(v, edge_weight_func(edge_weight, edge_size, node_degree));
        }
      });
    }, [&] {
      tbb::parallel_for(num_hypernodes, num_hypernodes + num_hyperedges, [&](const HyperedgeID u) {
        ASSERT(u + 1 < _indices.size());
        size_t pos = _indices[u];
        const HyperedgeID he = u - num_hypernodes;
        const HyperedgeWeight edge_weight = hypergraph.edgeWeight(he);
        const HypernodeID edge_size = hypergraph.edgeSize(he);
        local_max_degree.local() = std::max(
                local_max_degree.local(), static_cast<size_t>(edge_size));
        for ( const HypernodeID& pin : hypergraph.pins(he) ) {
          const NodeID v = pin;
          const HyperedgeID node_degree = hypergraph.nodeDegree(pin);
          ASSERT(pos < _indices[u + 1]);
          _arcs[pos++] = Arc(v, edge_weight_func(edge_weight, edge_size, node_degree));
        }
      });
    });
    _max_degree = local_max_degree.combine([&](const size_t& lhs, const size_t& rhs) {
      return std::max(lhs, rhs);
    });
    utils::Timer::instance().stop_timer("construct_arcs");
  }

  template<typename F>
  void Graph::constructGraph(const Hypergraph& hypergraph, const F& edge_weight_func) {
    _indices.resize("Preprocessing", "indices", _num_nodes + 1);
    _arcs.resize("Preprocessing", "arcs", _num_arcs);
    _node_volumes.resize("Preprocessing", "node_volumes", _num_nodes);
    _tmp_graph_buffer = new TmpGraphBuffer(_num_nodes, _num_arcs);

    // Initialize data structure
    utils::Timer::instance().start_timer("compute_node_degrees", "Compute Node Degrees");
    const HypernodeID num_hypernodes = hypergraph.initialNumNodes();
    tbb::parallel_for(ID(0), num_hypernodes, [&](const HypernodeID u) {
      ASSERT(u + 1 < _indices.size());
      _indices[u + 1] = hypergraph.nodeDegree(u);
    });

    parallel::TBBPrefixSum<size_t, ds::Array> indices_prefix_sum(_indices);
    tbb::parallel_scan(tbb::blocked_range<size_t>(0UL, num_hypernodes + 1), indices_prefix_sum);
    utils::Timer::instance().stop_timer("compute_node_degrees");

    utils::Timer::instance().start_timer("construct_arcs", "Construct Arcs");
    tbb::enumerable_thread_specific<size_t> local_max_degree(0);
    tbb::parallel_for(ID(0), num_hypernodes, [&](const HypernodeID u) {
      ASSERT(u + 1 < _indices.size());
      size_t pos = _indices[u];
      const HypernodeID hn = u;
      const HyperedgeID node_degree = hypergraph.nodeDegree(hn);
      local_max_degree.local() = std::max(
              local_max_degree.local(), static_cast<size_t>(node_degree));
      for ( const HyperedgeID& he : hypergraph.incidentEdges(hn) ) {
        const HyperedgeWeight edge_weight = hypergraph.edgeWeight(he);
        NodeID v = std::numeric_limits<NodeID>::max();
        for ( const HypernodeID& pin : hypergraph.pins(he) ) {
          if ( pin != hn ) {
            v = pin;
            break;
          }
        }
        ASSERT(v != std::numeric_limits<NodeID>::max());
        ASSERT(pos < _indices[u + 1]);
        _arcs[pos++] = Arc(v, edge_weight_func(edge_weight, ID(2), node_degree));
      }
    });
    _max_degree = local_max_degree.combine([&](const size_t& lhs, const size_t& rhs) {
      return std::max(lhs, rhs);
    });
    utils::Timer::instance().stop_timer("construct_arcs");
  }

  bool Graph::canBeUsed(const bool verbose) const {
    const bool result = _indices.size() >= numNodes() + 1 && _arcs.size() >= numArcs() && _node_volumes.size() >= numNodes();
    if (verbose && !result) {
      LOG << "Some of the graph's members were stolen. For example the contract function does this. "
             "Make sure you're calling functions with a fresh graph or catch this condition and reinitialize."
             "If you do reinitialize, feel free to silence this warning by passing false to the canBeUsed function";
    }
    return result;
  }

} // namespace mt_kahypar::ds