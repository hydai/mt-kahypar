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

#pragma once

#include <atomic>
#include <type_traits>
#include <mutex>

#include "tbb/parallel_invoke.h"

#include "kahypar/meta/mandatory.h"

#include "mt-kahypar/datastructures/hypergraph_common.h"
#include "mt-kahypar/datastructures/connectivity_set.h"
#include "mt-kahypar/parallel/atomic_wrapper.h"
#include "mt-kahypar/parallel/stl/scalable_vector.h"
#include "mt-kahypar/parallel/stl/thread_locals.h"
#include "mt-kahypar/utils/range.h"
#include "mt-kahypar/utils/timer.h"

namespace mt_kahypar {
namespace ds {

template <typename Hypergraph = Mandatory,
          typename HypergraphFactory = Mandatory>
class PartitionedGraph {
private:
  static_assert(!Hypergraph::is_partitioned,  "Only unpartitioned hypergraphs are allowed");

  // ! Function that will be called for each incident hyperedge of a moved vertex with the following arguments
  // !  1) hyperedge ID, 2) weight, 3) size, 4) pin count in from-block after move, 5) pin count in to-block after move
  // ! Can be implemented to obtain correct km1 or cut improvements of the move
  using DeltaFunction = std::function<void (const HyperedgeID, const HyperedgeWeight, const HypernodeID, const HypernodeID, const HypernodeID)>;
  #define NOOP_FUNC [] (const HyperedgeID, const HyperedgeWeight, const HypernodeID, const HypernodeID, const HypernodeID) { }

  static constexpr bool enable_heavy_assert = false;

  class ConnectivityIterator :
    public std::iterator<std::forward_iterator_tag,    // iterator_category
                         PartitionID,   // value_type
                         std::ptrdiff_t,   // difference_type
                         const PartitionID*,   // pointer
                         PartitionID> {
   public:
    /*!
     * Constructs a connectivity iterator based on a pin iterator 
     */
    ConnectivityIterator(typename Hypergraph::IncidenceIterator it_begin, const PartitionedGraph& graph, unsigned int count) :
      _first(0),
      _second(0),
      _iteration_count(count) {
        _first = graph.partID(*it_begin);
        it_begin++;
        _second = graph.partID(*it_begin);
        if (_first == _second && count == 0) {
          _iteration_count++;
        }
    }

    // ! Returns the current partiton id.
    PartitionID operator* () const {
      ASSERT(_iteration_count < 2);
      return _iteration_count == 0 ? _first : _second;
    }

    // ! Prefix increment. The iterator advances to the next valid element.
    ConnectivityIterator & operator++ () {
      ASSERT(_iteration_count < 2);
      ++_iteration_count;
      return *this;
    }

    // ! Postfix increment. The iterator advances to the next valid element.
    ConnectivityIterator operator++ (int) {
      ConnectivityIterator copy = *this;
      operator++ ();
      return copy;
    }

    bool operator!= (const ConnectivityIterator& rhs) {
      return _first != rhs._first || _second != rhs._second ||
             _iteration_count != rhs._iteration_count;
    }

    bool operator== (const ConnectivityIterator& rhs) {
      return _first == rhs._first && _second == rhs._second &&
             _iteration_count == rhs._iteration_count;
    }
    

   private:
    PartitionID _first = 0;
    PartitionID _second = 0;
    // state of the iterator
    unsigned int _iteration_count = 0;
  };

 public:
  static constexpr bool is_static_hypergraph = Hypergraph::is_static_hypergraph;
  static constexpr bool is_partitioned = true;
  static constexpr bool supports_connectivity_set = true;

  static constexpr HyperedgeID HIGH_DEGREE_THRESHOLD = ID(100000);

  using HypernodeIterator = typename Hypergraph::HypernodeIterator;
  using HyperedgeIterator = typename Hypergraph::HyperedgeIterator;
  using IncidenceIterator = typename Hypergraph::IncidenceIterator;
  using IncidentNetsIterator = typename Hypergraph::IncidentNetsIterator;

  PartitionedGraph() = default;

  explicit PartitionedGraph(const PartitionID k,
                            Hypergraph& hypergraph) :
    _is_gain_cache_initialized(false),
    _k(k),
    _hg(&hypergraph),
    _part_weights(k, CAtomic<HypernodeWeight>(0)),
    _part_ids(
        "Refinement", "part_ids", hypergraph.initialNumNodes(), false, false),
    _incident_weight_in_part(
        "Refinement", "incident_weight_in_part",
        static_cast<size_t>(hypergraph.initialNumNodes()) * static_cast<size_t>(k), true, false) {
    _part_ids.assign(hypergraph.initialNumNodes(), kInvalidPartition, false);
  }

  explicit PartitionedGraph(const PartitionID k,
                            const TaskGroupID,
                            Hypergraph& hypergraph) :
    _is_gain_cache_initialized(false),
    _k(k),
    _hg(&hypergraph),
    _part_weights(k, CAtomic<HypernodeWeight>(0)),
    _part_ids(),
    _incident_weight_in_part() {
    tbb::parallel_invoke([&] {
      _part_ids.resize(
        "Refinement", "vertex_part_info", hypergraph.initialNumNodes());
      _part_ids.assign(hypergraph.initialNumNodes(), kInvalidPartition);
    }, [&] {
      _incident_weight_in_part.resize(
        "Refinement", "incident_weight_in_part",
        static_cast<size_t>(hypergraph.initialNumNodes()) * static_cast<size_t>(k), true);
    });
  }

  PartitionedGraph(const PartitionedGraph&) = delete;
  PartitionedGraph & operator= (const PartitionedGraph &) = delete;

  PartitionedGraph(PartitionedGraph&& other) = default;
  PartitionedGraph & operator= (PartitionedGraph&& other) = default;

  ~PartitionedGraph() {
    freeInternalData();
  }

  // ####################### General Hypergraph Stats ######################

  Hypergraph& hypergraph() {
    ASSERT(_hg);
    return *_hg;
  }

  void setHypergraph(Hypergraph& hypergraph) {
    _hg = &hypergraph;
  }

  // ! Initial number of hypernodes
  HypernodeID initialNumNodes() const {
    return _hg->initialNumNodes();
  }

  // ! Number of removed hypernodes
  HypernodeID numRemovedHypernodes() const {
    return _hg->numRemovedHypernodes();
  }

  // ! Initial number of hyperedges
  HyperedgeID initialNumEdges() const {
    return _hg->initialNumEdges();
  }

  // ! Initial number of pins
  HypernodeID initialNumPins() const {
    return _hg->initialNumPins();
  }

  // ! Initial sum of the degree of all vertices
  HypernodeID initialTotalVertexDegree() const {
    return _hg->initialTotalVertexDegree();
  }

  // ! Total weight of hypergraph
  HypernodeWeight totalWeight() const {
    return _hg->totalWeight();
  }

  // ! Number of blocks this hypergraph is partitioned into
  PartitionID k() const {
    return _k;
  }


  // ####################### Iterators #######################

  // ! Iterates in parallel over all active nodes and calls function f
  // ! for each vertex
  template<typename F>
  void doParallelForAllNodes(const F& f) const {
    _hg->doParallelForAllNodes(f);
  }

  // ! Iterates in parallel over all active edges and calls function f
  // ! for each net
  template<typename F>
  void doParallelForAllEdges(const F& f) const {
    _hg->doParallelForAllEdges(f);
  }

  // ! Returns an iterator over the set of active nodes of the hypergraph
  IteratorRange<HypernodeIterator> nodes() const {
    return _hg->nodes();
  }

  // ! Returns an iterator over the set of active edges of the hypergraph
  IteratorRange<HyperedgeIterator> edges() const {
    return _hg->edges();
  }

  // ! Returns a range to loop over the incident nets of hypernode u.
  IteratorRange<IncidentNetsIterator> incidentEdges(const HypernodeID u) const {
    return _hg->incidentEdges(u);
  }

  // ! Returns a range to loop over the pins of hyperedge e.
  IteratorRange<IncidenceIterator> pins(const HyperedgeID e) const {
    return _hg->pins(e);
  }

  // ! Returns a range to loop over the set of block ids contained in hyperedge e.
  IteratorRange<ConnectivityIterator> connectivitySet(const HyperedgeID e) const {
    ASSERT(_hg->edgeIsEnabled(e), "Hyperedge" << e << "is disabled");
    ASSERT(e < _hg->initialNumEdges(), "Hyperedge" << e << "does not exist");
    IncidenceIterator pin_it = _hg->pins(e).begin();
    return IteratorRange<ConnectivityIterator>(
      ConnectivityIterator(pin_it, *this, 0),
      ConnectivityIterator(pin_it, *this, 2));
  }

  // ####################### Hypernode Information #######################

  // ! Weight of a vertex
  HypernodeWeight nodeWeight(const HypernodeID u) const {
    return _hg->nodeWeight(u);
  }

  // ! Sets the weight of a vertex
  void setNodeWeight(const HypernodeID u, const HypernodeWeight weight) {
    const PartitionID block = partID(u);
    if ( block != kInvalidPartition ) {
      ASSERT(block < _k);
      const HypernodeWeight delta = weight - _hg->nodeWeight(u);
      _part_weights[block] += delta;
    }
    _hg->setNodeWeight(u, weight);
  }

  // ! Degree of a hypernode
  HyperedgeID nodeDegree(const HypernodeID u) const {
    return _hg->nodeDegree(u);
  }

  // ! Returns, whether a hypernode is enabled or not
  bool nodeIsEnabled(const HypernodeID u) const {
    return _hg->nodeIsEnabled(u);
  }

  // ! Restores a degree zero hypernode
  void restoreDegreeZeroHypernode(const HypernodeID u, const PartitionID to) {
    _hg->restoreDegreeZeroHypernode(u);
    setNodePart(u, to);
  }

  // ####################### Hyperedge Information #######################

  // ! Target of an edge
  HypernodeID edgeTarget(const HyperedgeID e) const {
    return _hg->edgeTarget(e);
  }

  // ! Target of an edge
  HypernodeID edgeSource(const HyperedgeID e) const {
    return _hg->edgeSource(e);
  }

  // ! Weight of a hyperedge
  HypernodeWeight edgeWeight(const HyperedgeID e) const {
    return _hg->edgeWeight(e);
  }

  // ! Sets the weight of a hyperedge
  void setEdgeWeight(const HyperedgeID e, const HyperedgeWeight weight) {
    _hg->setEdgeWeight(e, weight);
  }

  // ! Number of pins of a hyperedge
  HypernodeID edgeSize(const HyperedgeID e) const {
    return _hg->edgeSize(e);
  }

  // ! Returns, whether a hyperedge is enabled or not
  bool edgeIsEnabled(const HyperedgeID e) const {
    return _hg->edgeIsEnabled(e);
  }

  bool isGraphEdge(const HyperedgeID e) const {
    return _hg->isGraphEdge(e);
  }

  HyperedgeID graphEdgeID(const HyperedgeID e) const {
    return _hg->graphEdgeID(e);
  }

  HyperedgeID nonGraphEdgeID(const HyperedgeID e) const {
    return _hg->nonGraphEdgeID(e);
  }

  HypernodeID graphEdgeHead(const HyperedgeID e, const HypernodeID tail) const {
    return _hg->graphEdgeHead(e, tail);
  }

  // ####################### Uncontraction #######################

  void uncontract(const Batch& batch) {
    ERROR("uncontract(batch) is not supported in static graph");
  }

  // ####################### Restore Hyperedges #######################

  void restoreLargeEdge(const HyperedgeID& he) {
    ERROR("restoreLargeEdge(he) is not supported in static graph");
  }

  void restoreSinglePinAndParallelNets(const parallel::scalable_vector<ParallelHyperedge>& hes_to_restore) {
    ERROR("restoreSinglePinAndParallelNets(hes_to_restore) is not supported in static graph");
  }

  // ####################### Partition Information #######################

  // ! Block that vertex u belongs to
  PartitionID partID(const HypernodeID u) const {
    ASSERT(u < initialNumNodes(), "Hypernode" << u << "does not exist");
    return _part_ids[u];
  }

  void setOnlyNodePart(const HypernodeID u, PartitionID p) {
    ASSERT(p != kInvalidPartition && p < _k);
    ASSERT(_part_ids[u] == kInvalidPartition);
    _part_ids[u] = p;
  }

  void setNodePart(const HypernodeID u, PartitionID p) {
    setOnlyNodePart(u, p);
    _part_weights[p].fetch_add(nodeWeight(u), std::memory_order_relaxed);
  }

  // ! Changes the block id of vertex u from block 'from' to block 'to'
  // ! Returns true, if move of vertex u to corresponding block succeeds.
  template<typename SuccessFunc, typename DeltaFunc>
  bool changeNodePart(const HypernodeID u,
                      PartitionID from,
                      PartitionID to,
                      HypernodeWeight max_weight_to,
                      SuccessFunc&& report_success,
                      DeltaFunc&& delta_func) {
    ASSERT(partID(u) == from);
    ASSERT(from != to);
    _part_ids[u] = to;
    _part_weights[from].fetch_sub(nodeWeight(u), std::memory_order_relaxed);
    _part_weights[to].fetch_add(nodeWeight(u), std::memory_order_relaxed);
    return true;
    // TODO

    // const HypernodeWeight wu = nodeWeight(u);
    // const HypernodeWeight to_weight_after = _part_weights[to].add_fetch(wu, std::memory_order_relaxed);
    // const HypernodeWeight from_weight_after = _part_weights[from].fetch_sub(wu, std::memory_order_relaxed);
    // if ( to_weight_after <= max_weight_to && from_weight_after > 0 ) {
    //   _part_ids[u] = to;
    //   report_success();
    //   for ( const HyperedgeID he : incidentEdges(u) ) {
    //     // updatePinCountOfHyperedge(he, from, to, delta_func);
    //   }
    //   return true;
    // } else {
    //   _part_weights[to].fetch_sub(wu, std::memory_order_relaxed);
    //   _part_weights[from].fetch_add(wu, std::memory_order_relaxed);
    //   return false;
    // }
  }

  bool changeNodePart(const HypernodeID u,
                      PartitionID from,
                      PartitionID to,
                      const DeltaFunction& delta_func = NOOP_FUNC) {
    return changeNodePart(u, from, to,
      std::numeric_limits<HypernodeWeight>::max(), []{}, delta_func);
  }

  // Make sure not to call phg.gainCacheUpdate(..) in delta_func for changeNodePartWithGainCacheUpdate
  template<typename SuccessFunc, typename DeltaFunc>
  bool changeNodePartWithGainCacheUpdate(const HypernodeID u,
                                         PartitionID from,
                                         PartitionID to,
                                         HypernodeWeight max_weight_to,
                                         SuccessFunc&& report_success,
                                         DeltaFunc&& delta_func) {
    ASSERT(_is_gain_cache_initialized, "Gain cache is not initialized");
    ASSERT(false, "TODO"); return false;

    // auto my_delta_func = [&](const HyperedgeID he, const HyperedgeWeight edge_weight, const HypernodeID edge_size,
    //         const HypernodeID pin_count_in_from_part_after, const HypernodeID pin_count_in_to_part_after) {
    //   delta_func(he, edge_weight, edge_size, pin_count_in_from_part_after, pin_count_in_to_part_after);
    //   gainCacheUpdate(he, edge_weight, from, pin_count_in_from_part_after, to, pin_count_in_to_part_after);
    // };
    // return changeNodePart(u, from, to, max_weight_to, report_success, my_delta_func);

  }

  bool changeNodePartWithGainCacheUpdate(const HypernodeID u, PartitionID from, PartitionID to) {
    return changeNodePartWithGainCacheUpdate(u, from, to, std::numeric_limits<HypernodeWeight>::max(), [] { },
                                             NoOpDeltaFunc());
  }

  // ! Weight of a block
  HypernodeWeight partWeight(const PartitionID p) const {
    ASSERT(p != kInvalidPartition && p < _k);
    return _part_weights[p].load(std::memory_order_relaxed);
  }

  // ! Returns, whether hypernode u is adjacent to a least one cut hyperedge.
  bool isBorderNode(const HypernodeID u) const {
    const PartitionID part_id = partID(u);
    if ( nodeDegree(u) <= HIGH_DEGREE_THRESHOLD ) {
      for ( const HyperedgeID& he : incidentEdges(u) ) {
        if ( partID(edgeTarget(he)) != part_id ) {
          return true;
        }
      }
      return false;
    } else {
      // TODO maybe we should allow these in label propagation? definitely not in FM
      // In case u is a high degree vertex, we omit the border node check and
      // and return false. Assumption is that it is very unlikely that such a
      // vertex can change its block.
      return false;
    }
  }

  HypernodeID numIncidentCutHyperedges(const HypernodeID u) const {
    const PartitionID part_id = partID(u);
    HypernodeID num_incident_cut_hyperedges = 0;
    for ( const HyperedgeID& he : incidentEdges(u) ) {
      if ( partID(edgeTarget(he)) != part_id ) {
        ++num_incident_cut_hyperedges;
      }
    }
    return num_incident_cut_hyperedges;
  }

  // ! Number of blocks which pins of hyperedge e belongs to
  PartitionID connectivity(const HyperedgeID e) const {
    ASSERT(e < _hg->initialNumEdges(), "Hyperedge" << e << "does not exist");
    ASSERT(edgeIsEnabled(e), "Hyperedge" << e << "is disabled");
    PartitionID source_id = partID(edgeSource(e));
    PartitionID target_id = partID(edgeTarget(e));
    ASSERT(source_id != kInvalidPartition && target_id != kInvalidPartition);
    return source_id == target_id ? 1 : 2;
  }

  // ! Returns the number pins of hyperedge e that are part of block id
  HypernodeID pinCountInPart(const HyperedgeID e, const PartitionID p) const {
    ASSERT(e < _hg->initialNumEdges(), "Hyperedge" << e << "does not exist");
    ASSERT(edgeIsEnabled(e), "Hyperedge" << e << "is disabled");
    ASSERT(p != kInvalidPartition && p < _k);
    HypernodeID count = 0;
    if (p == partID(edgeSource(e))) {
      count++;
    }
    if (p == partID(edgeTarget(e))) {
      count++;
    }
    return count;
  }

  HyperedgeWeight moveFromBenefit(const HypernodeID u) const {
    ASSERT(_is_gain_cache_initialized, "Gain cache is not initialized");
    return 0;
  }

  HyperedgeWeight moveToPenalty(const HypernodeID u, PartitionID p) const {
    ASSERT(_is_gain_cache_initialized, "Gain cache is not initialized");
    return _incident_weight_in_part[incident_weight_index(u, partID(u))].load(std::memory_order_relaxed) -
           _incident_weight_in_part[incident_weight_index(u, p)].load(std::memory_order_relaxed);
  }

  void initializeGainCacheEntry(const HypernodeID u, vec<Gain>& penalty_aggregator) {
    for (HyperedgeID e : incidentEdges(u)) {
      penalty_aggregator[partID(edgeTarget(e))] += edgeWeight(e);
    }
    
    for (PartitionID i = 0; i < _k; ++i) {
      _incident_weight_in_part[incident_weight_index(u, i)].store(
        penalty_aggregator[i], std::memory_order_relaxed);
      penalty_aggregator[i] = 0;
    }
  }

  HyperedgeWeight km1Gain(const HypernodeID u, PartitionID from, PartitionID to) const {
    unused(from);
    ASSERT(_is_gain_cache_initialized, "Gain cache is not initialized");
    ASSERT(from == partID(u), "While gain computation works for from != partID(u), such a query makes no sense");
    ASSERT(from != to, "The gain computation doesn't work for from = to");
    return moveFromBenefit(u) - moveToPenalty(u, to);
  }

  // ! Initializes the partition of the hypergraph, if block ids are assigned with
  // ! setOnlyNodePart(...). In that case, block weights must be initialized explicitly here.
  void initializePartition(const TaskGroupID) {
    initializeBlockWeights();
  }

  bool isGainCacheInitialized() const {
    return _is_gain_cache_initialized;
  }

  // ! Initialize gain cache
  void initializeGainCache() {
    // assert that part has been initialized
    ASSERT( _part_ids.size() == initialNumNodes()
            && std::none_of(nodes().begin(), nodes().end(),
                            [&](HypernodeID u) { return partID(u) == kInvalidPartition || partID(u) > k(); }) );
    // assert that current gain values are zero
    ASSERT(!_is_gain_cache_initialized
           && std::none_of(_incident_weight_in_part.begin(), _incident_weight_in_part.end(),
                           [&](const auto& weight) { return weight.load() != 0; }));

    // Calculate gain in parallel over all edges. Note that because the edges
    // are grouped by source node, this is still cache-efficient.
    // TODO(maas): is this less efficient than iteration over nodes?
    tbb::parallel_for(ID(0), _hg->initialNumEdges(), [&](const HyperedgeID e) {
      HypernodeID node = edgeSource(e);
      if (nodeIsEnabled(node)) {
        size_t index = incident_weight_index(node, partID(edgeTarget(e)));
        _incident_weight_in_part[index].fetch_add(edgeWeight(e), std::memory_order_relaxed);
      }
    });

    _is_gain_cache_initialized = true;
  }

  // ! Reset partition (not thread-safe)
  void resetPartition() {
    _part_ids.assign(_part_ids.size(), kInvalidPartition, false);
    for (auto& weight : _part_weights) {
      weight.store(0, std::memory_order_relaxed);
    }
  }

  // ! Only for testing
  void recomputePartWeights() {
    for (PartitionID p = 0; p < _k; ++p) {
      _part_weights[p].store(0);
    }

    for (HypernodeID u : nodes()) {
      _part_weights[ partID(u) ] += nodeWeight(u);
    }
  }

  // ! Only for testing
  HyperedgeWeight moveFromBenefitRecomputed(const HypernodeID u) const {
    return 0;
  }

  // ! Only for testing
  HyperedgeWeight moveToPenaltyRecomputed(const HypernodeID u, PartitionID p) const {
    PartitionID part_id = partID(u);
    HyperedgeWeight w = 0;
    for (HyperedgeID e : incidentEdges(u)) {
      if (edgeTarget(e) == part_id) {
        w += edgeWeight(e);
      } else if (edgeTarget(e) == p) {
        w -= edgeWeight(e);
      }
    }
    return w;
  }

  void recomputeMoveFromBenefit(const HypernodeID u) {
    // Nothing to do here
  }

  // ! Only for testing
  bool checkTrackedPartitionInformation() {
    bool success = true;

    for (HyperedgeID e : edges()) {
      PartitionID expected_connectivity = 0;
      for (PartitionID i = 0; i < k(); ++i) {
        expected_connectivity += (pinCountInPart(e, i) > 0);
      }
      if ( expected_connectivity != connectivity(e) ) {
        LOG << "Connectivity of hyperedge" << e << "=>" <<
            "Expected:" << V(expected_connectivity)  << "," <<
            "Actual:" << V(connectivity(e));
        success = false;
      }
    }

    if ( _is_gain_cache_initialized ) {
      for (HypernodeID u : nodes()) {
        if ( moveFromBenefit(u) != moveFromBenefitRecomputed(u) ) {
          LOG << "Move from benefit of hypernode" << u << "=>" <<
              "Expected:" << V(moveFromBenefitRecomputed(u)) << ", " <<
              "Actual:" <<  V(moveFromBenefit(u));
          success = false;
        }

        for (PartitionID i = 0; i < k(); ++i) {
          if (partID(u) != i) {
            if ( moveToPenalty(u, i) != moveToPenaltyRecomputed(u, i) ) {
              LOG << "Move to penalty of hypernode" << u << "in block" << i << "=>" <<
                  "Expected:" << V(moveToPenaltyRecomputed(u, i)) << ", " <<
                  "Actual:" <<  V(moveToPenalty(u, i));
              success = false;
            }
          }
        }
      }
    }
    return success;
  }

  // ####################### Memory Consumption #######################
  
  void memoryConsumption(utils::MemoryTreeNode* parent) const {
    ASSERT(parent);

    utils::MemoryTreeNode* hypergraph_node = parent->addChild("Hypergraph");
    _hg->memoryConsumption(hypergraph_node);

    parent->addChild("Part Weights", sizeof(CAtomic<HypernodeWeight>) * _k);
    parent->addChild("Part IDs", sizeof(PartitionID) * _hg->initialNumNodes());
    parent->addChild("Incident Weight in Part", sizeof(CAtomic<HyperedgeWeight>) * _incident_weight_in_part.size());
  }

  // ####################### Extract Block #######################

  // TODO
  // ! Extracts a block of a partition as separate hypergraph.
  // ! It also returns a vertex-mapping from the original hypergraph to the sub-hypergraph.
  // ! If cut_net_splitting is activated, hyperedges that span more than one block (cut nets) are split, which is used for the connectivity metric.
  // ! Otherwise cut nets are discarded (cut metric).
  std::pair<Hypergraph, parallel::scalable_vector<HypernodeID> > extract(const TaskGroupID& task_group_id, PartitionID block, bool cut_net_splitting) {
    ASSERT(block != kInvalidPartition && block < _k);

    // Compactify vertex ids
    parallel::scalable_vector<HypernodeID> hn_mapping(_hg->initialNumNodes(), kInvalidHypernode);
    parallel::scalable_vector<HyperedgeID> he_mapping(_hg->initialNumEdges(), kInvalidHyperedge);
    HypernodeID num_hypernodes = 0;
    HypernodeID num_hyperedges = 0;
    tbb::parallel_invoke([&] {
      for ( const HypernodeID& hn : nodes() ) {
        if ( partID(hn) == block ) {
          hn_mapping[hn] = num_hypernodes++;
        }
      }
    }, [&] {
      for ( const HyperedgeID& he : edges() ) {
        if ( pinCountInPart(he, block) > 1 &&
             (cut_net_splitting || connectivity(he) == 1) ) {
          he_mapping[he] = num_hyperedges++;
        }
      }
    });

    // Extract plain hypergraph data for corresponding block
    using HyperedgeVector = parallel::scalable_vector<parallel::scalable_vector<HypernodeID>>;
    HyperedgeVector edge_vector;
    parallel::scalable_vector<HyperedgeWeight> hyperedge_weight;
    parallel::scalable_vector<HypernodeWeight> hypernode_weight;
    tbb::parallel_invoke([&] {
      edge_vector.resize(num_hyperedges);
      hyperedge_weight.resize(num_hyperedges);
      doParallelForAllEdges([&](const HyperedgeID he) {
        if ( pinCountInPart(he, block) > 1 &&
             (cut_net_splitting || connectivity(he) == 1) ) {
          ASSERT(he_mapping[he] < num_hyperedges);
          hyperedge_weight[he_mapping[he]] = edgeWeight(he);
          for ( const HypernodeID& pin : pins(he) ) {
            if ( partID(pin) == block ) {
              edge_vector[he_mapping[he]].push_back(hn_mapping[pin]);
            }
          }
        }
      });
    }, [&] {
      hypernode_weight.resize(num_hypernodes);
      doParallelForAllNodes([&](const HypernodeID hn) {
        if ( partID(hn) == block ) {
          hypernode_weight[hn_mapping[hn]] = nodeWeight(hn);
        }
      });
    });

    // Construct hypergraph
    Hypergraph extracted_hypergraph = HypergraphFactory::construct(
            task_group_id, num_hypernodes, num_hyperedges,
            edge_vector, hyperedge_weight.data(), hypernode_weight.data());

    // Set community ids
    doParallelForAllNodes([&](const HypernodeID& hn) {
      if ( partID(hn) == block ) {
        const HypernodeID extracted_hn = hn_mapping[hn];
        extracted_hypergraph.setCommunityID(extracted_hn, _hg->communityID(hn));
      }
    });
    return std::make_pair(std::move(extracted_hypergraph), std::move(hn_mapping));
  }

  void freeInternalData() {
    if ( _k > 0 ) {
      parallel::parallel_free(_part_ids, _incident_weight_in_part);
    }
    _k = 0;
  }


  MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE
  void gainCacheUpdate(const HyperedgeID he, const HyperedgeWeight we,
                       const PartitionID from, const HypernodeID pin_count_in_from_part_after,
                       const PartitionID to, const HypernodeID pin_count_in_to_part_after) {
    ASSERT(false, "TODO");
    // if (pin_count_in_from_part_after == 1) {
    //   for (HypernodeID u : pins(he)) {
    //     nodeGainAssertions(u, from);
    //     if (partID(u) == from) {
    //       _move_from_benefit[u].fetch_add(we, std::memory_order_relaxed);
    //     }
    //   }
    // } else if (pin_count_in_from_part_after == 0) {
    //   for (HypernodeID u : pins(he)) {
    //     nodeGainAssertions(u, from);
    //     _move_to_penalty[penalty_index(u, from)].fetch_sub(we, std::memory_order_relaxed);
    //   }
    // }

    // if (pin_count_in_to_part_after == 1) {
    //   for (HypernodeID u : pins(he)) {
    //     nodeGainAssertions(u, to);
    //     _move_to_penalty[penalty_index(u, to)].fetch_add(we, std::memory_order_relaxed);
    //   }
    // } else if (pin_count_in_to_part_after == 2) {
    //   for (HypernodeID u : pins(he)) {
    //     nodeGainAssertions(u, to);
    //     if (partID(u) == to) {
    //       _move_from_benefit[u].fetch_sub(we, std::memory_order_relaxed);
    //     }
    //   }
    // }
  }

 private:
  MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE
  size_t incident_weight_index(const HypernodeID u, const PartitionID p) const {
    return size_t(u) * _k  + p;
  }

  void initializeBlockWeights() {
    tbb::parallel_for(tbb::blocked_range<HypernodeID>(HypernodeID(0), initialNumNodes()),
      [&](tbb::blocked_range<HypernodeID>& r) {
        // this is not enumerable_thread_specific because of the static partitioner
        vec<HypernodeWeight> part_weight_deltas(_k, 0);
        for (HypernodeID node = r.begin(); node < r.end(); ++node) {
          if (nodeIsEnabled(node)) {
            part_weight_deltas[partID(node)] += nodeWeight(node);
          }
        }
        for (PartitionID p = 0; p < _k; ++p) {
          _part_weights[p].fetch_add(part_weight_deltas[p], std::memory_order_relaxed);
        }
      },
      tbb::static_partitioner()
    );
  }

  // void nodeGainAssertions(const HypernodeID u, const PartitionID p) const {
  //   unused(u);
  //   unused(p);
  //   ASSERT(u < initialNumNodes(), "Hypernode" << u << "does not exist");
  //   ASSERT(nodeIsEnabled(u), "Hypernode" << u << "is disabled");
  //   ASSERT(p != kInvalidPartition && p < _k);
  //   ASSERT(penalty_index(u, p) < _move_to_penalty.size());
  //   ASSERT(u < _move_from_benefit.size());
  // }


  // ! Indicate wheater gain cache is initialized
  bool _is_gain_cache_initialized;

  // ! Number of blocks
  PartitionID _k = 0;

  // ! Hypergraph object around which this partitioned hypergraph is wrapped
  Hypergraph* _hg = nullptr;

  // ! Weight and information for all blocks.
  vec< CAtomic<HypernodeWeight> > _part_weights;

  // ! Current block IDs of the vertices
  Array< PartitionID > _part_ids;

  // ! For each node and block, the sum of incident edge weights where the target is in that part
  Array< CAtomic<HyperedgeWeight> > _incident_weight_in_part;
};

} // namespace ds
} // namespace mt_kahypar