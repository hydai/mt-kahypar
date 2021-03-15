
#pragma once

#include "external_tools/kahypar/external_tools/WHFC/datastructure/flow_hypergraph_builder.h"
#include "external_tools/kahypar/kahypar/datastructure/fast_reset_flag_array.h"

namespace mt_kahypar::community_detection {

  class HyperFlowInstance {

  public:
    static constexpr HypernodeID invalid_node = std::numeric_limits<HypernodeID>::max();
    std::vector<HypernodeID> _core;

    HyperFlowInstance(Hypergraph& hg, const Context& context, HypernodeID v) :
      //_flow_hg_builder(hg.initialNumNodes(), hg.initialNumEdges(), hg.initialNumPins()),
      _nodeIDMap(hg.initialNumNodes() + 2, whfc::invalidNode),
      //_edgeIDMap(hg.initialNumEdges() + 2, std::numeric_limits<HyperedgeID>::max()),
      _visitedNode(hg.initialNumNodes()),
      _visitedHyperedge(hg.initialNumEdges())
      {
        constructFlowgraphFromSourceNode(hg, v, context);
      }


    std::vector<HyperedgeID> computeCut();

  private:
    HypernodeID _globalSourceID = invalid_node, _globalTargetID = invalid_node;
    whfc::FlowHypergraphBuilder _flow_hg_builder;
    std::vector<HyperedgeID> _edgeIDMap;
    std::vector<whfc::Node> _nodeIDMap;
    kahypar::ds::FastResetFlagArray<> _visitedNode;
    kahypar::ds::FastResetFlagArray<> _visitedHyperedge;

    void constructFlowgraphFromSourceNode(const Hypergraph& hg, const HypernodeID v,const Context& context){
      size_t U = hg.initialNumNodes() / context.partition.k;
      std::pair<size_t,size_t> sizes =  initializeFlowHGBuilder(hg, v, U);
      BreathFirstSearch(hg, v, sizes.first, sizes.second, U);
    }

    std::pair<size_t,size_t> initializeFlowHGBuilder(const Hypergraph& hg, const HypernodeID start, size_t U);

    void BreathFirstSearch(const Hypergraph& hg, const HypernodeID start, size_t coreSize, size_t ringSize, size_t U);

  };
}


