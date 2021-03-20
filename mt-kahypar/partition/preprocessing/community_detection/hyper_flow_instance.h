
#pragma once

#include "external_tools/kahypar/external_tools/WHFC/datastructure/flow_hypergraph_builder.h"
#include "external_tools/kahypar/kahypar/datastructure/fast_reset_flag_array.h"

namespace mt_kahypar::community_detection {

  class HyperFlowInstance {

  public:
    static constexpr HypernodeID invalid_node = std::numeric_limits<HypernodeID>::max();
    std::vector<HypernodeID> _core;

    HyperFlowInstance(Hypergraph& hg, const Context& context, HypernodeID v) :
      _flow_hg_builder(),
      _nodeIDMap(hg.initialNumNodes() + 2, whfc::invalidNode),
      _edgeIDMap(0),
      _visitedNode(hg.initialNumNodes()),
      _visitedHyperedge(hg.initialNumEdges()),
      _core(0)
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
      size_t U = hg.initialNumNodes() / (context.partition.k * 50);
      size_t coreSize = U / 10;
      BreathFirstSearch(hg, v, coreSize, U);
    }

    void BreathFirstSearch(const Hypergraph& hg, const HypernodeID start, size_t coreSize, size_t U);

  };
}


