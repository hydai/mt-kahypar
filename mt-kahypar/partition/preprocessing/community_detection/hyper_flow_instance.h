
#pragma once

#include "external_tools/kahypar/external_tools/WHFC/datastructure/flow_hypergraph_builder.h"
#include "external_tools/kahypar/kahypar/datastructure/fast_reset_flag_array.h"

namespace mt_kahypar::community_detection {

  class HyperFlowInstance {

  public:
    static constexpr HypernodeID invalid_node = std::numeric_limits<HypernodeID>::max();
    std::vector<HypernodeID> _core;

    HyperFlowInstance(Hypergraph& hg, const Context& context, HypernodeID v, kahypar::ds::FastResetFlagArray<>& hypernodeProcessed) :
      _flow_hg_builder(),
      _nodeIDMap(hg.initialNumNodes() + 2, whfc::invalidNode),
      _edgeIDMap(0),
      _visitedNode(hg.initialNumNodes()),
      _visitedHyperedge(hg.initialNumEdges()),
      _core(0),
      shouldBeComputed(true)
      {
        constructFlowgraphFromSourceNode(hg, v, context, hypernodeProcessed);
      }


    std::vector<HyperedgeID> computeCut();

  private:
    HypernodeID _globalSourceID = invalid_node, _globalTargetID = invalid_node;
    whfc::FlowHypergraphBuilder _flow_hg_builder;
    std::vector<HyperedgeID> _edgeIDMap;
    std::vector<whfc::Node> _nodeIDMap;
    kahypar::ds::FastResetFlagArray<> _visitedNode;
    kahypar::ds::FastResetFlagArray<> _visitedHyperedge;
    bool shouldBeComputed;

    void constructFlowgraphFromSourceNode(const Hypergraph& hg, const HypernodeID v,const Context& context, kahypar::ds::FastResetFlagArray<>& hypernodeProcessed){
      size_t U = hg.initialNumNodes() / (context.partition.k);
      size_t coreSize = U / 10;
      BreathFirstSearch(hg, v, coreSize, U, hypernodeProcessed);
    }

    void BreathFirstSearch(const Hypergraph& hg, const HypernodeID start, size_t coreSize, size_t U, kahypar::ds::FastResetFlagArray<>& hypernodeProcessed);

  };
}


