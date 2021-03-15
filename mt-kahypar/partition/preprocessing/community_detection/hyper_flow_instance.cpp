
#include "hyper_flow_instance.h"
#include <external_tools/WHFC/algorithm/cutter_state.h>
#include <external_tools/WHFC/algorithm/dinic.h>
#include <external_tools/WHFC/algorithm/grow_assimilated.h>
#include "external_tools/kahypar/external_tools/WHFC/datastructure/flow_hypergraph_builder.h"
#include "external_tools/kahypar/external_tools/WHFC/datastructure/node_border.h"
#include "external_tools/kahypar/external_tools/WHFC/datastructure/flow_hypergraph.h"
#include "external_tools/kahypar/external_tools/WHFC/datastructure/queue.h"


namespace mt_kahypar::community_detection {

  using Queue = LayeredQueue<HypernodeID>;

  std::vector<HyperedgeID> HyperFlowInstance::computeCut() {
    std::vector<HyperedgeID> cut(0);
    whfc::TimeReporter timer("HyperFlowCommunityDetection");
    whfc::CutterState<whfc::Dinic> cs(_flow_hg_builder, timer);
    cs.initialize(_nodeIDMap[_globalSourceID],_nodeIDMap[_globalTargetID]);
    whfc::Dinic flow_algo(_flow_hg_builder);
    flow_algo.upperFlowBound = std::numeric_limits<whfc::Flow>::max();

    _flow_hg_builder.printHypergraph(std::cout);

    /*while (!cs.hasCut) {
      if (cs.augmentingPathAvailableFromPiercing) {
        cs.hasCut = flow_algo.exhaustFlow(cs);
        if (cs.hasCut) {
          cs.flipViewDirection();
          flow_algo.growReachable(cs);
        }
      }
      else {
        flow_algo.growReachable(cs);	// don't grow target reachable
        cs.hasCut = true;
      }
      cs.verifyFlowConstraints();

      if (cs.hasCut) {
        cs.verifySetInvariants();
        if (cs.sideToGrow() != cs.currentViewDirection()) {
          cs.flipViewDirection();
        }
        whfc::GrowAssimilated<whfc::Dinic>::grow(cs, flow_algo.getScanList());
        cs.verifyCutPostConditions();
      }
    }*/
    if (cs.augmentingPathAvailableFromPiercing) {
      std::cout << "test\n";
    }
    while (!cs.hasCut) {
      flow_algo.growReachable(cs);
      flow_algo.growFlowOrSourceReachable(cs);
      cs.hasCut = flow_algo.exhaustFlow(cs);
    }
    if (cs.augmentingPathAvailableFromPiercing) {
      std::cout << "test\n";
    }
    //cs.hasCut = true;
    std::cout << "Flow: " << cs.flowValue << std::endl;
    std::cout << "Border: " << cs.borderNodes.sourceSide.get() << std::endl;
    cs.verifyCutPostConditions();
    for ( whfc::Hyperedge e : cs.cuts.sourceSide.entries()) {
      cut.push_back(_edgeIDMap[e]);
    }
    return cut;
  }

  std::pair<size_t,size_t> HyperFlowInstance::initializeFlowHGBuilder(const Hypergraph& hg, const HypernodeID start, size_t U) {
    Queue queue(hg.initialNumNodes());
    queue.push(start);
    _visitedNode.set(start);
    size_t numNodesVisited = 0;
    size_t numEdgesVisited = 0;
    size_t numPins = 0;
    size_t nodesInLastLayer = 0;
    size_t nodesInLayer = 0;
    HypernodeID layerend = start;
    while (!queue.empty() && numNodesVisited < U) {
      HypernodeID v = queue.pop();
      nodesInLayer++;
      for (const HyperedgeID e : hg.incidentEdges(v)) {
        if (!_visitedHyperedge[e]) {
          numEdgesVisited++;
          _visitedHyperedge.set(e);
          HypernodeID last;
          for (HypernodeID u : hg.pins(e)) {
            if (!_visitedNode[u]) {
              queue.push(u);
              _visitedNode.set(u);
              last = u;
            }
            numPins++;
          }
          numNodesVisited++;
          if (v == layerend) {
            layerend == last;
            nodesInLastLayer = nodesInLayer;
            nodesInLayer = 0;
          }
        }
      }
    }
    size_t coreSize;
    size_t ringSize;
    if (numNodesVisited < U) {
      coreSize = (numNodesVisited - nodesInLastLayer) / 10;
      ringSize = nodesInLastLayer;
    } else {
      coreSize = numNodesVisited / 10;
      ringSize = queue.qend - queue.qfront;
    }
    _core.resize(coreSize);
    //_flow_hg_builder = whfc::FlowHypergraphBuilder(queue.qend + 2, numEdgesVisited + 2, 0);// numPins + coreSize + ringSize + 2);
    _flow_hg_builder = whfc::FlowHypergraphBuilder(0, 0, 0);// numPins + coreSize + ringSize + 2);
    _edgeIDMap = std::vector<HyperedgeID>(numEdgesVisited + 2, std::numeric_limits<HyperedgeID>::max());
    _visitedNode.reset();
    _visitedHyperedge.reset();
    return std::make_pair(coreSize, ringSize);
  }

  void HyperFlowInstance::BreathFirstSearch(const Hypergraph &hg, const HypernodeID start, size_t coreSize, size_t ringSize, size_t U) {
    //TODO reuse other queue
    size_t a = hg.initialNumNodes();
    Queue queue(hg.initialNumNodes()*2);
    _nodeIDMap[start] = whfc::Node::fromOtherValueType(queue.queueEnd());
    _flow_hg_builder.addNode(whfc::NodeWeight(hg.nodeWeight(start)));
    queue.push(start);
    _visitedNode.set(start);
    size_t numVisited = 0;
    size_t numEdges = 0;
    //TODO remove
    size_t numPins = 0;
    while (!queue.empty() && numVisited < U) {
      HypernodeID v = queue.pop();
      if (numVisited < coreSize) {
        _core[numVisited] = v;
        //_core.push_back(v);
      }
      for (const HyperedgeID e : hg.incidentEdges(v)) {
        if (!_visitedHyperedge[e]) {
          _visitedHyperedge.set(e);
          _flow_hg_builder.startHyperedge(hg.edgeWeight(e));
          //size_t x = _flow_hg_builder.numHyperedges();
          //_edgeIDMap[_flow_hg_builder.numHyperedges()] = e;
          _edgeIDMap[numEdges++] = e;
          for (const HypernodeID u : hg.pins(e)) {
            if (queue.qend > a) {
              size_t l = queue.qend;
            }
            if (!_visitedNode[u]) {
              if (u > a, u < 0) {
                size_t l2 = u;
              }
              std::cout << u << std::endl;
              _nodeIDMap[u] = whfc::Node::fromOtherValueType(queue.queueEnd());
              _flow_hg_builder.addNode(whfc::NodeWeight(hg.nodeWeight(u)));
              queue.push(u);
              _visitedNode.set(u);
            }
            _flow_hg_builder.addPin(_nodeIDMap[u]);
            numPins++;
          }
        }
      }
      numVisited++;
    }
    _flow_hg_builder.startHyperedge(std::numeric_limits<whfc::Flow>::max());
    if (numVisited < U) {
      queue.qfront = queue.qend - ringSize;
    }
    while (!queue.empty()) {
      HypernodeID v = queue.pop();
      _flow_hg_builder.addPin(_nodeIDMap[v]);
      numPins++;
    }
    _globalTargetID = hg.initialNumNodes();
    _nodeIDMap[_globalTargetID] = whfc::Node::fromOtherValueType(queue.queueEnd());
    _flow_hg_builder.addNode(0);
    whfc::NodeWeight w = _flow_hg_builder.nodeWeight(whfc::Node::fromOtherValueType(_nodeIDMap[_globalTargetID] + 272));
    whfc::NodeWeight w2 = _flow_hg_builder.nodeWeight(whfc::Node(0));
    _flow_hg_builder.addPin(_nodeIDMap[_globalTargetID]);
    numPins++;
    _globalSourceID = hg.initialNumNodes() + 1;
    _nodeIDMap[_globalSourceID] = whfc::Node::fromOtherValueType(queue.queueEnd()+1);
    _flow_hg_builder.addNode(0);
    _flow_hg_builder.startHyperedge(std::numeric_limits<whfc::Flow>::max());
    _flow_hg_builder.addPin(_nodeIDMap[_globalSourceID]);
    numPins++;
    for (HypernodeID v : _core) {
      _flow_hg_builder.addPin(_nodeIDMap[v]);
      numPins++;
    }
    _flow_hg_builder.finalize();
  }

  /*void HyperFlowInstance::BreathFirstSearch(const Hypergraph &hg, const HypernodeID start, size_t coreSize, size_t U) {
    //TODO reuse other queue
    Queue queue(U);
    _nodeIDMap[start] = whfc::Node::fromOtherValueType(queue.queueEnd());
    _flow_hg_builder.addNode(whfc::NodeWeight(hg.nodeWeight(start)));
    queue.push(start);
    _visitedNode.set(start);
    int numVisited = 0;
    while (!queue.empty() && numVisited < U) {
      HypernodeID v = queue.pop();
      if (numVisited < coreSize) {
        _core.push_back(v);
      }
      for (const HyperedgeID e : hg.incidentEdges(v)) {
        if (!_visitedHyperedge[e]) {
          _visitedHyperedge.set(e);
          _flow_hg_builder.startHyperedge(hg.edgeWeight(e));
          _edgeIDMap[_flow_hg_builder.numHyperedges()] = e;
          for (const HypernodeID u : hg.pins(e)) {
            if (!_visitedNode[u]) {
              _nodeIDMap[u] = whfc::Node::fromOtherValueType(queue.queueEnd());
              _flow_hg_builder.addNode(whfc::NodeWeight(hg.nodeWeight(u)));
              queue.push(u);
              _visitedNode.set(u);
              _flow_hg_builder.addPin(_nodeIDMap[u]);
            }
          }
        }
      }
      numVisited++;
    }
    _flow_hg_builder.startHyperedge(std::numeric_limits<whfc::Flow>::max());
    while (!queue.empty()) {
      HypernodeID v = queue.pop();
      _flow_hg_builder.addPin(_nodeIDMap[v]);
    }
    _globalTargetID = hg.initialNumNodes();
    _nodeIDMap[_globalTargetID] = whfc::Node::fromOtherValueType(queue.queueEnd());
    _flow_hg_builder.addNode(std::numeric_limits<whfc::NodeWeight>::max() - 1);
    _flow_hg_builder.addPin(_nodeIDMap[_globalTargetID]);
    _globalSourceID = hg.initialNumNodes() + 1;
    _nodeIDMap[_globalSourceID] = whfc::Node::fromOtherValueType(queue.queueEnd()+1);
    _flow_hg_builder.addNode(std::numeric_limits<whfc::NodeWeight>::max() - 1);
    _flow_hg_builder.startHyperedge(std::numeric_limits<whfc::Flow>::max());
    _flow_hg_builder.addPin(_nodeIDMap[_globalSourceID]);
    size_t x = _core.size();
    for (HypernodeID v : _core) {
      _flow_hg_builder.addPin(_nodeIDMap[v]);
    }
    _flow_hg_builder.finalize();
  }*/

}