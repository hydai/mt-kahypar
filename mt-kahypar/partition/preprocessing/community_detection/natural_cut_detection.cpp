
#include "natural_cut_detection.h"
#include "hyper_flow_instance.h"
#include "external_tools/kahypar/kahypar/datastructure/fast_reset_flag_array.h"
#include "external_tools/kahypar/external_tools/WHFC/datastructure/queue.h"
#include "mt-kahypar/utils/timer.h"

namespace mt_kahypar::community_detection {

  using Queue = LayeredQueue<HypernodeID>;

  ds::Clustering run_natural_cut_detection(Hypergraph& hypergraph, const Context& context, bool disable_randomization) {
    kahypar::ds::FastResetFlagArray<> hypernodeProcessed(hypergraph.initialNumNodes());
    kahypar::ds::FastResetFlagArray<> visitedHyperedge(hypergraph.initialNumEdges());
    ds::Clustering communities(hypergraph.initialNumNodes());
    parallel::scalable_vector<HypernodeID> vertices;
    vertices.resize(hypergraph.initialNumNodes());
    tbb::parallel_for(ID(0), hypergraph.initialNumNodes(), [&](const HypernodeID hn) {
      ASSERT(hn < vertices.size());
      vertices[hn] = hn;
    });
    //TODO make deterministic
    if ( !disable_randomization ) {
      utils::Randomize::instance().parallelShuffleVector(
        vertices, 0UL, vertices.size());
    }

    size_t progress = 0;
    // Do flow calculations from every Hypernode
    //tbb::enumerable_thread_specific <std::vector<HyperedgeID>> cut_edges_local;
    //for (HypernodeID id = 0; id < hypergraph.initialNumNodes(); id++) {
    tbb::parallel_for(ID(0), hypergraph.initialNumNodes(), [&](const HypernodeID id) {    // REVIEW no control how often a vertex appears in a core. potentially quadratic running  time in allocations
      ASSERT(id < vertices.size());
      HypernodeID v = vertices[id];
      if (!hypernodeProcessed[v]) {
        std::cout << "Starting Flow Iteration" << std::endl;
        auto t = tbb::tick_count::now();
        HyperFlowInstance hfib(hypergraph, context, v);
        auto t2 = tbb::tick_count::now();
        std::cout << "Starting Flow Computation" << std::endl;
        std::vector<HyperedgeID> cut = hfib.computeCut();
        auto t3 = tbb::tick_count::now();
        std::cout << "Found cut with " << cut.size() << " edges" << std::endl;
        std::cout << "Time building Flowgraph " << (t2-t).seconds() << std::endl;
        std::cout << "Time calculating Cut " << (t3-t2).seconds() << std::endl;
        //cut_edges_local.local().insert(cut_edges_local.local().end(), cut.begin(), cut.end());
        for (HyperedgeID he : cut) {
          visitedHyperedge.set(he);
        }
        for (HypernodeID hn : hfib._core) {
          if (!hypernodeProcessed[hn]) {
            progress++;   // REVIEW data race
          }
          hypernodeProcessed.set(hn);
        }
        std::cout << "Progress: " << progress << "/" << hypergraph.initialNumNodes() << std::endl;
      }
    //}
    });

    // Compute Connected Components
    hypernodeProcessed.reset();
    int current_community = 0;
    for (HypernodeID v = 0; v < hypergraph.initialNumNodes(); v++) {
      if (!hypernodeProcessed[v]) {
        Queue queue(hypergraph.initialNumNodes());
        queue.push(v);
        hypernodeProcessed.set(v);
        communities[v] = current_community;
        while (!queue.empty()) {
          HypernodeID u = queue.pop();
          for (const HyperedgeID e : hypergraph.incidentEdges(u)) {
            if (!visitedHyperedge[e]) {
              visitedHyperedge.set(e);
              for (const HypernodeID u : hypergraph.pins(e)) {
                if (!hypernodeProcessed[u]) {
                  queue.push(u);
                  hypernodeProcessed.set(u);
                  communities[u] = current_community;
                }
              }
            }
          }
        }
        current_community++;
      }
    }

    return communities;
  }
}