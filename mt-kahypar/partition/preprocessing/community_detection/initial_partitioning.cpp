#include "initial_partitioning.h"
#include <mt-kahypar/partition/initial_partitioning/i_initial_partitioner.h>
#include "mt-kahypar/partition/factories.h"

namespace mt_kahypar::community_detection {
  ds::Clustering run_initial_partioning(Hypergraph & hypergraph, const Context& context, bool disable_randomization){
    ds::Clustering communities(hypergraph.initialNumNodes());
    PartitionedHypergraph phg(2, hypergraph);

    std::unique_ptr<IInitialPartitioner> initial_partitioner =
      InitialPartitionerFactory::getInstance().createObject(
        context.initial_partitioning.mode, phg,
        context, true, TBBNumaArena::GLOBAL_TASK_GROUP);
    initial_partitioner->initialPartition();

    tbb::parallel_for(ID(0), hypergraph.initialNumNodes(), [&](const HypernodeID hn) {
      communities[hn] = phg.partID(hn);
    });
    return communities;
  }
}