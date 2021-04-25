#include "initial_partitioning.h"
#include <mt-kahypar/partition/initial_partitioning/i_initial_partitioner.h>
#include "mt-kahypar/partition/multilevel.h"
#include "mt-kahypar/partition/factories.h"

namespace mt_kahypar::community_detection {
  ds::Clustering run_initial_partioning(Hypergraph & hypergraph, const Context& context){
    ds::Clustering communities(hypergraph.initialNumNodes());

    Context community_detection_context(context);
    community_detection_context.refinement.label_propagation.algorithm = LabelPropagationAlgorithm::do_nothing;
    community_detection_context.refinement.fm.algorithm = FMAlgorithm::do_nothing;
    //community_detection_context.partition.verbose_output = false;
    PartitionedHypergraph phg = multilevel::partition(
      hypergraph, community_detection_context, false, TBBNumaArena::GLOBAL_TASK_GROUP);


    tbb::parallel_for(ID(0), hypergraph.initialNumNodes(), [&](const HypernodeID hn) {
      communities[hn] = phg.partID(hn);
    });
    return communities;
  }
}