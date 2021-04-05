
#pragma once

#include "external_tools/kahypar/kahypar/datastructure/fast_reset_flag_array.h"

namespace mt_kahypar::community_detection {
  void depthFirstSearch(HypernodeID v, HypernodeID d, Hypergraph& hypergraph,
                        kahypar::ds::FastResetFlagArray<>& visitedHypernode, kahypar::ds::FastResetFlagArray<>& processedHypernode,
                        std::vector<HypernodeID>& depth, std::vector<HypernodeID>& lowPoint, std::vector<HypernodeID>& parent);
  ds::Clustering run_natural_cut_detection(Hypergraph& hypergraph, const Context& context, bool disable_randomization = false);
}