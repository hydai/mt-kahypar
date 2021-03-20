
#pragma once

namespace mt_kahypar::community_detection {
  ds::Clustering run_natural_cut_detection(Hypergraph& hypergraph, const Context& context, bool disable_randomization = false);
}