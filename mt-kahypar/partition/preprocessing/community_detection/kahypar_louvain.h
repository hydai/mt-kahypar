/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2020 Tobias Heuer <tobias.heuer@kit.edu>
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

#include "kahypar/partition/context.h"
#include "kahypar/application/command_line_options.h"
#include "kahypar/partition/preprocessing/louvain.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/datastructures/clustering.h"
#include "mt-kahypar/io/hypergraph_io.h"

namespace mt_kahypar {

class KaHyParLouvain {

  static constexpr bool debug = false;

 public:
  static ds::Clustering run(const Hypergraph& hypergraph, const Context& context) {
    // Load KaHyPar Context and Hypergraph
    auto converted_kahypar_hypergraph = io::convertToKaHyParHypergraph(hypergraph, 2);
    std::unique_ptr<kahypar::Hypergraph> kahypar_hypergraph = std::move(converted_kahypar_hypergraph.first);
    parallel::scalable_vector<HypernodeID> node_mapping = std::move(converted_kahypar_hypergraph.second);
    kahypar::Context kahypar_context = setupKaHyParContext(
      *kahypar_hypergraph, context.partition.kahypar_context);

    // Perform KaHyPar Louvain Community Detection
    kahypar::detectCommunities(*kahypar_hypergraph, kahypar_context);

    // Store Communities
    ds::Clustering clustering(hypergraph.initialNumNodes());
    auto& communities = kahypar_hypergraph->communities();
    hypergraph.doParallelForAllNodes([&](const HypernodeID hn) {
      clustering[hn] = communities[node_mapping[hn]];
    });

    return clustering;
  }

 private:
  static kahypar::Context setupKaHyParContext(const kahypar::Hypergraph& hypergraph,
                                              const std::string& kahypar_ini) {
    kahypar::Context kahypar_context;
    kahypar::parseIniToContext(kahypar_context, kahypar_ini);

    // Setup edge weight function
    if (kahypar_context.preprocessing.community_detection.edge_weight ==
        kahypar::LouvainEdgeWeight::hybrid) {
      const double density = static_cast<double>(hypergraph.initialNumEdges()) /
                            static_cast<double>(hypergraph.initialNumNodes());
      if (density < 0.75) {
        kahypar_context.preprocessing.community_detection.edge_weight = kahypar::LouvainEdgeWeight::degree;
      } else {
        kahypar_context.preprocessing.community_detection.edge_weight = kahypar::LouvainEdgeWeight::uniform;
      }
    }

    return kahypar_context;
  }
};
}
