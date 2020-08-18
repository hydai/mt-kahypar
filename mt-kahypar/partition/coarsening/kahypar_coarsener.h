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

#include <string>

#include "tbb/parallel_for.h"

#include "kahypar/partition/context.h"
#include "kahypar/application/command_line_options.h"
#include "kahypar/partition/coarsening/ml_coarsener.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/coarsening/nlevel_coarsener_base.h"
#include "mt-kahypar/partition/coarsening/i_coarsener.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/utils/progress_bar.h"
#include "mt-kahypar/utils/timer.h"

namespace mt_kahypar {

class KaHyParCoarsener : public ICoarsener,
                         private NLevelCoarsenerBase {
 private:

  using Base = NLevelCoarsenerBase;
  using ContractionFunc = std::function<void (const HypernodeID, const HypernodeID)>;
  using EndOfPassFunc = std::function<void ()>;
  using InternalKaHyParCoarsener = kahypar::MLCoarsener<kahypar::HeavyEdgeScore,
                                                        kahypar::NoWeightPenalty,
                                                        kahypar::UseCommunityStructure,
                                                        kahypar::NormalPartitionPolicy,
                                                        kahypar::BestRatingPreferringUnmatched<>,
                                                        kahypar::AllowFreeOnFixedFreeOnFreeFixedOnFixed,
                                                        kahypar::RatingType,
                                                        ContractionFunc,
                                                        EndOfPassFunc>;

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

 public:
  KaHyParCoarsener(Hypergraph& hypergraph,
                  const Context& context,
                  const TaskGroupID task_group_id,
                  const bool top_level) :
    Base(hypergraph, context, task_group_id, top_level),
    kahypar_hg(nullptr),
    to_kahypar_hg(),
    to_mt_kahypar_hg(),
    _progress_bar(hypergraph.initialNumNodes(), 0, false) {
    _progress_bar += hypergraph.numRemovedHypernodes();
  }

  KaHyParCoarsener(const KaHyParCoarsener&) = delete;
  KaHyParCoarsener(KaHyParCoarsener&&) = delete;
  KaHyParCoarsener & operator= (const KaHyParCoarsener &) = delete;
  KaHyParCoarsener & operator= (KaHyParCoarsener &&) = delete;

  ~KaHyParCoarsener() = default;

 private:
  void coarsenImpl() override {
    if ( _context.partition.verbose_output && _context.partition.enable_progress_bar ) {
      _progress_bar.enable();
    }

    utils::Timer::instance().start_timer("initialize_kahypar_hypergraph", "Initialize KaHyPar Hypergraph");
    auto converted_kahypar_hypergraph = io::convertToKaHyParHypergraph(_hg, _context.partition.k);
    kahypar_hg = std::move(converted_kahypar_hypergraph.first);
    to_kahypar_hg = std::move(converted_kahypar_hypergraph.second);
    to_mt_kahypar_hg.assign(kahypar_hg->initialNumNodes(), 0);
    _hg.doParallelForAllNodes([&](const HypernodeID hn) {
      const HypernodeID kahypar_hn = to_kahypar_hg[hn];
      to_mt_kahypar_hg[kahypar_hn] = hn;
    });
    kahypar::Context kahypar_context = setupKaHyParContext(_context.partition.kahypar_context);
    initializeCommunities();
    utils::Timer::instance().stop_timer("initialize_kahypar_hypergraph");

    utils::Timer::instance().start_timer("interleaved_kahypar_coarsening", "Interleaved (Mt-)KaHyPar Coarsening");
    InternalKaHyParCoarsener kahypar_coarsener(*kahypar_hg, kahypar_context, kahypar_hg->weightOfHeaviestNode(),
      [&](const HypernodeID kahypar_u, const HypernodeID kahypar_v) {
        const HypernodeID u = to_mt_kahypar_hg[kahypar_u];
        const HypernodeID v = to_mt_kahypar_hg[kahypar_v];
        _hg.registerContraction(u, v);
        _hg.contract(v);
        _progress_bar += 1;
      }, [&]() {
        Base::removeSinglePinAndParallelNets();
      });
    kahypar_coarsener.coarsen(kahypar_context.coarsening.contraction_limit);
    utils::Timer::instance().stop_timer("interleaved_kahypar_coarsening");

    const HypernodeID initial_num_nodes = _hg.initialNumNodes() - _hg.numRemovedHypernodes();
    _progress_bar += (initial_num_nodes - _progress_bar.count());
    _progress_bar.disable();
    Base::finalize();
  }

  Hypergraph& coarsestHypergraphImpl() override {
    return Base::compactifiedHypergraph();
  }

  PartitionedHypergraph& coarsestPartitionedHypergraphImpl() override {
    return Base::compactifiedPartitionedHypergraph();
  }

  PartitionedHypergraph&& uncoarsenImpl(std::unique_ptr<IRefiner>& label_propagation,
                                        std::unique_ptr<IRefiner>& fm) override {
    return Base::doUncoarsen(label_propagation, fm);
  }

  kahypar::Context setupKaHyParContext(const std::string& kahypar_ini) {
    ASSERT(kahypar_hg);
    kahypar::Context kahypar_context;
    kahypar::parseIniToContext(kahypar_context, kahypar_ini);

    kahypar_context.partition.k = _context.partition.k;
    kahypar_context.partition.epsilon = _context.partition.epsilon;
    kahypar_context.partition.mode = _context.partition.mode;
    kahypar_context.partition.objective = _context.partition.objective;
    kahypar_context.partition.perfect_balance_part_weights = _context.partition.perfect_balance_part_weights;
    kahypar_context.partition.max_part_weights = _context.partition.max_part_weights;
    kahypar_context.coarsening.contraction_limit = _context.coarsening.contraction_limit;
    kahypar_context.coarsening.max_allowed_node_weight = _context.coarsening.max_allowed_node_weight;

    return kahypar_context;
  }

  void initializeCommunities() {
    ASSERT(kahypar_hg);
    std::vector<kahypar::PartitionID> communities(kahypar_hg->initialNumNodes());
    _hg.doParallelForAllNodes([&](const HypernodeID hn) {
      communities[to_kahypar_hg[hn]] = _hg.communityID(hn);
    });
    kahypar_hg->setCommunities(std::move(communities));
  }

  using Base::_hg;
  using Base::_context;
  std::unique_ptr<kahypar::Hypergraph> kahypar_hg;
  parallel::scalable_vector<HypernodeID> to_kahypar_hg;
  parallel::scalable_vector<HypernodeID> to_mt_kahypar_hg;
  utils::ProgressBar _progress_bar;
};

}  // namespace mt_kahypar
