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

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/coarsening/nlevel_coarsener_base.h"
#include "mt-kahypar/partition/coarsening/i_coarsener.h"
#include "mt-kahypar/utils/progress_bar.h"
#include "mt-kahypar/utils/timer.h"

namespace mt_kahypar {

class KaHyParCoarsener : public ICoarsener,
                         private NLevelCoarsenerBase {
 private:

  using Base = NLevelCoarsenerBase;
  using InternalKaHyParCoarsener = typename Base::InternalKaHyParCoarsener;

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

 public:
  KaHyParCoarsener(Hypergraph& hypergraph,
                  const Context& context,
                  const TaskGroupID task_group_id,
                  const bool top_level) :
    Base(hypergraph, context, task_group_id, top_level),
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
    _kahypar_hg = std::move(converted_kahypar_hypergraph.first);
    _to_kahypar_hg = std::move(converted_kahypar_hypergraph.second);
    _to_mt_kahypar_hg.assign(_kahypar_hg->initialNumNodes(), 0);
    _hg.doParallelForAllNodes([&](const HypernodeID hn) {
      const HypernodeID kahypar_hn = _to_kahypar_hg[hn];
      _to_mt_kahypar_hg[kahypar_hn] = hn;
    });
    Base::initializeCommunities(_kahypar_hg, _to_kahypar_hg);
    utils::Timer::instance().stop_timer("initialize_kahypar_hypergraph");

    utils::Timer::instance().start_timer("interleaved_kahypar_coarsening", "Interleaved (Mt-)KaHyPar Coarsening");
    _kahypar_coarsener = std::make_unique<InternalKaHyParCoarsener>(
      *_kahypar_hg, _kahypar_context, _kahypar_hg->weightOfHeaviestNode(),
      [&](const HypernodeID kahypar_u, const HypernodeID kahypar_v) {
        const HypernodeID u = _to_mt_kahypar_hg[kahypar_u];
        const HypernodeID v = _to_mt_kahypar_hg[kahypar_v];
        _hg.registerContraction(u, v);
        _hg.contract(v);
        _progress_bar += 1;
      }, [&]() {
        Base::removeSinglePinAndParallelNets();
      }, [&](const kahypar::Metrics&) { });
    _kahypar_coarsener->coarsen(_kahypar_context.coarsening.contraction_limit);
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
    if ( _context.refinement.use_kahypar_refinement ) {
      return Base::doKaHyParUncoarsen();
    } else {
      return Base::doUncoarsen(label_propagation, fm);
    }
  }

  using Base::_hg;
  using Base::_context;
  using Base::_kahypar_hg;
  using Base::_kahypar_context;
  using Base::_kahypar_coarsener;
  using Base::_to_kahypar_hg;
  using Base::_to_mt_kahypar_hg;
  utils::ProgressBar _progress_bar;
};

}  // namespace mt_kahypar
