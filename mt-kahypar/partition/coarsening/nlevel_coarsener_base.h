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

#include "tbb/task_group.h"

#include "kahypar/partition/context.h"
#include "kahypar/application/command_line_options.h"
#include "kahypar/partition/coarsening/ml_coarsener.h"
#include "kahypar/partition/refinement/kway_fm_km1_refiner.h"
#include "kahypar/partition/refinement/policies/fm_stop_policy.h"
#include "kahypar/partition/refinement/policies/fm_improvement_policy.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/partition/refinement/i_refiner.h"
#include "mt-kahypar/partition/refinement/rebalancing/rebalancer.h"
#include "mt-kahypar/parallel/stl/scalable_vector.h"
#include "mt-kahypar/io/partitioning_output.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/utils/progress_bar.h"
#include "mt-kahypar/utils/timer.h"
#include "mt-kahypar/utils/stats.h"

namespace mt_kahypar {

class NLevelCoarsenerBase {
 private:

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

  using ParallelHyperedgeVector = parallel::scalable_vector<parallel::scalable_vector<ParallelHyperedge>>;
  using ContractionFunc = std::function<void (const HypernodeID, const HypernodeID)>;
  using EndOfPassFunc = std::function<void ()>;
  using UncontractionFunc = std::function<void (const kahypar::Metrics&)>;

  using KaHyParRefiner = kahypar::KWayKMinusOneRefiner<kahypar::AdvancedRandomWalkModelStopsSearch,
                                                       kahypar::CutDecreasedOrInfeasibleImbalanceDecreased>;
  using KaHyParMemento = typename kahypar::Hypergraph::Memento;

 public:
  using InternalKaHyParCoarsener = kahypar::MLCoarsener<kahypar::HeavyEdgeScore,
                                                        kahypar::NoWeightPenalty,
                                                        kahypar::UseCommunityStructure,
                                                        kahypar::NormalPartitionPolicy,
                                                        kahypar::BestRatingPreferringUnmatched<>,
                                                        kahypar::AllowFreeOnFixedFreeOnFreeFixedOnFixed,
                                                        kahypar::RatingType,
                                                        ContractionFunc,
                                                        EndOfPassFunc,
                                                        UncontractionFunc>;

  NLevelCoarsenerBase(Hypergraph& hypergraph,
                      const Context& context,
                      const TaskGroupID task_group_id,
                      const bool top_level) :
    _is_finalized(false),
    _hg(hypergraph),
    _context(context),
    _task_group_id(task_group_id),
    _top_level(top_level),
    _phg(),
    _compactified_hg(),
    _compactified_phg(),
    _compactified_hn_mapping(),
    _hierarchy(),
    _removed_hyperedges_batches(),
    _kahypar_hg(nullptr),
    _kahypar_context(setupKaHyParContext(context)),
    _kahypar_coarsener(nullptr),
    _to_kahypar_hg(),
    _to_mt_kahypar_hg() { }

  NLevelCoarsenerBase(const NLevelCoarsenerBase&) = delete;
  NLevelCoarsenerBase(NLevelCoarsenerBase&&) = delete;
  NLevelCoarsenerBase & operator= (const NLevelCoarsenerBase &) = delete;
  NLevelCoarsenerBase & operator= (NLevelCoarsenerBase &&) = delete;

  virtual ~NLevelCoarsenerBase() = default;

 protected:

  Hypergraph& compactifiedHypergraph() {
    ASSERT(_is_finalized);
    return _compactified_hg;
  }

  PartitionedHypergraph& compactifiedPartitionedHypergraph() {
    ASSERT(_is_finalized);
    return _compactified_phg;
  }

  void finalize() {
    // Create compactified hypergraph containing only enabled vertices and hyperedges
    // with consecutive IDs => Less complexity in initial partitioning.
    utils::Timer::instance().start_timer("compactify_hypergraph", "Compactify Hypergraph");
    auto compactification = HypergraphFactory::compactify(_task_group_id, _hg);
    _compactified_hg = std::move(compactification.first);
    _compactified_hn_mapping = std::move(compactification.second);
    _compactified_phg = PartitionedHypergraph(_context.partition.k, _task_group_id, _compactified_hg);
    utils::Timer::instance().stop_timer("compactify_hypergraph");

    // Create n-level batch uncontraction hierarchy
    utils::Timer::instance().start_timer("create_batch_uncontraction_hierarchy", "Create n-Level Hierarchy");
    _hierarchy = _hg.createBatchUncontractionHierarchy(TBBNumaArena::GLOBAL_TASK_GROUP, _context.refinement.max_batch_size);
    ASSERT(_removed_hyperedges_batches.size() == _hierarchy.size() - 1);
    utils::Timer::instance().stop_timer("create_batch_uncontraction_hierarchy");

    _is_finalized = true;
  }

  void removeSinglePinAndParallelNets() {
    utils::Timer::instance().start_timer("remove_single_pin_and_parallel_nets", "Remove Single Pin and Parallel Nets");
    _removed_hyperedges_batches.emplace_back(_hg.removeSinglePinAndParallelHyperedges());
    utils::Timer::instance().stop_timer("remove_single_pin_and_parallel_nets");
  }

  void initializeCommunities(std::unique_ptr<kahypar::Hypergraph>& kahypar_hg,
                             const parallel::scalable_vector<HypernodeID>& to_kahypar_hg) {
    ASSERT(kahypar_hg);
    std::vector<kahypar::PartitionID> communities(kahypar_hg->initialNumNodes());
    _hg.doParallelForAllNodes([&](const HypernodeID hn) {
      communities[to_kahypar_hg[hn]] = _hg.communityID(hn);
    });
    kahypar_hg->setCommunities(std::move(communities));
  }


  PartitionedHypergraph&& doUncoarsen(std::unique_ptr<IRefiner>& label_propagation,
                                      std::unique_ptr<IRefiner>& fm) {
    ASSERT(_is_finalized);
    kahypar::Metrics current_metrics = initialize(_compactified_phg);

    // Project partition from compactified hypergraph to original hypergraph
    utils::Timer::instance().start_timer("initialize_partition", "Initialize Partition");
    createPartitionedHypergraph();
    if ( _context.refinement.fm.algorithm != FMAlgorithm::do_nothing && fm ) {
      _phg.initializeGainInformation();
    }
    utils::Timer::instance().stop_timer("initialize_partition");

    utils::ProgressBar uncontraction_progress(_hg.initialNumNodes(),
      _context.partition.objective == kahypar::Objective::km1 ? current_metrics.km1 : current_metrics.cut,
      _context.partition.verbose_output && _context.partition.enable_progress_bar && !debug);
    uncontraction_progress += _compactified_hg.initialNumNodes();

    // Initialize Refiner
    if ( label_propagation ) {
      label_propagation->initialize(_phg);
    }
    if ( fm ) {
      fm->initialize(_phg);
    }

    // Perform batch uncontractions
    utils::Timer::instance().start_timer("batch_uncontractions", "Batch Uncontractions");
    size_t num_batches = 0;
    size_t total_batches_size = 0;
    while ( !_hierarchy.empty() ) {
      BatchVector& batches = _hierarchy.back();

      // Uncontract all batches of a specific version of the hypergraph
      while ( !batches.empty() ) {
        const Batch& batch = batches.back();
        if ( batch.size() > 0 ) {
          _phg.uncontract(batch);
          HEAVY_REFINEMENT_ASSERT(_phg.checkTrackedPartitionInformation());

          // Perform refinement
          refine(_phg, batch, label_propagation, fm, current_metrics);

          ++num_batches;
          total_batches_size += batch.size();
          // Update Progress Bar
          uncontraction_progress.setObjective(current_metrics.getMetric(
            _context.partition.mode, _context.partition.objective));
          uncontraction_progress += batch.size();
        }
        batches.pop_back();
      }

      // Restore single-pin and parallel nets to continue with the next vector of batches
      if ( !_removed_hyperedges_batches.empty() ) {
        utils::Timer::instance().start_timer("restore_single_pin_and_parallel_nets", "Restore Single Pin and Parallel Nets");
        _phg.restoreSinglePinAndParallelNets(_removed_hyperedges_batches.back());
        _removed_hyperedges_batches.pop_back();
        utils::Timer::instance().stop_timer("restore_single_pin_and_parallel_nets");
      }
      _hierarchy.pop_back();
    }
    utils::Timer::instance().stop_timer("batch_uncontractions");

    // If we finish batch uncontractions and partition is imbalanced, we try to rebalance it
    if ( _top_level && !metrics::isBalanced(_phg, _context)) {
      const HyperedgeWeight quality_before = current_metrics.getMetric(
        kahypar::Mode::direct_kway, _context.partition.objective);
      if ( _context.partition.verbose_output ) {
        LOG << RED << "Partition is imbalanced (Current Imbalance:"
            << metrics::imbalance(_phg, _context) << ") ->"
            << "Rebalancer is activated" << END;

        LOG << "Part weights: (violations in red)";
        io::printPartWeightsAndSizes(_phg, _context);
      }

      utils::Timer::instance().start_timer("rebalance", "Rebalance");
      if ( _context.partition.objective == kahypar::Objective::km1 ) {
        Km1Rebalancer rebalancer(_phg, _context);
        rebalancer.rebalance(current_metrics);
      } else if ( _context.partition.objective == kahypar::Objective::cut ) {
        CutRebalancer rebalancer(_phg, _context);
        rebalancer.rebalance(current_metrics);
      }
      utils::Timer::instance().stop_timer("rebalance");

      const HyperedgeWeight quality_after = current_metrics.getMetric(
        kahypar::Mode::direct_kway, _context.partition.objective);
      if ( _context.partition.verbose_output ) {
        const HyperedgeWeight quality_delta = quality_after - quality_before;
        if ( quality_delta > 0 ) {
          LOG << RED << "Rebalancer worsen solution quality by" << quality_delta
              << "(Current Imbalance:" << metrics::imbalance(_phg, _context) << ")" << END;
        } else {
          LOG << GREEN << "Rebalancer improves solution quality by" << abs(quality_delta)
              << "(Current Imbalance:" << metrics::imbalance(_phg, _context) << ")" << END;
        }
      }
    }

    double avg_batch_size = static_cast<double>(total_batches_size) / num_batches;
    utils::Stats::instance().add_stat("num_batches", static_cast<int64_t>(num_batches));
    utils::Stats::instance().add_stat("avg_batch_size", avg_batch_size);
    DBG << V(num_batches) << V(avg_batch_size);

    ASSERT(metrics::objective(_phg, _context.partition.objective) ==
           current_metrics.getMetric(kahypar::Mode::direct_kway, _context.partition.objective),
           V(current_metrics.getMetric(kahypar::Mode::direct_kway, _context.partition.objective)) <<
           V(metrics::objective(_phg, _context.partition.objective)));

    return std::move(_phg);
  }

  PartitionedHypergraph&& doKaHyParUncoarsen() {
    ASSERT(_is_finalized);

    // Store mapping from compactified hg to original hypergraph
    parallel::scalable_vector<HypernodeID> to_original_hg(_compactified_hg.initialNumNodes(), 0);
    _hg.doParallelForAllNodes([&](const HypernodeID hn) {
      const HypernodeID compactified_hn = _compactified_hn_mapping[hn];
      ASSERT(compactified_hn < ID(to_original_hg.size()));
      to_original_hg[compactified_hn] = hn;
    });


    const bool is_kahypar_initialized = _kahypar_hg != nullptr;
    std::vector<KaHyParMemento> tmp_mementos;
    if ( !is_kahypar_initialized ) {
      // Create KaHyPar n-Level Hierarchy
      for ( const BatchVector& batches : _hierarchy ) {
        for ( const Batch& batch : batches ) {
          for ( const Memento& memento : batch ) {
            tmp_mementos.push_back(KaHyParMemento { memento.u, memento.v });
          }
        }
      }
    }

    // Uncoarsen Mt-KaHyPar hypergraph
    doUncoarsenWithoutRefinement();

    if ( !is_kahypar_initialized ) {
      // Simulate n-Level contractions on kahypar hypergraph
      utils::Timer::instance().start_timer("simulate_mt_kahypar_n_level", "Simulate Mt-KaHyPar n-Level Hierarchy");

      // Convert to KaHyPar Hypergraph
      auto converted_kahypar_hypergraph = io::convertToKaHyParHypergraph(_hg, _context.partition.k);
      _kahypar_hg = std::move(converted_kahypar_hypergraph.first);
      _to_kahypar_hg = std::move(converted_kahypar_hypergraph.second);
      _to_mt_kahypar_hg.assign(_kahypar_hg->initialNumNodes(), 0);
      _hg.doParallelForAllNodes([&](const HypernodeID hn) {
        const HypernodeID kahypar_hn = _to_kahypar_hg[hn];
        _to_mt_kahypar_hg[kahypar_hn] = hn;
      });
      initializeCommunities(_kahypar_hg, _to_kahypar_hg);

      // Map mementos to KaHyPar Hypergraph
      std::vector<KaHyParMemento> mementos;
      for ( const KaHyParMemento& memento : tmp_mementos ) {
        mementos.push_back(KaHyParMemento { _to_kahypar_hg[memento.u], _to_kahypar_hg[memento.v] });
      }

      // Simulate KaHyPar n-Level Hierarchy
      if (_context.partition.verbose_output && _context.partition.enable_progress_bar) {
        LOG << "Simulate n-Level Hierarchy on KaHyPar Hypergraph:";
      }
      utils::ProgressBar kahypar_contraction_progress(_kahypar_hg->initialNumNodes(), 0,
        _context.partition.verbose_output && _context.partition.enable_progress_bar && !debug);
      _kahypar_coarsener = std::make_unique<InternalKaHyParCoarsener>(
        *_kahypar_hg, _kahypar_context, _kahypar_hg->weightOfHeaviestNode(),
        [&](const HypernodeID, const HypernodeID) {
          kahypar_contraction_progress += 1;
        }, [&]() { }, [&](const kahypar::Metrics&) { });
      _kahypar_coarsener->simulateContractions(mementos);
      kahypar_contraction_progress += (_kahypar_hg->initialNumNodes() - kahypar_contraction_progress.count());
      kahypar_contraction_progress.disable();

      utils::Timer::instance().stop_timer("simulate_mt_kahypar_n_level");
    }

    // Initialize Mt-KaHyPar Partition on KaHyPar Hypergraph
    utils::Timer::instance().start_timer("apply_mt_kahypar_partition", "Apply Mt-KaHyPar Partition");
    ASSERT(_compactified_phg.initialNumNodes() == _kahypar_hg->currentNumNodes(),
      V(_compactified_phg.initialNumNodes()) << V(_kahypar_hg->currentNumNodes()));
    for ( const HypernodeID& hn : _compactified_phg.nodes() ) {
      const HypernodeID kahypar_hn = _to_kahypar_hg[to_original_hg[hn]];
      _kahypar_hg->setNodePart(kahypar_hn, _compactified_phg.partID(hn));
    }
    // Check if there are unassigned vertices
    ASSERT([&] {
      for ( const HypernodeID& hn : _kahypar_hg->nodes() ) {
        if ( _kahypar_hg->partID(hn) == kInvalidPartition ) {
          LOG << "Hypernode" << hn << "is unassigned";
          return false;
        }
      }
      return true;
    }(), "KaHyPar hypergraph contains unassigned vertices");
    _kahypar_hg->initializeNumCutHyperedges();
    utils::Timer::instance().stop_timer("apply_mt_kahypar_partition");

    // Perform KaHyPar Uncoarsening
    utils::Timer::instance().start_timer("kahypar_uncoarsening", "KaHyPar Uncoarsening");
    if (_context.partition.verbose_output && _context.partition.enable_progress_bar) {
      LOG << "Perform KaHyPar n-Level Uncoarsening with k-Way FM Km1 Refiner:";
    }
    utils::ProgressBar kahypar_uncontraction_progress(_kahypar_hg->initialNumNodes(), 0,
      _context.partition.verbose_output && _context.partition.enable_progress_bar && !debug);
    kahypar_uncontraction_progress += _kahypar_hg->currentNumNodes();
    _kahypar_coarsener->setUncontractionFunction([&](const kahypar::Metrics& metrics) {
      kahypar_uncontraction_progress += 1;
      kahypar_uncontraction_progress.setObjective(metrics.km1);
    });
    std::unique_ptr<kahypar::IRefiner> kahypar_refiner =
      std::make_unique<KaHyParRefiner>(*_kahypar_hg, _kahypar_context);
    _kahypar_coarsener->uncoarsen(*kahypar_refiner);
    utils::Timer::instance().stop_timer("kahypar_uncoarsening");

    // Apply Partition to Mt-KaHyPar Hypergraph
    utils::Timer::instance().start_timer("apply_kahypar_partition", "Apply KaHyPar Partition");
    _phg.resetPartition();
    _phg.doParallelForAllNodes([&](const HypernodeID& hn) {
      const HypernodeID kahypar_hn = _to_kahypar_hg[hn];
      _phg.setOnlyNodePart(hn, _kahypar_hg->partID(kahypar_hn));
    });
    _phg.initializePartition(_task_group_id);
    utils::Timer::instance().stop_timer("apply_kahypar_partition");

    return std::move(_phg);
  }

 protected:
  kahypar::Metrics computeMetrics(PartitionedHypergraph& phg) {
    HyperedgeWeight cut = 0;
    HyperedgeWeight km1 = 0;
    tbb::parallel_invoke([&] {
      cut = metrics::hyperedgeCut(phg);
    }, [&] {
      km1 = metrics::km1(phg);
    });
    return { cut, km1,  metrics::imbalance(phg, _context) };
  }

  kahypar::Metrics initialize(PartitionedHypergraph& current_hg) {
    kahypar::Metrics current_metrics = computeMetrics(current_hg);
    int64_t num_nodes = current_hg.initialNumNodes();
    int64_t num_edges = current_hg.initialNumEdges();
    utils::Stats::instance().add_stat("initial_num_nodes", num_nodes);
    utils::Stats::instance().add_stat("initial_num_edges", num_edges);
    utils::Stats::instance().add_stat("initial_cut", current_metrics.cut);
    utils::Stats::instance().add_stat("initial_km1", current_metrics.km1);
    utils::Stats::instance().add_stat("initial_imbalance", current_metrics.imbalance);
    return current_metrics;
  }

  void refine(PartitionedHypergraph& partitioned_hypergraph,
              const Batch& batch,
              std::unique_ptr<IRefiner>& label_propagation,
              std::unique_ptr<IRefiner>& fm,
              kahypar::Metrics& current_metrics) {
    if ( debug && _top_level ) {
      io::printHypergraphInfo(partitioned_hypergraph, "Refinement Hypergraph", false);
      DBG << "Start Refinement - km1 = " << current_metrics.km1
          << ", imbalance = " << current_metrics.imbalance;
    }

    bool is_timer_disabled = false;
    if ( utils::Timer::instance().isEnabled() ) {
      utils::Timer::instance().disable();
      is_timer_disabled = true;
    }

    bool improvement_found = true;
    while( improvement_found ) {
      improvement_found = false;

      if ( label_propagation &&
           _context.refinement.label_propagation.algorithm != LabelPropagationAlgorithm::do_nothing ) {
        improvement_found |= label_propagation->refine(partitioned_hypergraph, batch, current_metrics, std::numeric_limits<double>::max());
      }

      if ( fm &&
           _context.refinement.fm.algorithm != FMAlgorithm::do_nothing ) {
        improvement_found |= fm->refine(partitioned_hypergraph, batch, current_metrics, std::numeric_limits<double>::max());
      }

      if ( _top_level ) {
        ASSERT(current_metrics.km1 == metrics::km1(partitioned_hypergraph),
               "Actual metric" << V(metrics::km1(partitioned_hypergraph))
                               << "does not match the metric updated by the refiners" << V(current_metrics.km1));
      }

      if ( !_context.refinement.refine_until_no_improvement ) {
        break;
      }
    }

    if ( is_timer_disabled ) {
      utils::Timer::instance().enable();
    }

    if ( _top_level) {
      DBG << "--------------------------------------------------\n";
    }
  }

  void createPartitionedHypergraph() {
    _phg = PartitionedHypergraph(_context.partition.k, _task_group_id, _hg);
    _phg.doParallelForAllNodes([&](const HypernodeID hn) {
      ASSERT(static_cast<size_t>(hn) < _compactified_hn_mapping.size());
      const HypernodeID compactified_hn = _compactified_hn_mapping[hn];
      const PartitionID block_id = _compactified_phg.partID(compactified_hn);
      ASSERT(block_id != kInvalidPartition && block_id < _context.partition.k);
      _phg.setOnlyNodePart(hn, block_id);
    });
    _phg.initializePartition(_task_group_id);

    ASSERT(metrics::objective(_compactified_phg, _context.partition.objective) ==
            metrics::objective(_phg, _context.partition.objective),
            V(metrics::objective(_compactified_phg, _context.partition.objective)) <<
            V(metrics::objective(_phg, _context.partition.objective)));
    ASSERT(metrics::imbalance(_compactified_phg, _context) ==
            metrics::imbalance(_phg, _context),
            V(metrics::imbalance(_compactified_phg, _context)) <<
            V(metrics::imbalance(_phg, _context)));
  }

  void doUncoarsenWithoutRefinement() {
    ASSERT(_is_finalized);
    kahypar::Metrics current_metrics = initialize(_compactified_phg);

    // Project partition from compactified hypergraph to original hypergraph
    utils::Timer::instance().start_timer("initialize_partition", "Initialize Partition");
    createPartitionedHypergraph();
    utils::Timer::instance().stop_timer("initialize_partition");

    if (_context.partition.verbose_output && _context.partition.enable_progress_bar) {
      LOG << "Uncoarsen Mt-KaHyPar Hypergraph Without Refinement:";
    }
    utils::ProgressBar uncontraction_progress(_hg.initialNumNodes(),
      _context.partition.objective == kahypar::Objective::km1 ? current_metrics.km1 : current_metrics.cut,
      _context.partition.verbose_output && _context.partition.enable_progress_bar && !debug);
    uncontraction_progress += _compactified_hg.initialNumNodes();

    // Perform batch uncontractions
    utils::Timer::instance().start_timer("batch_uncontractions", "Batch Uncontractions");
    while ( !_hierarchy.empty() ) {
      BatchVector& batches = _hierarchy.back();

      // Uncontract all batches of a specific version of the hypergraph
      while ( !batches.empty() ) {
        const Batch& batch = batches.back();
        if ( batch.size() > 0 ) {
          _phg.uncontract(batch);
          HEAVY_REFINEMENT_ASSERT(_phg.checkTrackedPartitionInformation());

          // Update Progress Bar
          uncontraction_progress.setObjective(current_metrics.getMetric(
            _context.partition.mode, _context.partition.objective));
          uncontraction_progress += batch.size();
        }
        batches.pop_back();
      }

      // Restore single-pin and parallel nets to continue with the next vector of batches
      if ( !_removed_hyperedges_batches.empty() ) {
        utils::Timer::instance().start_timer("restore_single_pin_and_parallel_nets", "Restore Single Pin and Parallel Nets");
        _phg.restoreSinglePinAndParallelNets(_removed_hyperedges_batches.back());
        _removed_hyperedges_batches.pop_back();
        utils::Timer::instance().stop_timer("restore_single_pin_and_parallel_nets");
      }
      _hierarchy.pop_back();
    }
    utils::Timer::instance().stop_timer("batch_uncontractions");
  }

  kahypar::Context setupKaHyParContext(const Context& context) {
    kahypar::Context kahypar_context;
    kahypar::parseIniToContext(kahypar_context, context.partition.kahypar_context);

    kahypar_context.partition.k = context.partition.k;
    kahypar_context.partition.epsilon = context.partition.epsilon;
    kahypar_context.partition.mode = context.partition.mode;
    kahypar_context.partition.objective = context.partition.objective;
    kahypar_context.partition.perfect_balance_part_weights = context.partition.perfect_balance_part_weights;
    kahypar_context.partition.max_part_weights = context.partition.max_part_weights;
    kahypar_context.coarsening.contraction_limit = context.coarsening.contraction_limit;
    kahypar_context.coarsening.max_allowed_node_weight = context.coarsening.max_allowed_node_weight;

    return kahypar_context;
  }

  // ! True, if coarsening terminates and finalize function was called
  bool _is_finalized;

  // ! Original hypergraph
  Hypergraph& _hg;

  const Context& _context;
  const TaskGroupID _task_group_id;
  const bool _top_level;

  // ! Original partitioned hypergraph
  PartitionedHypergraph _phg;
  // ! Once coarsening terminates we generate a compactified hypergraph
  // ! containing only enabled vertices and hyperedges within a consecutive
  // ! ID range, which is then used for initial partitioning
  Hypergraph _compactified_hg;
  // ! Compactified partitioned hypergraph
  PartitionedHypergraph _compactified_phg;
  // ! Mapping from vertex IDs of the original hypergraph to the IDs
  // ! in the compactified hypergraph
  parallel::scalable_vector<HypernodeID> _compactified_hn_mapping;

  // ! Represents the n-level hierarchy
  // ! A batch is vector of uncontractions/mementos that can be uncontracted in parallel
  // ! without conflicts. All batches of a specific version of the hypergraph are assembled
  // ! in a batch vector. Each time we perform single-pin and parallel net detection we create
  // ! a new version (simply increment a counter) of the hypergraph. Once a batch vector is
  // ! completly processed single-pin and parallel nets have to be restored.
  VersionedBatchVector _hierarchy;
  // ! Removed single-pin and parallel nets.
  // ! All hyperedges that are contained in one vector must be restored once
  // ! we completly processed a vector of batches.
  ParallelHyperedgeVector _removed_hyperedges_batches;

  std::unique_ptr<kahypar::Hypergraph> _kahypar_hg;
  kahypar::Context _kahypar_context;
  std::unique_ptr<InternalKaHyParCoarsener> _kahypar_coarsener;
  parallel::scalable_vector<HypernodeID> _to_kahypar_hg;
  parallel::scalable_vector<HypernodeID> _to_mt_kahypar_hg;
};
}  // namespace mt_kahypar
