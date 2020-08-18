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

#include <cstdlib>

#include "tbb/parallel_for.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/io/hypergraph_io.h"

namespace mt_kahypar {

class KaHyParInitialPartitioner {

  static constexpr bool debug = false;

 public:
  KaHyParInitialPartitioner(PartitionedHypergraph& hypergraph,
                            const Context& context) :
    _hg(hypergraph),
    _context(context) { }

  void initial_partition() {
    // Write hypergraph file to tmp folder
    std::string hypergraph_file_basename =
      _context.partition.graph_filename.substr(
      _context.partition.graph_filename.find_last_of('/') + 1);
    std::string tmp_hypergraph_file = "/tmp/" + hypergraph_file_basename;

    io::writeHypergraphFile(_hg, tmp_hypergraph_file);

    // Build KaHyPar Binary Call
    std::string epsilon_str = std::to_string(_context.partition.epsilon);
      epsilon_str.erase(epsilon_str.find_last_not_of('0') + 1, std::string::npos);
    std::string kahypar_call = _context.initial_partitioning.kahypar_binary + " " +
      "-h " + tmp_hypergraph_file + " " +
      "-k " + std::to_string(_context.partition.k) + " " +
      "-e " + epsilon_str + " " +
      "-o km1" + " " +
      "-m direct" + " " +
      "-p " + _context.initial_partitioning.kahypar_context + " " +
      "--seed=" + std::to_string(_context.partition.seed) + " " +
      "--write-partition=true " +
      "--quiet=" + std::to_string(_context.initial_partitioning.kahypar_quiet_mode);

    if ( _context.partition.verbose_output ) {
      LOG << "Calling KaHyPar:";
      LOG << kahypar_call << "\n";
    }

    // Call KaHyPar Binary
    int ret = system(kahypar_call.c_str());
    if ( ret == 0 ) {
      // Read KaHyPar Partition File
      std::string tmp_partition_file = "/tmp/" + hypergraph_file_basename +
        ".part" + std::to_string(_context.partition.k) +
        ".epsilon" + epsilon_str +
        ".seed" + std::to_string(_context.partition.seed) +
        ".KaHyPar";
      std::vector<PartitionID> partition;
      io::readPartitionFile(tmp_partition_file, partition);

      // Apply Partition to Mt-KaHyPar hypergraph
      ASSERT(_hg.initialNumNodes() == ID(partition.size()));
      tbb::parallel_for(0UL, partition.size(), [&](const size_t i) {
        const HypernodeID hn = ID(i);
        const PartitionID block = partition[i];
        ASSERT(block != kInvalidPartition && block < _context.partition.k);
        _hg.setOnlyNodePart(hn, block);
      });

      // Delete tmp files
      std::string delete_tmp_hypergraph_file = "rm -f " + tmp_hypergraph_file;
      std::string delete_tmp_partition_file = "rm -f " + tmp_partition_file;
      int ret_1 = system(delete_tmp_hypergraph_file.c_str());
      int ret_2 = system(delete_tmp_partition_file.c_str());
      if ( ret_1 != 0 || ret_2 != 0 ) {
        ERROR("Failed to delete tmp hypergraph file and partition file");
      }
    } else {
      ERROR("KaHyPar terminates with exit code" << ret);
    }
  }

 private:
  PartitionedHypergraph& _hg;
  const Context& _context;
};
} // namespace mt_kahypar
