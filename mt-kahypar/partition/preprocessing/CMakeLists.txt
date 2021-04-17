set(PreprocessingSources
        community_detection/initial_partitioning.cpp
        community_detection/parallel_louvain.cpp
        community_detection/local_moving_modularity.cpp)

foreach(modtarget IN LISTS TARGETS_WANTING_ALL_SOURCES)
    target_sources(${modtarget} PRIVATE ${PreprocessingSources})
endforeach()