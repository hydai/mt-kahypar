set(PreprocessingSources
        community_detection/parallel_louvain.cpp
        community_detection/local_moving_modularity.cpp
        community_detection/natural_cut_detection.cpp
        community_detection/hyper_flow_instance.cpp)

foreach(modtarget IN LISTS TARGETS_WANTING_ALL_SOURCES)
    target_sources(${modtarget} PRIVATE ${PreprocessingSources})
endforeach()