from MERGE_Main import *

i = ModelRunner("/Users/peter/projects/MERGE/Combined_corpus/")
i.set_params(gapsize=1, iteration_count=100)
i.run()
