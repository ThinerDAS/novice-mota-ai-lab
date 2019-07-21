# proof of YINHE 2-10F max based on assumptions

This code uses DP (dynamic programming) to solve 2-10F max of the mota game [yinhe](https://h5mota.com/games/yinhe/).

There are several pruning strategies employed in the solver.

This is based on the following assumptions:

1. The player is 214HP 5AT 2DF 4MF 26G 1Y2B when reaching 2F and takes the initial keys & potions.
2. The shop is unused, and the enemy set is exactly the same as the route indicates.

[The latest runner code that should be run using pypy](ai4.py), [the 2-10F graph info](ai_info.data), [the pruneless output](output.txt), [output route info](route_info.data), and [usable .h5route file](upto10.h5route) are all here.

The proof remains to be verified. And I hope the prune strategies here is inspiring.