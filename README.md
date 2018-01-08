# MultiRankWalk
Implementation of semi-supervised learning algorithm. 
The paper is here:https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf

# Optional Command Line Arguments
-i: a file path to a text file containing the data
-d: the damping factor (float between 0 and 1)
-k: number of data points per class to use as seeds
-t: type of seed selection to use, “random” or “degree”
-e: the epsilon threshold, or squared difference of ⃗rt and ⃗rt+1 to determine conver- gence
-g: value of gamma for the pairwise RBF affinity kernel
-o: a file path to an output file, where the predicted labels will be written
