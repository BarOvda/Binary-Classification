The Perceptron Algorithm implement concurrently with MPI + CUDA + OMP

Problem Definition

Given a set of N points in K-dimensional space. Each point X is marked as belonging to set A or B. Implement a Simplified Binary Classification algorithm to find a Linear Classifier. The result depends on the maximum iteration allowed, value of the chosen parameter a and the time value t. The purpose of the project is to define a minimal value of t that leads to the Classifier with acceptable value of Quality of Classifier.

====================================================================

---------------------------------------------------------------------
MPI

The MPI logic works in dynamic way.
at first the Master read all points from the file.
Then he takes the tMax time and split it according to the number of process.

Then, Each process has its own "Time Zone" for making the perceptron alogrithm.
When a process finish his job, he send it to the master and get the next "Time Zone" to check (unless his result was satisfied).
If a process will find a satisfied solution, the master will stop distribute jobs and return the result.

The dynamic work of the master process can handle with a large amount of "Time Zones".
The slave processes will find solutions until ones is satisfied.
this will save a lot of time, because they will not check the entire time spectrum. 

Complexity - O(Tmax/ dt)


-----------------------------------------------------------------------
CUDA

Will calculate and set the new points coordinates according to the local time the slave is handle.
The rational of choosing the specific architecture - The big advantage of Cuda is that it can handle massive amount of small tasks on parallel, In this case, it handle massive amount of points which need to be relocated - its a perfect match! complexity evaluation - In this problem the max amount of input points are 500,000, and Invidia GPU have more then 500,000 threads.
Which means that Each Cuda thread can handle a single point, loop through its dimensions.
Complexity - O(k).

-------------------------------------------------------------------------
OMP

The OMP will do two things:
1.	It will check parallelly which point is the first point that misclassified.
After that the weight function will update according to that point. 
2.	If there is points that misclassified, the OMP will check and return the number of this kind of points. 

Complexity = O(limit * N * K)



