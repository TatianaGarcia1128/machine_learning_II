# DBSCAN

<h4>Starting from some points and an integer k, the algorithm aims to divide the points into k groups, called clusters, homogeneous and compact. Let's look at the following example:
<h4>

<img src="https://datascientest.com/es/wp-content/uploads/sites/7/2022/11/DBSCAN1.webp" alt="MarineGEO circle logo"/>

<br>
DBSCAN is a simple algorithm that defines clusters by estimating local density. It can be divided into 4 stages:

For each observation we look at the number of points at a maximum distance ε from it. This area is called the ε-neighborhood of the observation.
If an observation has at least a certain number of neighbors, including itself, it is considered a core observation. In this case, a high density observation has been detected.
All observations in the neighborhood of a central observation belong to the same cluster. There may be central observations close to each other. Therefore, from one step to the next, a long sequence of central observations is obtained that constitute a single cluster.
Any observation that is not a core observation and does not have any core observations in its vicinity is considered an anomaly.
Therefore, it is necessary to define two pieces of information before using DBSCAN:

What distance ε must be determined for each observation in the ε-neighborhood? What is the minimum number of neighbors needed to consider an observation as a central observation?

These two data are freely provided by the user. Unlike the k-means algorithm or ascending hierarchical sorting, the number of clusters does not need to be defined in advance, making the algorithm less rigid.

Another advantage of DBSCAN is that it also allows you to handle outliers or anomalies. In the previous figure it is observed that the algorithm has determined 3 main clusters: blue, green and yellow. The purple dots are anomalies detected by DBSCAN. Obviously, depending on the value of ε and the number of minimum neighbors, the partition can vary.


## Noción de distancia y elección de ε
What is the metric used to evaluate the distance between an observation and its neighbors? What is the ideal ε?

In DBSCAN the Euclidean distance is generally used, where p = (p1,….,pn) and q = (q1,….,qn):

<img src="https://datascientest.com/es/wp-content/uploads/sites/7/2022/11/dbscan2.webp" alt="MarineGEO circle logo"/>

In each observation, to count the number of neighbors at most a distance ε, we calculate the Euclidean distance between the neighbor and the observation and check if it is less than ε.

Now we need to know how to choose the correct epsilon. Suppose in our example we choose to test the algorithm with different values ​​of ε. Here is the result:

<img src="https://datascientest.com/es/wp-content/uploads/sites/7/2022/11/dbscan3.webp" alt="MarineGEO circle logo"/>

<br>

In all three examples, the minimum number of neighbors is always set to 5.

If ε is too small, the ε-neighborhood is too small and all observations in the data set are considered anomalies.

This is the case of the figure on the left eps = 0.05.

On the other hand, if epsilon is too large, each observation contains in its ε-neighborhood all other observations in the data set. Consequently, we obtain a single cluster. Therefore, it is very important to properly calibrate the ε to obtain a quality partition.

A simple method to optimize ε is to find for each observation how far away its nearest neighbor is. Then it is enough to set an ε that allows a "sufficiently large" proportion of the observations to have a distance to their nearest neighbor less than ε. By “large enough” we mean 90-95% of observations must have at least one neighbor in their ε-neighborhood.

It is particularly useful in cases where clusters have variable densities, irregular shapes, and there is noise in the data set. Some specific scenarios where DBSCAN could be most useful include:

1. Spatial Data Analysis: DBSCAN is often applied in geographic data analysis, such as identifying regions of similar land use in a satellite image or clustering GPS coordinates of points of interest.

2. Anomaly Detection: Due to its ability to handle noise and outliers effectively, DBSCAN is often used for anomaly detection tasks, where the goal is to identify data points that do not fit the overall pattern of the data.

3. Group Shape Variation: Unlike K-means, which assumes spherical groups, DBSCAN can identify groups of arbitrary shapes. This makes it useful in scenarios where groups have complex or irregular shapes.

4. Noisy Data: DBSCAN is robust to noise in the data. It can automatically classify outliers as noise, helping you focus on the main groups.


## The mathematical fundamentals
DBSCAN operates based on two parameters: epsilon (ε) and minPts.

Epsilon (ε): This parameter defines the radius within which to search for neighboring points.

minPts: This parameter specifies the minimum number of points required to form a dense region (center point).

The algorithm works as follows:

Center Point: A point is considered a center point if there are at least 'minPts' points (including itself) within a distance of 'ε' from it.

Edge Point: A point is considered an edge point if it is within distance ε of a center point but does not have enough neighbors to be considered a center point itself.

Noise Point: A point is considered noise if it is neither a center point nor an edge point.

The algorithm starts with an arbitrary point and expands the group by recursively adding points reachable from the center points. It ends when all points have been visited.

## Is there any relation between DBSCAN and Spectral Clustering? If so, what is it?

In some cases, these algorithms can be complementary. For example, DBSCAN can be used for initial noise reduction or outlier detection, followed by Spectral Clustering to identify clusters in the preprocessed data. Additionally, Spectral Clustering can sometimes be used to construct similarity graphs for DBSCAN when the distance metric is not well defined in the original feature space.


