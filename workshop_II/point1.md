# Spectral Clustering

<h4>Spectral clustering is a graph-based algorithm for finding k clusters with arbitrary shapes in data. The technique involves the representation of data in a low dimension. In the low dimension, the data clusters have greater separation, which allows you to use algorithms such as the formation of k-means or k-medoids clusters. This low dimension is based on eigenvectors of a Laplacian matrix. A Laplacian matrix is ​​a way of representing a similarity graph that models the local proximity relationships between data points as an undirected graph. You can use spectral clustering when the number of clusters is known, but the algorithm also provides a way to calculate the number of clusters in the data.
Method that searches for that partition of the graph that groups vertices with high weight and separates groups where the vertices between groups have low weight.

## Cases might it be more useful to apply
Some cases in which spectral clustering is commonly used or applied:

1. Social Network Analysis: In social network analysis, spectral clustering can be used to identify communities within a network, where nodes (users) within the same community have stronger connections with each other than with external nodes.

2. Image Segmentation: In image processing, spectral clustering can be useful to segment images into regions with similar characteristics. This can be useful in applications such as segmenting objects in medical images or identifying objects in satellite images.

3. Text Analysis: In text analysis, spectral clustering can be applied to group similar documents or words based on the semantic similarity between them. This can be useful in organizing large collections of documents, retrieving information, or detecting themes in texts.

4. Signal Processing: In signal processing, spectral clustering can be used to analyze patterns in time series data, such as biomedical signals, audio signals, or sensor data. For example, it can be applied to identify anomalous activities or recurring patterns in patient monitoring data.

5. Pattern Recognition: In general, spectral clustering can be applied in pattern recognition problems where the data are not linearly separable in the original space, since it can capture complex and not necessarily convex structures.

6. Data Compression: Spectral clustering can also be used in data compression techniques, where similar data is grouped together and represented with less information to reduce the size of the data while maintaining relevant information.

## The mathematical fundamentals

The mathematical foundations of spectral clustering include representing data in a feature space, measuring similarity between data, constructing the graph Laplacian, spectral breaking, and clustering eigenvectors. These concepts are fundamental to understanding how spectral clustering is applied to group data in practice.


## What is the algorithm to compute it?
The spectral clustering algorithm involves several steps to compute clusters from a given dataset. Here's a high-level overview of the algorithm:

1. **Construct Similarity Graph**: Given a dataset \( X \) consisting of \( n \) data points, the first step is to construct a similarity graph \( G \). This graph represents the relationships between data points. Common methods for constructing the graph include k-nearest neighbors graph, epsilon-neighborhood graph, or fully connected graph with weights defined by a similarity measure (e.g., Gaussian kernel).

2. **Compute Graph Laplacian**: Once the similarity graph is constructed, compute the graph Laplacian matrix. There are different types of Laplacians, including the unnormalized Laplacian, normalized Laplacian, and symmetric normalized Laplacian. The choice of Laplacian depends on the specific problem and desired properties. The Laplacian matrix is usually defined as \( L = D - W \), where \( D \) is the degree matrix (a diagonal matrix with node degrees as diagonal elements) and \( W \) is the weighted adjacency matrix of the graph.

3. **Compute Eigenvalues and Eigenvectors**: Compute the eigenvalues and eigenvectors of the Laplacian matrix \( L \). Typically, you compute the smallest \( k \) eigenvectors (also known as the "first \( k \) eigenvectors") corresponding to the smallest eigenvalues. These eigenvectors form a matrix \( U \).

4. **Cluster Eigenvectors**: Use a clustering algorithm (commonly k-means) on the rows of the matrix \( U \) to assign data points to clusters. Each row of \( U \) corresponds to a data point, and the \( i \)th coordinate of the \( j \)th row in \( U \) represents the \( j \)th component of the \( i \)th eigenvector. The number of clusters \( k \) is typically specified by the user or determined by other criteria (e.g., silhouette score).

5. **Assign Data Points to Clusters**: After clustering, assign each data point to the cluster determined by the clustering algorithm applied to the eigenvectors.

6. **Output**: Return the clustering result, which consists of the cluster assignments for each data point.

It's important to note that spectral clustering is a broad class of algorithms, and variations exist based on different choices for constructing the similarity graph, Laplacian matrix, and clustering technique. Additionally, there are optimizations and extensions to improve efficiency and handle specific scenarios, such as large-scale data or non-standard data distributions.

## Does it hold any relation to some of the concepts previously mentioned in class? Which, and how?

PCA, SVD, TSNE, and UMAP are dimensionality reduction techniques that are based on spectral matrix theory and are used to visualize high-dimensional data. Spectral clustering is a data clustering technique that is based on spectral matrix theory and is used to group data into clusters.