# dimensionality-reduction
We try to compare the the effect of reducing dimension between pca,t-sne,auto-encoder in MNIST. We use the first two thousand samples according to tensorflow.tutorals <br>
![image](https://github.com/learnerRen/dimensionality-reduction/blob/master/pca.png)<br>
Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. PCA is a famous linear dimensionality reduction method, however, it is not useful in high denebsional samples. 
![image](https://github.com/learnerRen/dimensionality-reduction/blob/master/tsne.png)<br>
t-sne
[Maaten L V D, Hinton G. Visualizing Data using t-SNE[J]. Journal of Machine Learning Research, 2008, 9(2605):2579-2605.](https://www.seas.harvard.edu/courses/cs281/papers/tsne.pdf)
t-sne is a famous non-linear dimensionality reduction method, and it is very powerful. From 784 dimensional dimension reduction to 2 dimension, it still can maintain fairly good visualization effect, so we can see that this algorithm is very powerful. Of course, This algorithm is very complex in mathematics...
![image](https://github.com/learnerRen/dimensionality-reduction/blob/master/autoencoder.png)<br>
As for auto-encoder, we can see the effect of dementional redunction is very great, but I strongly recommend that you'd better use t-sne.<br>
Why?<br>
(1)autoencoder is very easy to overfit.(2) We need to add noise, dropout, regularizer and so on, it is very high computationally
