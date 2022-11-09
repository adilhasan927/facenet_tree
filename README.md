# Facenet TF
Adaption of Facenet to perform one shot learning for dictionary classification of tropical rainforest species.

FaceNet is a fully end-to-end, one-shot learning deep learning algorithm that did not require significant amounts of data. FaceNet was modified to accommodate multispectral images for greater data aggregation. The combination of the different methods yielded a test accuracy of 54% across 15 classes.

- `Triplet Loss` : A straight-forward procedure which finds triplets by iterating through all triplets formed.
- `Hard Triplet Loss` : A procedure which chooses triplet pairs such that distance b/w -ve and anchor is less and the distance b/w anchor and +ve is more.
- `Adaptive Triplet Loss` : The main idea to correct the triplet selection bias, so we try to minize the distribution shift between the batch and the tripet set.

### Acknowlegments

 - Dr. Ji Jon Sit from NTU EEE, for providing all the insights in this project, especially when dabbling with the greater granularities when it came to 

### References

```
F. Schroff, D. Kalenichenko and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering,
" 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, MA, 2015, 
pp. 815-823, doi: 10.1109/CVPR.2015.7298682.
```

```
Yu B., Liu T., Gong M., Ding C., Tao D. (2018) Correcting the Triplet Selection Bias for Triplet Loss. 
In: Ferrari V., Hebert M., Sminchisescu C., Weiss Y. (eds) Computer Vision – ECCV 2018. ECCV 2018. 
Lecture Notes in Computer Science, vol 11210. Springer, Cham. https://doi.org/10.1007/978-3-030-01231-1_5
```
