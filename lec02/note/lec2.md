# Lec2 for ML-4360
## 2.1 Primitives & Transformations
### $Homogeneous$ 
* homogeneous coordinates & inhomogeneous coordinates
* argmented vector[one element out of the whole equivalent class],homogeneous vectors,homogeneous vectors
* points at infinity
![Alt text](<a.png>)
* In homogeneous coordinates, the intersection of two lines is given by:
>$\tilde{x}$=$\tilde{l1}$ × $\tilde{l2}$ 
![Alt text](<b.png>)
* the line joining two points can be compactly written as:
> $\tilde{l}$=$\tilde{x1}$ × $\tilde{x2}$ 

### $Transformations$
* translation 2DOF
* Euclidean 3DOF
![Alt text](<c.png>)
* 正交矩阵$A^TA=I$

>$A^{-1}=A^T$

>$\left|A\right|$ = +-1

>A的行（列）向量组为n维单位正交向量组
* Similarity 4DOF
* Affine: 6DOF
>arbitrary $2×2$ matrix

> Parallels!
* Projective :8DOF——preserve straight lines
>> DOF（2D）:n(n-1)/2

![Alt text](<d.png>)
>$\tilde{l}'$ = $(\tilde{H}^T)^{-1}$$\tilde{l}$
### Direct Linear Transform for Homography Estimation[algorithm DLT]

