# LPI-deepGBDT

## Data
Data is available at [NONCODE](http://www.noncode.org/), [NPInter](http://bigdata.ibp.ac.cn/npinter3/index.htm), and [PlncRNADB](http://bis.zju.edu.cn/PlncRNADB/).
# Feature selection
LncRNA feature: [PyFeat](https://github.com/mrzResearchArena/PyFeat)
Protein feature: [Biotriangle](http://biotriangle.scbdd.com/protein/index/)

# System Requirements

The LPI-deepGBDT is supported on python 3.

torch version=1.4.0 

scikit-learn version=0.23.2

numpy version=1.17.4 

pandas version=1.1.5

## Dimension Reduction

[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html?highlight=pca#sklearn.decomposition.PCA)

## Run model 

```python
python example/example_CV1_2.py
python example/example_CV3.py
```

