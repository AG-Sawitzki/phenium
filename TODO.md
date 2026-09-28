# general:
- [ ] quantify overlap
    - [ ] in raw
    - [ ] after proseg
        - [ ] pure
        - [ ] with priors
- [ ] quantify overlap-change
- [ ] investigate cell-type in IHC for proseg-identified overlap-regions

# quantify overlap
- [ ] run ovrlpy on raw
    - [X] what is the output?
        - UMAP & pseudocells
        - tissue signal integrity
        - doublets
        - VSI & doublets of ROI
- [ ] visualize quantity?
    - average of region?
    - compare signal reduction and integrity-change
        - abs signal difference
        - avg integrity before & after (?)

# Imag Registration
- [ ] stars this is taking ages...
- tried:
    - OPCV ECC + Homography
    - OPCV SIFT + FLANN + Homography
    - OPCV SIFT + BF + Homography
    - OPCV ORB + FLANN + Homography
    - OPCV ORB + Homography
    - DIPY Affine
    - DIPY Diffeomorphic
    - simpleITK M1-4, M3 works "best"
- trying with scaled IHC img!
    - OPCV ECC + Homography
    - OPCV SIFT + FLANN + Homography
    - OPCV SIFT + BF + Homography
    - simpleITK M3<M1<M2~M4
    - SITK with BSpline warps image. doesn't translate.
    - DIPY so far best performance

# after proseg
- [ ] run ovrlpy on proseg-separated transcripts
    - [ ] segment with proseg
    - [ ] separate transcripts of each layer
        - a layer should be define by the thicknes of the sample

# SC Analysis
- [ ] PCA
- [ ] Nearest Neighbour
- [ ] UMAP
- [ ] leiden?

# miscellaneous
- [ ] pixel size must be calculated considering Physical Unit (296?)