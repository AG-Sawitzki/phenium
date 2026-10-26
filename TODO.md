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
    - [x] what is the output?
        - UMAP & pseudocells
        - tissue signal integrity
        - doublets
        - VSI & doublets of ROI
- [ ] run ovrlpy on cell_ids in proseg's "cells"
- [ ] run ovrlpy on layers and cell_ids from proseg
- [ ] visualize quantity?
    - average of region?
    - compare signal reduction and integrity-change
        - abs signal difference
        - avg integrity before & after (?)
- [ ] intersection with geopandas
    - for each cell(layer) find coverage with all others cells
        - obv excluding itself

# Imag Registration
- [x] stars this is taking ages...
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
- [x] found ANTsPy !

# after proseg
- [-] run ovrlpy on proseg-separated transcripts
    - [-] segment with proseg
    - [-] separate transcripts of each layer
        - a layer should be define by the thicknes of the sample
    - ...
        - [-] transcripts_locations as points in gdf
            - shapely.geometry.Point works on pl.DataFrame **row**
        - [-] check if point in polygon
            - for each layer
        - [-] apply check-array on transcripts
        - [-] asign to new layer df

# cell and layer assignment
- [-] find layer cell is biggest in?
- [X] find cells containing prots (location!)
    - over all layers  
- [X] asign prot to cell
    - [X] as sparse matrix

## protein assigning
- pixel by pixel
    - signal above certain threshold
- identify cells covering that point
- find probabilities of belonging to cell1, cell2, ...
- assign fraction of signal to each cell above threshold
- distribute signal over cell layers by cell size

# SC Analysis
- [X] QC metric (a nice to have)
- [X] Nearest Neighbour
- [X] leiden?
- [X] UMAP
- [-] celltypist

# miscellaneous
- [ ] pixel size must be calculated considering Physical Unit (296?)
- [ ] z-slides based on transcript-distribution (phenium) and cell size (user) + tissue thickness (user)

# Präsi for Fr. 13.11.2026
- [ ] find other Deconvolution methods
- [ ] investigate QQ, Earth mover's distance & Minimum Spanning Tree