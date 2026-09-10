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

## after proseg
- [ ] run ovrlpy on proseg-separated transcripts
    - [ ] segment with proseg
    - [ ] separate transcripts of each layer
        - a layer should be define by the thicknes of the sample