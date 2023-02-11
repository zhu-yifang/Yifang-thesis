# Yifang-thesis
<!-- TO DO -->
- [x] Read all the files
- [x] Get all the phones
- [x] Get the MFCC sequences of all the phones
- [x] Train
- [x] Test
- [x] Get the number of each phone
- [ ] Remove the slicence and pause and do the folding
- [ ] Build Confusion matrix, get 50 samples for each phone
- [ ] Try reduced training sets
- [ ] Investigate normalizing instances
- [ ] Power spectra, diff MFCC features
- [ ] Clustering training instances
- [ ] Write up what I have

<!-- Notes -->
The stats of the training set: Counter({'h#': 9240, 'ix': 8642, 's': 7475, 'n': 7068, 'iy': 6953, 'tcl': 6644, 'r': 6539, 'kcl': 5859, 'l': 5801, 'ih': 5051, 'dcl': 4942, 'k': 4874, 't': 4364, 'ae': 3997, 'm': 3903, 'eh': 3853, 'z': 3773, 'ax': 3610, 'q': 3590, 'd': 3548, 'axr': 3407, 'w': 3140, 'aa': 3064, 'ao': 2940, 'dh': 2826, 'dx': 2709, 'pcl': 2644, 'p': 2588, 'ay': 2390, 'ah': 2306, 'ey': 2282, 'sh': 2238, 'gcl': 2223, 'f': 2216, 'b': 2181, 'ow': 2136, 'er': 2046, 'g': 2017, 'v': 1994, 'bcl': 1909, 'ux': 1908, 'y': 1715, 'epi': 1464, 'ng': 1330, 'jh': 1209, 'hv': 1154, 'nx': 971, 'hh': 957, 'pau': 952, 'el': 951, 'ch': 822, 'th': 751, 'aw': 729, 'en': 723, 'oy': 684, 'uw': 555, 'uh': 535, 'ax-h': 375, 'zh': 151, 'em': 124, 'eng': 38})

The stats of the testing set: Counter({'h#': 3360, 'ix': 2945, 'iy': 2710, 's': 2639, 'r': 2525, 'n': 2501, 'l': 2356, 'tcl': 2334, 'kcl': 1964, 'ih': 1709, 'dcl': 1643, 'k': 1614, 't': 1535, 'm': 1526, 'eh': 1440, 'ae': 1407, 'axr': 1383, 'ax': 1346, 'z': 1273, 'd': 1245, 'q': 1244, 'w': 1239, 'ao': 1156, 'aa': 1133, 'dh': 1053, 'pcl': 965, 'p': 957, 'dx': 940, 'f': 912, 'b': 886, 'ah': 879, 'ay': 852, 'gcl': 808, 'ey': 806, 'er': 800, 'sh': 796, 'ow': 777, 'bcl': 776, 'g': 755, 'v': 710, 'y': 634, 'ux': 580, 'epi': 536, 'ng': 414, 'pau': 391, 'jh': 372, 'hv': 369, 'nx': 360, 'hh': 356, 'el': 343, 'th': 267, 'oy': 263, 'ch': 259, 'en': 251, 'uh': 221, 'aw': 216, 'uw': 170, 'ax-h': 118, 'zh': 74, 'em': 47, 'eng': 5})

Remove h#, #h, sil, pau, epi

merge/fold (axr, er), (m, em), (n, en, nx), (ng, eng), (pcl, p), (tcl, t), (kcl, k), (bcl, b), (dcl, d), (gcl, g)