This is a Python 3 implementation of the Visual Information Fidelity (VIF) Image Quality Assessment (IQA) metric. This code uses the [pyrtools](https://pyrtools.readthedocs.io/en/latest/) library to compute the Steerable Pyramid decomposition, and [integral images](https://en.wikipedia.org/wiki/Summed-area_table) to accelerate filtering by averaging filters.

To compute VIF, follow this template.
```
# img_ref, img_dist are two images of the same size.
from vif_utils import vif
print(vif(img_ref, img_dist))
```
