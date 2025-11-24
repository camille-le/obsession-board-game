### Testing out image editing


Many of OpenCV's predefined filters use a kernel. A kernel is a set of weights that determines how each output pixel is calculated from a neighborhood of pixels. 

A kernel is also known as a convolution matrix. It mixes up or convolves the pixels in a region. A kernel-based filter may also be called a convolution filter.

#### Edge-Finding Filters
Filters include `Laplacian`, `Sobel` and `Scharr`. 

`Laplacian` produces bold edge lines, especially in gray scale images. 

They turn non-edge regions into black and turn edge regions into white or saturated colors.

They are prone to misidentifying noise as edges. This can be mitigated by blurring an image before trying to find its edges.

#### Blurring Filters 
Filters include `blur` (a simple average), `medianBlur` and `GaussianBlur`. 

`mediaBlur` is effective in removing digital video noise, especially in color images. 

The arguments for both edge-finding and blurring filters always include `ksized` which is an odd number that represents the width and height (in pixels) of a filter's kernel. 

#### Filter 2D
Open provides the `filter2D` function that can apply any kernel or convolution matrix. 