CUDA Image Filters
==================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Nischal K N
* Tested on: Windows 10, i7-2670QM @ 2.20GHz 8GB, GTX 540 2GB (Personal)

### SUMMARY
This Project implements the following image processing algorithms on CPU and GPU
* Tone Mapping: It is used to map one set of colors to another to approximate the appearance of high-dynamic-range images in a medium that has a more limited dynamic range. It is similar to the HDR mode of camera.
* Seam Carving for content aware resizing: It uses low energy seams to remove redundant pixels in each row/column to resize the image. This way the object of importance have a significant size even when resized.
* Red eye Reduction: It is a filter used to remove red pupils in color photographs of eyes when captured with a flash under low ambient light condition.
* Edge Preserving Blur: It is a noise-reducing smoothing filter for images while preserving edges.

A CPU and GPU implementation of the above 4 algorithms were evaluated for variation in performance of various stages of each filter. Also additional performance improvements are documented.

### BUILD INSTRUCTIONS
* Setup cuda development environment use [this guide](https://github.com/nischalkn/Project0-CUDA-Getting-Started/blob/master/INSTRUCTION.md#part-1-setting-up-your-development-environment)
* Download openCV 2.4.13 from [here](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/)
* Set the environment variables as shown [here](http://opencv-srf.blogspot.com/2013/05/installing-configuring-opencv-with-vs.html)
* Create a new CUDA project and configure the project following from step 4 of **Configure Visual Studio Express** section in the above link.

### Tone Mapping
Tone-mapping is a process that transforms the intensities in the image so that the brightest values aren't nearly so far away from the mean. That way when we transform the values into [0-255] we can actually see the entire image. It uses 3 steps to map the image. First is to Convert the image to Chrominance-Luminance(YUV) color space. Then we calculate the histogram and CDF of the Luminance channel. Finally we remap the intensity values. The following left image is the input image and the right image is the corresponding output image.

Input Image                |  Output Image
:-------------------------:|:-------------------------:
![](image/input1.jpg)      |  ![](image/output1.png)

They GPU implementation of this algorithm uses 4 kernels
* Color space conversion kernel: rgb2xyY
* Histogram calculation kernel: histogram_kernel
* Normalized CDF calculation kernel: scan_kernel and normalized_cdf
* Final mapping kernel: tonemap

The performance of CPU and GPU implementation of tone mapping with different block sizes is shown in the figure below. It is seen that the GPU variants performs twice faster than the CPU implementation

![](image/graph1a.jpg)

The average time taken by different stages of the algorithm is documented below. It is seen that all of the stages perform much better on the GPU. The tone mapping stage has the highest improvement.
![](image/graph1b.jpg)

Variation of different stages of the algorithm for different block sizes is shown in the image below. The kernel to find minimum and maximum of the array and the tone mapping kernel have a sharp improvement in performance with mid size kernel compared to small sizes. This is because each min/max calls involves multiple calls of the kernel which has a high overhead with small block sizes.
![](image/graph1c.jpg)

[comment]: <> (additionally the performance of the algorithm was further improved by changing .... The effect of this can be seen in the following image.)

[comment]: <> (![](image/graph3.png))

### Seam Carving
Seam carving is a process of finding a link of pixels from top to bottom or left to right such that it represents the minimum energy seam. By removing this seam we can resize the image without reducing the size of critical objects in the frame [[1](http://perso.crans.org/frenoy/matlab2012/seamcarving.pdf)].The purpose of the algorithm is image retargeting, which is the problem of displaying images without distortion on media of various sizes (cell phones, projection screens) using document standards, like HTML, that already support dynamic changes in page layout and text but not images.
In the following image, the left image is the input image, the center image resized naively by 100x100 pixels and the right most image is content aware resized by 100x100 pixels.

Input Image                |  Naive Resizing           |  Seam carving            
:-------------------------:|:-------------------------:|:------------------------:
![](image/input2.png)      |  ![](image/output2a.png)  | ![](image/output2.jpg)  

It involves 3 stages, first stage is to generate the energy map of the image. Second is to calculate the minimum energy seam and finally to remove the pixels along the minimum energy seam.

The GPU implementation of this algorithm uses 3 kernels
* Energy distribution of the Image: computeEnergyGPU
* Find minimum energy seam
* Removing pixels along the seam using Stream Compaction

The performance of these stages on CPU and GPU have been recorded and shown in Fig.

![](image/graph2a.jpg)
![](image/graph2b.jpg)
![](image/graph2c.jpg)

It is seen that the

### Red Eye Reduction
Red eye reduction is an algorithm used to remove the effect of red pupil on images of faces taken under flash in low ambient light conditions. This is done by first finding the normalized cross correlation scores using the template of a red eye. Then the next step is to sort the pixels according to the decreasing order of the normalized scores and finally remapping the red channel of pixels with high correlation score to remove the red eye effect. The following image shows the image before and after red eye reduction filter

Input Image                |  Output Image
:-------------------------:|:-------------------------:
![](image/input3.jpg)      |  ![](image/output3.jpg)

The following are the 3 major kernels involved in the red eye reduction process
* Finding the normalized cross correlation scores
* Sorting the normalization scores
* Remapping of the red eye pixels

The performance of these stages on CPU and GPU have been recorded and shown in Fig.

![](image/graph3a.jpg)
![](image/graph3b.jpg)
It is seen that the

### Edge Preserving Blur

Input Image                |  Output Image
:-------------------------:|:-------------------------:
![](image/input4.jpg)      |  ![](image/output4.jpg)

The performance of these stages on CPU and GPU have been recorded and shown in Fig.

![](image/graph4a.jpg)

It is seen that the
