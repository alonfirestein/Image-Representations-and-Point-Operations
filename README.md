# Image-Representations-and-Point-Operations  
  
  
### Image Processing and Computer Vision Course Assignment 1:


*In this assignment the following tasks were implemented using Python and the OpenCV library:*
- Reading an image into a given representation
- Displaying an image
- Transforming an RGB image to YIQ color space
- Transforming an YIQ image to RGB color space
- Histogram equalization
- Optimal image quantization
- Gamma Correction


## Image Outputs of the tasks listed above using the OpenCV and Matplotlib libraries:

##### Transforming an RGB image to YIQ color space:  
  
![Figure_1](https://user-images.githubusercontent.com/57404551/114478347-b2e4ad80-9c06-11eb-98a3-7c162e79de01.png)
  
  
## Histogram Equalization:  
The idea is to make the distribution of the intensities of the image(the colours of each pixel ranging from 0 to 255) more balanced.

To do that, first we create a histogram of the intensities, then compute the CDF(cumulative distribution function) of the distribution. 
The main idea is to straighten out the CDF of the given picture and making it linear by mapping a given intensity i to another intensity j such that: CDF(j)= j/255 from all pixels in the picture.  
  
  
##### Histogram equalization For Grayscale Image:
  
![Figure_2](https://user-images.githubusercontent.com/57404551/114478423-dd366b00-9c06-11eb-81ef-676292d952f5.png)

##### Original Grayscale Image:  
![Figure_3](https://user-images.githubusercontent.com/57404551/114478449-ecb5b400-9c06-11eb-8f18-72a1fb8f5588.png)

##### Equalized Grayscale Image:  
![Figure_4](https://user-images.githubusercontent.com/57404551/114478481-048d3800-9c07-11eb-8696-a322245a87a1.png)

##### Original RGB Image:  
![Figure_7](https://user-images.githubusercontent.com/57404551/114478513-140c8100-9c07-11eb-87d5-d1427639ba77.png)

##### Equalized RGB Image:  
![Figure_6](https://user-images.githubusercontent.com/57404551/114478532-1ec71600-9c07-11eb-9fef-0d2378d43d1a.png)

##### Displaying the cumsum: (Red is the original images cumsum, Green is the equalized images cumsum)  
![Figure_5](https://user-images.githubusercontent.com/57404551/114478616-52a23b80-9c07-11eb-8c41-8abc06bae0e6.png)  

## Optimal Quantization:  
Quantization, involved in image processing, is a lossy compression technique achieved by compressing a range of values to a single quantum value. When the number of discrete symbols in a given stream is reduced, the stream becomes more compressible.  
Color quantization reduces the number of colors used in an image; this is important for displaying images on devices that support a limited number of colors and for efficiently compressing certain kinds of images.  

##### Quantized Grayscale Image:  
![Figure_9](https://user-images.githubusercontent.com/57404551/114478864-e96ef800-9c07-11eb-9104-755ad33b5c73.png)

##### Quantized Grayscale Image Error Plot:  
![Figure_10](https://user-images.githubusercontent.com/57404551/114478951-1ae7c380-9c08-11eb-8769-a9548c689641.png)

##### Quantized RGB Image:  
![Figure_11](https://user-images.githubusercontent.com/57404551/114478989-2e932a00-9c08-11eb-9f20-ed99d89be10a.png)

##### Quantized RGB Image Error Plot:  
![Figure_12](https://user-images.githubusercontent.com/57404551/114479006-34890b00-9c08-11eb-8951-d0f11026bd94.png)


#### Gamma Correction: (Showing 3 different gamma values to show the difference)  
Gamma correction or gamma is a nonlinear operation used to encode and decode luminance or tristimulus values in video or still image systems.
![Screen Shot 2021-04-12 at 20 31 16](https://user-images.githubusercontent.com/57404551/114479277-e6283c00-9c08-11eb-985e-2828a5b20cad.png)  
  
    
    
![Screen Shot 2021-04-12 at 20 31 01](https://user-images.githubusercontent.com/57404551/114479288-f04a3a80-9c08-11eb-9c67-837d0c98b5ef.png)  
  
    
    
![Screen Shot 2021-04-12 at 20 32 42](https://user-images.githubusercontent.com/57404551/114479308-fa6c3900-9c08-11eb-9a9a-f38841f60dd9.png)






  



