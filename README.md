# DepthDSA
Python code to recover depth from projected synthetic 2D Digital Subtraction Angiography (DSA). 

This plugin build a depth map from a single DSA image using semantic segmentation.
Synthetic data are generated using [gvirtualxray](http://gvirtualxray.sourceforge.net/gvirtualxray.php)

This code uses Keras on Tensorflow and follows a simple UNet architecture.


![Learning Depth from a Single 2D Digital Subtraction Angiography](https://github.com/rouge1616/3DDSA/blob/master/3DDSA.png)
