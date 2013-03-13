em4gmm
======

Fast and clean C implementation of the Expectation Maximization (EM) algorithm for estimating Gaussian Mixture Models (GMMs).

Speed Results
-------------

Compiling with GCC 4.7 on Mac Os X (2,66GHz Intel Core 2 Duo):

     Training with 134479 samples (11 dimensions) using 128 mixtures takes 11.31 second.
     Classify 134479 samples (11 dimensions) using 128 mixtures takes less than 1 second.

Data Files
----------

The data files used by this software are very simple. They are plain text files of decimal numbers, with a header, and one line per sample vector. This is an expmple:

     AKREALTF
     11     4
     1025     7706     6830     5571     4169     2858     1809     1094      688      500      417
     1147     5755     6636     6234     4118     4593     2750     3649      774     1568     1104
      932     5381     5567     5175     3613     3499     2429     2536      652      913      337
      838     6401     5961     5277     4418     3468     2516     1644      921      391       74

On the header, the first number are the dimension and the second the number of samples. The sample's vectors can be integers or decimals (using "." as separator), and the dimensions must be space-separated.

Compiling
---------

On Mac Os X and Linux distributions you can simple use the make command on the shell.
On Windows you can import all to a Dev-Cpp project (for instance) and compile from it.

Usage
-----

You can train a model using the gmmtrain utility on a feature train file:

     ./gmmtrain <mixtures> <features> <model> [sigma]
          mixtures: number of gaussian mixtures
          features: feature file described before
          model: the place to save the resultant model
          sigma: optional stop criterion (usually 0.1-0.001)

Also, yo can obtain the score/log-probability of one feature test file:

     ./gmmclass <features> <model> [world]
           features: feature file described before
           model: model used to classify the features
           world: optional world model to uniform data

The standard process is to train a model for each class, and then classify at the class with highest probability.

License
-------
     Expectation Maximization for Gaussian Mixture Models.
     Copyright (C) 2012-2013 Juan Daniel Valor Miro
     
     This program is free software; you can redistribute it and/or
     modify it under the terms of the GNU General Public License as
     published by the Free Software Foundation; either version 2 of
     the License, or (at your option) any later version.
     
     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
     General Public License for more details.

