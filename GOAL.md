# Laser Detector

## Observations

* Lasers take a small portion of the image
* The path the laser takes through the image is consistent
* We don't need to consider an entire image to find the laser
* The laser may shift in color
* We have both green and blue lasers

## How humans label

* Look at the entire image to get context
* if there is a fish or object towards the center, zoom into it
* a laser is often on the object of interest
* when zoomed in, the user may pan to confirm the location of the user
* the user finally selects the laser

## Idea

* We can use a ViT to get context
* We can have a reinforcement learning model which can produce a policy that has the options to zoom, pan, and select the center of the screen as the laser.  This mirrors how a human finds the laser

## Data

* We have 60,000 labels.  Not all of these are created equal.
* All labels from the same dive need to be colinear.  We'll have to test for this
* All of the labels are available using the fishsense-sdk.  which you can find on github UCSD-E4E/fishsense-lite