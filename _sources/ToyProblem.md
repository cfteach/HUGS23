# Toy problem

This section outlines the toy Problem that is used for the hands-on session through out the lectures with the goal to introduce optimization technique and useful frameworks in python.


## Problem Statement

Consider a toy wire based tracker {numref}`toy-tracker`. the toy wire based tracker consists in a 2D tracking system with of 4 layers of wires along `z`.


```{figure} ./images/toy_tracker.png
:name: toy-tracker

Toy wire based Tracking system
```
A total of 8 parameters can be tuned. The adjustable parameters are the radius of the wire, the pitch (along the y-axis), and the shift along y and z of a plane with respect to the previous one.

Straignt tracks are generated at different angles and random origin. The tracker geometry and random origin. The
tracker geometry and the tracker generation is already defined in the imported module [`detector.py`](https://github.com/cfteach/modules/blob/master/detector2.py)

## The Objectives

One can evaluate the performance of the tracker on multiple metrics. These are are refered to as `objectives`. In the following lectures, the `objectives` against which the performance of the tracker will be evaulated are,

+ `Efficiency` - Defined as the fraction of tracks that has atleast 2 wires hit in the event.

+ `Volume` - Volume occupied by the tracking system. This is a proxy for the cost of the tracker.

+ `Resolution` - Resolution extracted from the tracks.
