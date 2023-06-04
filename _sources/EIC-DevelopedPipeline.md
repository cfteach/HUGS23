# Developed Pipeline

## Fun4All framework

[Fun4All framework](https://eic.github.io/software/fun4all.html) (based on `GEANT4`) was used to simulate the detector response of a given design solution.

The existing framework of Fun4All had to be modified to incorporate the dynamically changing detector dimensions. Therefore, separate version of Fun4All framework was maintained for this project. However, the latter was always updated to reflect the most up-to-date version of Fun4All framework.


## The optimization plugin

Currently, the framework was developed to support 2 types of optimization libraries in python

+ Multi-objective optimization using `pymoo` framework (specifically `NSGA-II` with `GA`)

+ Multi-objective Bayesian optimizing using `Ax` framework (`qNEHVI + SAASBO`)

## Checks on design point

Each time the optimization algorithm suggests a point, a series of checks are run before and after the its simulation.
This is to ensure not only the feasiblity of the design solution but also to the checks serve as a QA test of the simulated sample. THe details of the checks are outline in the figure below.

```{mermaid}
graph TD
    A[/<h3><strong> Design Point </strong></h3>\] --> B(<strong>Check Strong Constraints</strong> <br> <small> Cannot be broken  <br> like Engineering constraints </small>)
    B --> |fa:fa-check passed| C(<strong> GEANT4 model overcheck </strong> <br> <small> Overlaps could cause <br> unphysical interactions in GEANT4 </small>)
    B --> |fa:fa-times Fail| G{{Penalize heavily <br> <small> Impossible Design fa:fa-fire </small>}}
    C -->|fa:fa-check passed| D(<strong> Start simulation </strong> <br> <small>with timeout. <br> HPC Cluster job fails </small>)
    C -->|fa:fa-times Fail| E{{Penalize heavily <br> <small> Alert User fa:fa-exclamation fa:fa-bullhorn </small>}}
    D -->|fa:fa-check Finished| F(Additional checks on result <br> & Evaulate Performance)
    D -->|fa:fa-clock Timeout| H{{Do not penalize <br> omit design point}}
    F -->|fa:fa-check All Good| I(Healthy design point)
    F -->|fa:fa-times Fail| J{{Fits Failed <br> <small> Dont penalize <br> Omit design </small>}}
```

## Putting it all together

Putting all the pieces together, The overall pipeline is summarized below.

```{figure} ./images/eic-optimization-pipeline.png
---
name: eic-optimization-pipeline
height: 300px
---

This figure has to be changes and I am making a figure similar to the mermaid one above.
```
