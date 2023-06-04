# EIC Tracking system

The EIC Detector 1 tracking detector consists of the silicon vertex/tracking detector and $\mu$Rwell gas tracker. The AC-LGAD based Time of Flight (ToF) detector will provide hit information for the tracking reconstruction.

The EIC Detector 1 concept is classified based on its pseduo rapidity $\eta$ as shown in {numref}`eic-detector-coverage`.

```{figure} ./images/eic-detector-coverage.png
---
height: 300px
name: eic-detector-coverage
---

EIC Detector 1 $\eta$ coverage
```


## Central Barrel Region

| **Endcap Region<br>Detector Subsystem** 	| **No Of Layers** 	| **Technology** 	| **Pitch/res<br>[$\mu$m]** 	| **Thickness<br>[X/X0]** 	|                                             **Description**                                            	|
|:----------------------------------------:	|:----------------:	|:--------------:	|:-------------------------:	|:-----------------------:	|:------------------------------------------------------------------------------------------------------:	|
|               Vertex Barrel              	|         3        	|    MAPS-ITS3   	|             10            	|           0.05          	|               Monolithic Active Pixel Sensor; EIC R&D eRD111.<br>High precision tracking.              	|
|              Sagitta Barrel              	|         2        	|    MAPS-ITS3   	|             10            	|           0.05          	|               Monolithic Active Pixel Sensor; EIC R&D eRD11. <br>High precision tracking.              	|
|               Outer Barrel               	|         3        	|   $\mu$Rwell   	|             55            	|           0.2           	|             𝛍Rwell is a gaseous based tracker. EIC R&D ERD6.<br> Low Cost tracking solution            	|
|                CTTL (TOF)                	|         1        	|     AC-LGAD    	|             30            	|           ~0.1          	| Low Gain Avalanche Detectors (ACLGAD): EIC R&D ERD112. <br>High precision tracking and Time of flight. 	|

## End cap region (Both Forward and Backward region)

| **Central Region<br>Detector Subsystem** 	| **No Of Layers** 	| **Technology** 	| **Pitch/res<br>[$\mu$m]** 	| **Thickness<br>[X/X0]** 	|                                      **Description**                                      	|
|:----------------------------------------:	|:----------------:	|:--------------:	|:-------------------------:	|:-----------------------:	|:-----------------------------------------------------------------------------------------:	|
|                    EST                   	|         4        	|    MAPS-ITS3   	|             10            	|           0.3           	|        Monolithic Active Pixel Sensor; EIC R&D eRD11. <br>High precision tracking.        	|
|                    FST                   	|         5        	|    MAPS-ITS3   	|             10            	|           0.3           	|        Monolithic Active Pixel Sensor; EIC R&D eRD111. <br>High precision tracking.       	|
|                   ETTL                   	|         1        	|     AC-LGAD    	|             30            	|           ~0.1          	| Low Gain Avalanche Detectors (ACLGAD): EIC R&D ERD112. High precision tracking and timing 	|
|                   FTTL                   	|         1        	|     AC-LGAD    	|             30            	|           ~0.1          	| Low Gain Avalanche Detectors (ACLGAD): EIC R&D ERD112. High precision tracking and timing 	|
