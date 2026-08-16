### AUG,12,2026
- Storm path prediction is super cool, so i do it, here are some basic information:
> Project: Storm path prediction
> Dataset link: https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/besttrack.html

- 0:23 GMT+7: I've just downloaded the file, it have some weird format, but i can still handle it : )
- 0:27 GMT+7: now its time for some basic EDA, the data seems to missing in some places(just like all other datasets), i think i must first check out what time period did all the stuffs kinda full and train on it.
- 0:44 GMT+7: I should get some sleep : ), but this log seems weird, i think my code is wrong somewhere:
```
The number of rows are: 71223
Top 5 columns with the most missing data:
Max_Sustained_Wind_kt    71223
Landfall_Indicator       71223
Short_Rad_50kt_nm        42381
Dir_Long_Rad_30kt        42381
Long_Rad_30kt_nm         42381
dtype: int64

The column lacking the most data is 'Max_Sustained_Wind_kt' with 71223 missing values.
```
- 1:00 GMT+7: the data seems promising, somehow.
- 2:00 GMT+7: the knowledge about storms, are way too much...
- 2:27 GMT+7: super stress, the data is missing way too much stuffs... maybe a nap could help : )

### AUG,13,2026
- 10:12 GMT+7: back to coding again, slept like... alot, i've just realized that the missing percentage depends on the storm grade, weird... i wonder how can i use this, and the fetch_data.py did wrong, i naively use split which broke some code : (.
- 10:14 GMT+7: actually, the data is pretty good, now i must figure out which feature is suited to predict the storm path, the wind ones, everyone used time stuffs, its not really cool.
```
=== Overall Ranking ===
TC of TS intensity or higher: 76.92% missing
Tropical Depression (TD): 72.31% missing
Extra-tropical Cyclone (L): 72.05% missing
Entering RSMC area: 34.73% missing
Tropical Storm (TS): 7.64% missing
Typhoon (TY): 7.62% missing
Severe Tropical Storm (STS): 7.60% missing


=== 1977-present Ranking ===
Extra-tropical Cyclone (L): 69.23% missing
Tropical Depression (TD): 69.23% missing
Entering RSMC area: 28.91% missing
Tropical Storm (TS): 7.64% missing
Typhoon (TY): 7.62% missing
Severe Tropical Storm (STS): 7.60% missing


=== 1980-present Ranking ===
Extra-tropical Cyclone (L): 69.23% missing
Tropical Depression (TD): 69.23% missing
Entering RSMC area: 28.91% missing
Tropical Storm (TS): 7.64% missing
Typhoon (TY): 7.62% missing
Severe Tropical Storm (STS): 7.60% missing


=== 1990-present Ranking ===
Extra-tropical Cyclone (L): 69.23% missing
Tropical Depression (TD): 69.23% missing
Entering RSMC area: 25.64% missing
Tropical Storm (TS): 7.62% missing
Typhoon (TY): 7.60% missing
Severe Tropical Storm (STS): 7.57% missing
```
- 10:31 GMT+7, i guess i should begin training some kinds of models, first, i guess it should be LSTM or XGBoost... the loss function should take account for both the wrong of direction and magnitude over the next step, wish me luck, first i think i might train on raw data.
- 10:42 GMT+7, i've underestimated the stuffs about storm, gotta go checkout what is 30 and 50 knots wind in those storms and how would it affect the storm.
- 11:00 GMT+7: oh yeah so basically the storm have a range of wind speed surrounding ints centre, the 30 knots is uncool and 50 knots mean that your house can like, flyover : ), each of them spinning in different speed and direction, i wonderm how can i take advantage of these things to predict the storm movement, i guess its time for some bigger eda.
- 1:47 GMT+7: somehow, the model work suprisingly good, i seriously don't know what's wrong
- 2:10 GMT+7: actually this model is ok, not as bad as i think, even after i fixed all the data leakage problems, maybe its because this type of prediction rely heavily on momentum, hence it can do lots of magical stuffs, haha:
```
Evaluating models (Cosine Similarity)...
Average Cosine Similarity for 6h: 0.7464 | Mean Distance Error: 71.81 km
Average Cosine Similarity for 12h: 0.7560 | Mean Distance Error: 138.37 km
Average Cosine Similarity for 18h: 0.7617 | Mean Distance Error: 203.28 km
Average Cosine Similarity for 24h: 0.7687 | Mean Distance Error: 267.98 km
```
- 2:33 GMT+7: i don't really understand why the lat is so importance, i guess i gotta go search and learn some physics : )
- lots of physics, and after the test, i found out that the mode perform worse after each time step, its easy to understand easpecially when the storm goes wild, quite easy to understand, i think now i must train this thing on some kinds of model that can learn based on the past step : )... but first i gotta sleep...
- 21:39 GMT+7: i think that now i will try my luck on some deep learning models, but first, i gotta figure out to encode the storm's state in a single step, i gotta work with some kinds of encoder that can generate new relations between the storm's features and also be able to handle missing data...
- 21:56 GMT+7: after a while, i realized that there are 2 main candidate for encode and decode the tabular data:
```
HI-VAE (Heterogeneous Incomplete VAE)
VIME (Value Imputation and Mask Estimation)
```
I know there are some transformer based one, but i hate the attention of them cuz they are way too heavy, i want something lightweight, fast and percise, now i will investigate these two one, wish me luck...
- 23:23 GMT+7: HI-VAE seems to learn to encodde data based on the distribution of the whole table, which is, not good, because that thing could cause decoding become a serious challenge, plus it is super hard to implement, i think i will try my luck with VIME...
- 23:55 GMT+7: seriously, i truly need to think about some autoencoder that truly care about the current storm


### AUG,14,2026
- 0:26 GMT+7: the problem with VIME is that, it trying to learn from corrupted data, hence, it would be extremely dumb to just naively dump our data into VIME, aint no way vime can decipher lat lon from pressures, wind directions..., VIME would simply treat our data lines seperately but in reality, storms state deeply depends on each other, maybe i have to design one my own, it kinda hopeless haha.

### AUG,15,2026
- 11:39 GMT+7: after a while i realized, the more i read, the more i panic so i've just create a super simple autoencoder with just standard scaler, it result seems to be super cool:
```
===============================================================================================
123.9s	164	           PER-FEATURE PHYSICAL RECONSTRUCTION MATRIX (SCALED BACK TO REAL UNITS)
123.9s	165	===============================================================================================
124.0s	166	          Feature   Physical Range Physical MAE Physical RMSE Rel Error (%) Fidelity (%) R² (Physical)  Valid Records
124.0s	167	            grade       [2.0, 7.0]        0.022         0.030        0.447%      99.553%        0.9995 8,578 (100.0%)
124.0s	168	              lat      [1.4, 69.0]      0.064 °       0.095 °        0.095%      99.905%        0.9999 8,578 (100.0%)
124.0s	169	              lon    [98.0, 188.0]      0.261 °       0.305 °        0.290%      99.710%        0.9997 8,578 (100.0%)
124.0s	170	     pressure_hpa  [885.0, 1018.0]    0.569 hPa     0.761 hPa        0.428%      99.572%        0.9989 8,576 (100.0%)
124.0s	171	      max_wind_kt     [0.0, 125.0]     0.557 kt      0.590 kt        0.446%      99.554%        0.9997 8,578 (100.0%)
124.0s	172	         dir_50kt       [0.0, 9.0]        0.076         0.079        0.844%      99.156%        0.9997  5,470 (63.8%)
124.0s	173	 rad_50kt_long_nm     [0.0, 325.0]     2.404 nm      2.517 nm        0.740%      99.260%        0.9974  5,470 (63.8%)
124.0s	174	rad_50kt_short_nm     [0.0, 225.0]     1.894 nm      1.959 nm        0.842%      99.158%        0.9981  5,470 (63.8%)
124.0s	175	         dir_30kt       [1.0, 9.0]        0.047         0.051        0.593%      99.407%        0.9997  5,470 (63.8%)
124.0s	176	 rad_30kt_long_nm    [20.0, 850.0]     1.856 nm      2.309 nm        0.224%      99.776%        0.9995  5,470 (63.8%)
124.0s	177	rad_30kt_short_nm     [0.0, 600.0]     1.926 nm      2.161 nm        0.321%      99.679%        0.9993  5,470 (63.8%)
124.0s	178	             year [1980.0, 2026.0]        0.078         0.103        0.169%      99.831%        0.9999 8,578 (100.0%)
124.0s	179	            month      [1.0, 12.0]        0.018         0.026        0.163%      99.837%        0.9998 8,578 (100.0%)
124.0s	180	              day      [1.0, 31.0]        0.085         0.098        0.284%      99.716%        0.9999 8,578 (100.0%)
124.0s	181	             hour      [0.0, 23.0]        0.126         0.142        0.548%      99.452%        0.9996 8,578 (100.0%)
124.0s	182	
124.0s	183	===============================================================================================
124.0s	184	          GEOSPATIAL POSITION ACCURACY (SCALED BACK TO KILOMETERS)
124.0s	185	===============================================================================================
124.0s	186	 Mean Position Error (Distance)   : 27.68 km
124.0s	187	 Median Position Error (Distance) : 24.94 km
124.0s	188	 90th Percentile Error            : 47.86 km
124.0s	189	 Max Position Error               : 273.49 km
124.0s	190	
124.0s	191	===============================================================================================
124.0s	192	 Grade Classification Exact Match: 100.00% (0 misclassifications out of 8,578 records)
124.0s	193	===============================================================================================
124.0s	194	
124.0s	195	===============================================================================================
124.0s	196	     SAMPLE STORM RECORD INSPECTION (ACTUAL vs RECONSTRUCTED IN REAL PHYSICAL UNITS)
124.0s	197	===============================================================================================
124.0s	198	          Feature Actual (Physical) Reconstructed Difference |Actual - Pred|
124.0s	199	            grade              2.00          2.04                      0.040
124.0s	200	              lat           15.00 °       15.09 °                    0.091 °
124.0s	201	              lon          116.00 °      116.29 °                    0.285 °
124.0s	202	     pressure_hpa       1002.00 hPa   1001.15 hPa                  0.849 hPa
124.0s	203	      max_wind_kt           0.00 kt       0.44 kt                   0.436 kt
124.0s	204	         dir_50kt      MISSING (-1)           N/A                        N/A
124.0s	205	 rad_50kt_long_nm      MISSING (-1)           N/A                        N/A
124.0s	206	rad_50kt_short_nm      MISSING (-1)           N/A                        N/A
124.0s	207	         dir_30kt      MISSING (-1)           N/A                        N/A
124.0s	208	 rad_30kt_long_nm      MISSING (-1)           N/A                        N/A
124.0s	209	rad_30kt_short_nm      MISSING (-1)           N/A                        N/A
124.0s	210	             year           1995.00       1994.94                      0.062
124.0s	211	            month              9.00          8.97                      0.026
124.0s	212	              day             15.00         14.96                      0.037
124.0s	213	             hour              0.00         -0.11                      0.115
124.0s	214	
124.0s	215	===============================================================================================
124.0s	216	             64-D LATENT SPACE VARIANCE ANALYSIS
124.0s	217	===============================================================================================
124.0s	218	Top 3 Principal Components explain : 55.12% of latent variance
124.0s	219	Top 5 Principal Components explain : 73.31% of latent variance
124.0s	220	Top 10 Principal Components explain: 95.55% of latent variance
124.0s	221	===============================================================================================
```
Actually, the 273.49 km of error is kinda scary, that much difference in 24 hours would be not too good, but now i think i must test this one one some kinds of models first.

- 12:20 GMT+7: now i gotta learn how can LNN be used to generate new stuffs, actually, i don't think i want it to autoregressive like gpt transformers, i think that i should just generate 4 couples of numbers to avoid error accumulation.
- 14:31 GMT+7: the result is actualy better than i thought:
```
Horizon     Cosine Sim     Median Distance Error     Mean Distance Error     Lat MAE / Lon MAE
  6h          0.8794             42.01 km                 51.27 km            0.285° / 0.345°
 12h          0.8770             83.25 km                101.35 km            0.557° / 0.693°
 18h          0.8705            127.65 km                155.31 km            0.844° / 1.083°
 24h          0.8632            175.08 km                212.97 km            1.137° / 1.523°
 ``` 
 - 14:41 GMT+7: i realized that it just failed to capture the speed and momentum, why get good result by playing the safe card, predict things so small instead of truly capture the storm momentum, i think must somehow work around this.
 - 15:00 :GMT+7: i'm dead wrong, the model achitecture is wrong, i must redo all the stuffs and plus, i think my autoencoder training is kinda no needed because in this task i need the encoder to be able to capture rich info features, not try to reproduce it.
 - 15:49 GMT+7: for real tho, trusting the agents might fix my problem is super dumb, i guess i gotta work the LNN model with simple Autoencoder by myself, wish me luck, and also i think i might use missing mask, instead of just simply filling stuffs naively.
 - 18:00 GMT+7: i've wrong, the time different are not just simply 6 hours, there are also 3 hours and 0 hours, i guess i gotta do this whole thing all over again, and i have to create some kinds of time feature into that, idk, i must take the time change into accountability.
 - 18:07 GMT+7: because the timestamp is a whole mess, after all i think this will just be a game of liquid neural net with varios other tabular data encoder.
 

 ### AUG,16,2026
 - 10:29 GMT+7: after some eda, i find out that the time stamp that is have a weird diffrent of any value other than 6 is not that bad, i just have to prepare the the data in a better way before training anything.
 - 11:13 GMT+7: the time diff is actually not really the time different between 2 states.
 - 12:02 GMT+: i've just finish the data, and also retrain xgboost, it really got better, i also add max error and min error:
 ```
 --- Evaluation Results ---
Average Cosine Similarity for 6h: 0.9095 | Distance Error: Mean = 41.84 km, Min = 0.50 km, Max = 465.88 km
Average Cosine Similarity for 12h: 0.8969 | Distance Error: Mean = 86.54 km, Min = 0.60 km, Max = 635.74 km
Average Cosine Similarity for 18h: 0.8830 | Distance Error: Mean = 138.65 km, Min = 2.37 km, Max = 883.57 km
Average Cosine Similarity for 24h: 0.8703 | Distance Error: Mean = 194.87 km, Min = 0.96 km, Max = 1223.63 km
 ```

 - 12:03 GMT+7: now i will continue working with the LNN, i will try to find out some super great decoder that truly can capture the physical essence of my super cool model, haha, i think i will start with tabnet, cuz, why not?