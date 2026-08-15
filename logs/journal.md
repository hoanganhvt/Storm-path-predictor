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
======================================================================
         PER-FEATURE RECONSTRUCTION MATRIX & ERROR METRICS
======================================================================
          Feature Physical MAE Physical RMSE R² Score  Valid Records
            grade        0.015         0.018   0.9998 8,578 (100.0%)
              lat      0.086 °       0.127 °   0.9999 8,578 (100.0%)
              lon      0.094 °       0.177 °   0.9999 8,578 (100.0%)
     pressure_hpa    0.501 hPa     0.533 hPa   0.9995 8,576 (100.0%)
      max_wind_kt     0.145 kt      0.195 kt   1.0000 8,578 (100.0%)
         dir_50kt        0.025         0.032   0.9999  5,470 (63.8%)
 rad_50kt_long_nm     1.117 kt      1.225 kt   0.9994  5,470 (63.8%)
rad_50kt_short_nm     0.531 kt      0.639 kt   0.9998  5,470 (63.8%)
         dir_30kt        0.053         0.055   0.9997  5,470 (63.8%)
 rad_30kt_long_nm     2.803 kt      3.037 kt   0.9991  5,470 (63.8%)
rad_30kt_short_nm     1.117 kt      1.409 kt   0.9997  5,470 (63.8%)
             year        0.132         0.167   0.9998 8,578 (100.0%)
            month        0.024         0.033   0.9998 8,578 (100.0%)
              day        0.103         0.116   0.9998 8,578 (100.0%)
             hour        0.107         0.118   0.9997 8,578 (100.0%)

======================================================================
 Grade Classification Exact Match Accuracy: 100.00%
======================================================================

======================================================================
             64-D LATENT SPACE VARIANCE ANALYSIS
======================================================================
Top 3 Principal Components explain : 55.64% of latent variance
Top 5 Principal Components explain : 73.77% of latent variance
Top 10 Principal Components explain: 95.12% of latent variance
======================================================================
Test evaluation finished successfully!
```