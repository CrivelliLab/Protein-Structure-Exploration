10:58 AM 27 JULY 2017

Training on 4 cards on 40K subset of KRAS/HRAS training split (80/20) killed at
145 epochs. The categorical accuracy is consistently at 98%, while validation
accuracy has been oscillating between the high 70s and just below 93%. We
suspect this is due to the set simply being too small and the validation split
containing examples completely foreign to the trained net. More research is
required to truly determine the cause, however. This run is being replaced by a
similar run with the full dataset, and perhaps with an extra dense layer as
well. 
