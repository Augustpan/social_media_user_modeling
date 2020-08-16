# Source code is in "src" folder!

# Important Note:

1. All of these scripts are original and written all by mtself.
2. Several open source packages/libraries are used in my program:
	- jieba: 	a python packages for Chinese words spliting.
	- gensim: 	a python packages for topic modeling task of NLP.
	- libsvm:	a library for support vector machines.
3. Currently, interfaces for application of our models haven't been implemented. The only function of these scripts  is to validate our models, which means that the outputs would be prediction accuracy of our models in particular task rather than prediction results. 
4. These scripts have not been optimized at all, loading data and training may takes several hours.
5. I have some precomputed data to boost the speed, but they are too large to mail:)
6. The test data provided in "src/data" is a subset of NLPCC provided dataset, our program may not work well on it (There's problem with training set and validation set spliting if the dataset is small). So, please get original NLPCC provided dataset to test this program.

---

# Files:

1. build.py: 		a helper class for spliting original dataset into training set and validation set
2. nlpa_wrap.py: 	a wrap for sevral basic natural language processing algorithms from "jieba" and "gensim"
3. nlpcc_parser.py: a helper class for storing and easy accessing for original data
4. util.py: 		contains some utility functions
5. tag_svm.py: 		generate libsvm style data
6. recommend.py: 	item-based recommend system for POIs prediction
7. theme_bayes.py: 	preidct user's gender with tweet topic, by Bayes Netowrk method
8. tag_bayes.py: 	preidct user's gender with tag, by Bayes Netowork method
9. hybrid_bayes.py: preidct user's gender with tag and topic, by Bayes Netowork method

---

# Usage:
 
1. Put your checkins.txt, profile.txt, social.txt, tags.txt, tweets.txt and stopwords.txt in "PROJECT_ROOT/data". Please refer to the given samples of those file for the format.
2. To perform user's gender prediction, refer to "Task Two"
3. To perform user's POIs prediction, refer to "Task One"

## Task One:
- run "python recommend.py" to predict users' faverable POIs with item-based receommend system

## Task Two:
- put your data in a folder named "data" and create a new one named "save"; their parent folder must contains python scripts of this program
1. By Bayes Method
	- run "python tag_bayes.py" to make prediction of uers' gender on tag data only (by Bayes Net)
	- run "python theme_bayes.py" to make prediction of uers' gender on tweet data only (by Bayes Net)
	- run "python hybrid_bayes.py" to make prediction of uers' gender on both data (by Bayes Net)
2. By SVM Method
	- run "python tag_svm.py" to build libsvm style training and validation data, these data will be generated in "src/save"
	- run libsvm to to make prediction of uers' gender by SVM
