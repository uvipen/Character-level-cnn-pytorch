Sogou News Topic Classification Dataset

Version 3, Updated 09/09/2015


ORIGIN

The SogouCA and SogouCS Chinese news dataset is collected by SogouLab from various news website. The use of the original datasets is subject to License for Use of Sogou Lab Data. For more information, please refer to the links http://www.sogou.com/labs/dl/ca.html and http://www.sogou.com/labs/dl/cs.html .

The Sogou news topic classification dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from a combination of SogouCA and SogouCS. It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).


DESCRIPTION

The Sogou news topic classification dataset is constructed by manually labeling each news article according to its URL, which represents roughly the categorization of news in their websites. We chose 5 largest categories for the dataset, each having 90,000 samples for training and 12,000 for testing. The Pinyin texts are converted using pypinyin combined with jieba Chinese segmentation system. In total there are 450,000 training samples and 60,000 testing samples.

The file classes.txt contains a list of classes corresponding to each label.

The files train.csv and test.csv contain all the training samples as comma-sparated values. There are 3 columns in them, corresponding to class index (1 to 5), title and content. The title and content are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed with an "n" character, that is "\n".
