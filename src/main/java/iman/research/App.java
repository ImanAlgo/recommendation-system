package iman.research;

import java.util.List;

import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.conf.Configuration.Resource;
import net.librec.data.DataModel;
import net.librec.data.model.TextDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.rating.MAEEvaluator;
import net.librec.filter.GenericRecommendedFilter;
import net.librec.filter.RecommendedFilter;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.UserKNNRecommender;
import net.librec.similarity.PCCSimilarity;
import net.librec.similarity.RecommenderSimilarity;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws LibrecException
    {
        System.out.println( "Hello World!" );


        /*
# set data directory
dfs.data.dir=../data
# set result directory
# recommender result will output in this folder
dfs.result.dir=../result

# convertor
# load data and splitting data
# into two (or three) set
# setting dataset name
data.input.path=filmtrust
# setting dataset format(UIR, UIRT)
data.column.format=UIR
# setting method of split data
# value can be ratio, loocv, given, KCV
data.model.splitter=ratio
#data.splitter.cv.number=5
# using rating to split dataset
data.splitter.ratio=rating
# filmtrust dataset is saved by text
# text, arff is accepted
data.model.format=text
# the ratio of trainset
# this value should in (0,1)
data.splitter.trainset.ratio=0.8

# Detailed configuration of loocv, given, KCV
# is written in User Guide

# set the random seed for reproducing the results (split data, init parameters and other methods using random)
# default is set 1l
# if do not set ,just use System.currentTimeMillis() as the seed and could not reproduce the results.
rec.random.seed=1

# binarize threshold mainly used in ranking
# -1.0 - maxRate, binarize rate into -1.0 and 1.0
# binThold = -1.0， do nothing
# binThold = value, rating > value is changed to 1.0 other is 0.0, mainly used in ranking
# for PGM 0.0 maybe a better choose
data.convert.binarize.threshold=-1.0

# evaluation the result or not
rec.eval.enable=true

# specifies evaluators
# rec.eval.classes=auc,precision,recall...
# if rec.eval.class is blank
# every evaluator will be calculated
# rec.eval.classes=auc,precision,recall

# evaluator value set is written in User Guide
# if this algorithm is ranking only true or false
rec.recommender.isranking=false

#can use user,item,social similarity, default value is user, maximum values:user,item,social
#rec.recommender.similarities=user
        */
  	// recommender configuration
    Configuration conf = new Configuration();
    conf.set("dfs.data.dir", "ml-1m");
    conf.set("data.input.path", "ratings.dat");
	// Resource resource = new Resource("rec/cf/userknn-test.properties");
	// conf.addResource(resource);

	// build data model
	DataModel dataModel = new TextDataModel(conf);
	dataModel.buildDataModel();
	
	// set recommendation context
	RecommenderContext context = new RecommenderContext(conf, dataModel);
	RecommenderSimilarity similarity = new PCCSimilarity();
	similarity.buildSimilarityMatrix(dataModel);
	context.setSimilarity(similarity);

	// training
	Recommender recommender = new UserKNNRecommender();
	recommender.recommend(context);

	// evaluation
	RecommenderEvaluator evaluator = new MAEEvaluator();
	recommender.evaluate(evaluator);

	// recommendation results
	List recommendedItemList = recommender.getRecommendedList();
	RecommendedFilter filter = new GenericRecommendedFilter();
    recommendedItemList = filter.filter(recommendedItemList);
    
    }
}
