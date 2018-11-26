package iman.research;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Collections;
import java.util.List;

import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.conf.Configured;
import net.librec.conf.Configuration.Resource;
import net.librec.data.DataModel;
import net.librec.data.convertor.TextDataConvertor;
import net.librec.data.model.TextDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.rating.MAEEvaluator;
import net.librec.eval.rating.RMSEEvaluator;
import net.librec.filter.GenericRecommendedFilter;
import net.librec.filter.RecommendedFilter;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.UserKNNRecommender;
import net.librec.recommender.cf.rating.SVDPlusPlusRecommender;
import net.librec.recommender.item.RecommendedItem;
import net.librec.recommender.item.RecommendedItemList;
import net.librec.recommender.item.RecommendedList;
import net.librec.similarity.CosineSimilarity;
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
    
        // try {
        //     String current = new File(".").getCanonicalPath();
        //     System.out.println("Current dir:"+current);
        // } catch (IOException e1) {
        //     // TODO Auto-generated catch block
        //     e1.printStackTrace();
        // }
        

  	// recommender configuration
    Configuration conf = new Configuration();
    // set data directory
    conf.set("dfs.data.dir", "data/movielense");
    // set result directory
    // recommender result will output in this folder
    conf.set("dfs.result.dir", "result");

    // convertor
    // load data and splitting data
    // into two (or three) set

    // setting dataset name
    conf.set("data.input.path", "ml-1m");
    // setting dataset format(UIR, UIRT)
    conf.set("data.column.format", "UIRT");

    // # movielense dataset is saved by text
    // # text, arff is accepted
    conf.set("data.model.format", "text");

    // setting method of split data
    // value can be ratio, loocv, given, KCV
    conf.set("data.model.splitter", "ratio");
    // #data.splitter.cv.number=5
    // # using rating to split dataset
    conf.set("data.splitter.ratio", "rating");
    // # the ratio of trainset
     // # this value should in (0,1)
    conf.set("data.splitter.trainset.ratio", "0.8");

    // TextDataConvertor convertor = new TextDataConvertor(conf.get(Configured.CONF_DATA_COLUMN_FORMAT), )
    // convertor.processData();
    // RatioDataSplitter splitter = new RatioDataSplitter(convertor, conf);
    // splitter.splitData();


    


	// Resource resource = new Resource("rec/cf/userknn-test.properties");
	// conf.addResource(resource);

	// build data model
	DataModel dataModel = new TextDataModel(conf);
	dataModel.buildDataModel();

    // set Similarity
    conf.set("rec.recommender.similarities","user");
    RecommenderSimilarity similarity = new PCCSimilarity();
    similarity.buildSimilarityMatrix(dataModel);
    
	// set recommendation context
	RecommenderContext context = new RecommenderContext(conf, dataModel, similarity);
	// RecommenderSimilarity similarity = new PCCSimilarity();
	// similarity.buildSimilarityMatrix(dataModel);
	// context.setSimilarity(similarity);

	// training

    // rec.recommender.class=svdpp
    // rec.iterator.learnrate=0.002
    // rec.iterator.learnrate.maximum=0.05
    // rec.iterator.maximum=100
    // rec.user.regularization=0.01
    // rec.item.regularization=0.01
    // rec.impItem.regularization=0.01
    // rec.bias.regularization=0.01
    // rec.factor.number=20
    // rec.learnrate.bolddriver=false
    // rec.learnrate.decay=1.0

    conf.set("rec.iterator.maximum", "2");

    Recommender recommender = new SVDPlusPlusRecommender();
    recommender.recommend(context);

    // evaluate the recommended result
    RecommenderEvaluator evaluator = new RMSEEvaluator();
    double rmse = recommender.evaluate(evaluator);
    List<RecommendedItem> output = recommender.getRecommendedList();

    recommender.saveModel("recomended-result");
    

	// recommendation results
	List recommendedItemList = recommender.getRecommendedList();
	RecommendedFilter filter = new GenericRecommendedFilter();
    recommendedItemList = filter.filter(recommendedItemList);

    }
}
