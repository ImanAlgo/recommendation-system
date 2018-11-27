package iman.research;

import java.util.List;

import com.google.common.collect.BiMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.model.TextDataModel;
import net.librec.math.structure.SparseMatrix;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.rating.SVDPlusPlusRecommender;
import net.librec.recommender.item.RecommendedItem;
import net.librec.similarity.PCCSimilarity;
import net.librec.similarity.RecommenderSimilarity;


/**
 * Hello world!
 *
 */
public class App 
{
    /** Log */
    private static final Log LOG = LogFactory.getLog(App.class);

    Configuration conf = null;
    TextDataModel MotherDataModel = null;

    public static void main( String[] args ) throws LibrecException
    {

        App app = new App();
        app.setupData();
        SparseMatrix imputedTrainSet = app.impute();

        recommendByFFM();

    }

    private static void recommendByFFM() {

        // rec.recommender.maxrate=12.0
        // rec.recommender.minrate=0.0

        // rec.factor.number=10

        // rec.fm.regw0=0.01
        // reg.fm.regW=0.01
        // reg.fm.regF=10

        Recommender recommender = new FFMRecommender();

    }

    void setupData() throws LibrecException {
        conf = new Configuration();

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

        conf.set("rec.random.seed", "1");
        conf.set("data.convert.binarize.threshold", "-1.0");

        MotherDataModel = new TextDataModel(conf);
        MotherDataModel.buildDataModel();
    }

    private SparseMatrix impute() throws LibrecException {

        LOG.info("*** Imputing Phase ...");

        Configuration dmConf = new Configuration();
        dmConf.set("data.model.splitter", "ratio");
        dmConf.set("data.splitter.ratio", "rating");
        dmConf.set("data.splitter.trainset.ratio", "0.9");
        
        SparseMatrix sparseTrainData = ((SparseMatrix)MotherDataModel.getTrainDataSet()).clone();
        MatrixDataModel dataModel = new MatrixDataModel(dmConf, sparseTrainData, MotherDataModel.getUserMappingData(), MotherDataModel.getItemMappingData());
        dataModel.buildDataModel();

        Configuration svdConf = new Configuration();
        svdConf.set("rec.recommender.similarity.key", "user");
        svdConf.set("rec.recommender.class", "svdpp");
        svdConf.set("rec.iterator.learnrate", "0.01");
        svdConf.set("rec.iterator.learnrate.maximum", "0.01");
        svdConf.set("rec.iterator.maximum", "13");
        svdConf.set("rec.user.regularization", "0.01");
        svdConf.set("rec.item.regularization", "0.01");
        svdConf.set("rec.impItem.regularization", "0.001");
        svdConf.set("rec.factor.number", "10");
        svdConf.set("rec.learnrate.bolddriver", "false");
        svdConf.set("rec.learnrate.decay", "1.0");
        
        // build recommender context
        
        RecommenderContext context = new RecommenderContext(svdConf, dataModel);
        // build similarity
        RecommenderSimilarity similarity = new PCCSimilarity();
        similarity.buildSimilarityMatrix(dataModel);
        context.setSimilarity(similarity);
        // build recommender
        Recommender recommender = new SVDPlusPlusRecommender();
        recommender.setContext(context);
        // run recommender algorithm
        recommender.recommend(context);

        // Merge predicted test data with the train data for imputing 
        SparseMatrix newTrainData = merge(dataModel.getDataSplitter().getTrainData(), // == (SparseMatrix) dataModel.getTrainDataSet()
        recommender.getRecommendedList(),
        recommender.getDataModel().getUserMappingData(),
        recommender.getDataModel().getItemMappingData());

        return newTrainData;
    }

    static private SparseMatrix merge(SparseMatrix data, List<RecommendedItem> recommendedList, BiMap<String, Integer> userMappingDate, BiMap<String, Integer> itemMappingData){

        for(RecommendedItem rItem : recommendedList){
            int row = userMappingDate.get(rItem.getUserId());
            int column = itemMappingData.get(rItem.getItemId());
            double value = rItem.getValue();
            data.set(row, column, value);
        }

        SparseMatrix.reshape(data);

        return data;
    }
}
