package iman.research;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.DataModel;
import net.librec.data.model.TextDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.rating.RMSEEvaluator;
import net.librec.math.structure.SparseMatrix;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.rating.MFALSRecommender;
import net.librec.recommender.cf.rating.SVDPlusPlusRecommender;
import net.librec.recommender.item.RecommendedItem;
import net.librec.similarity.PCCSimilarity;
import net.librec.similarity.RecommenderSimilarity;

/**
 * Hello world!
 *
 */
public class App {
    /** Log */
    private static final Log LOG = LogFactory.getLog(App.class);
    private final int IMPUTE_ITERATION_NUMBER = 1;

    public static void main(String[] args) throws LibrecException {

        App app = new App();

        DataModel sparsDataModel = app.setupData();

        DataModel imputedDataModel = app.impute(sparsDataModel);

        Recommender recBySpars = app.recommendByFFM(sparsDataModel);
        Recommender recByImputed = app.recommendByFFM(imputedDataModel);

        app.evaluate(recBySpars, recByImputed);

        System.out.println("END!");

    }

    public Recommender recommendByFFM(DataModel dataModel) throws LibrecException {

        Configuration fmalsConf = new Configuration();
        fmalsConf.set("rec.recommender.similarity.key", "user");
        fmalsConf.set("rec.recommender.class", "fmals");

        fmalsConf.set("rec.iterator.learnRate", "0.001");
        fmalsConf.set("rec.iterator.maximum", "100");

        fmalsConf.set("rec.recommender.maxrate", "12.0");
        fmalsConf.set("rec.recommender.minrate", "0.0");

        fmalsConf.set("rec.factor.number", "10");

        fmalsConf.set("rec.fm.regw0", "0.01");
        fmalsConf.set("reg.fm.regW", "0.01");
        fmalsConf.set("reg.fm.regF", "10");

        // build recommender context

        RecommenderContext context = new RecommenderContext(fmalsConf, dataModel);
        // build similarity
        RecommenderSimilarity similarity = new PCCSimilarity();
        similarity.buildSimilarityMatrix(dataModel);
        context.setSimilarity(similarity);
        // build recommender
        Recommender recommender = new MFALSRecommender();
        recommender.setContext(context);
        // run recommender algorithm
        recommender.recommend(context);

        return recommender;
    }

    DataModel setupData() throws LibrecException {
        Configuration conf = new Configuration();

        // set data directory
        conf.set("dfs.data.dir", "data/movielense");
        // set result directory
        // recommender result will output in this folder
        conf.set("dfs.result.dir", "result");

        // convertor
        // load data and splitting data
        // into two (or three) set

        // setting dataset name ml-1m, ml-100k
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

        // DataModel sparsDataModel = new ArffDataModel(conf);
        DataModel sparsDataModel = new TextDataModel(conf);
        sparsDataModel.buildDataModel();

        return sparsDataModel;
    }

    public MatrixDataModel impute(DataModel sparsDataModel) throws LibrecException {

        LOG.info("================== IMPUTING PHASE ==================");

        List<String> spliterRatioBy = Arrays.asList("rating" ,"user", "userfixed", "item");
        List<Recommender> trainedRecommenders = new ArrayList<Recommender>();

        for (int i = 0; i < 4*this.IMPUTE_ITERATION_NUMBER; ++i) {
            
            Configuration dmConf = new Configuration();
            dmConf.set("data.model.splitter", "ratio");
            dmConf.set("data.splitter.ratio", spliterRatioBy.get(i%4));
            dmConf.set("data.splitter.trainset.ratio", "0.9");

            LOG.info("=== New round with split data based on " + dmConf.get("data.splitter.ratio"));

            SparseMatrix sparseTrainData = new SparseMatrix((SparseMatrix) sparsDataModel.getTrainDataSet());
            MatrixDataModel dataModel = new MatrixDataModel(dmConf, sparseTrainData,
                    sparsDataModel.getUserMappingData(), sparsDataModel.getItemMappingData());
            dataModel.buildDataModel();

            Configuration svdConf = new Configuration();
            svdConf.set("rec.recommender.similarity.key", "user");
            svdConf.set("rec.recommender.class", "svdpp");
            svdConf.set("rec.iterator.learnrate", "0.01");
            svdConf.set("rec.iterator.learnrate.maximum", "0.01");
            svdConf.set("rec.iterator.maximum", "1");
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

            trainedRecommenders.add(recommender);
        }

        SparseMatrix imputedTrainMatrix = new SparseMatrix((SparseMatrix) sparsDataModel.getTrainDataSet());

        Map<Integer, Map<Integer, CombinedRecommendersCell> > combinedRecMap = new HashMap<>();
        TreeSet<CombinedRecommendersCell> orderedCombinedRecByVariance = new TreeSet<>();

        for(Recommender rec : trainedRecommenders){
            List<RecommendedItem> recommendedList = rec.getRecommendedList();
            
            for (RecommendedItem rItem : recommendedList) {
                int userIndex = sparsDataModel.getUserMappingData().get(rItem.getUserId());
                int itemIndex = sparsDataModel.getItemMappingData().get(rItem.getItemId());
                
                if (!imputedTrainMatrix.contains(userIndex, itemIndex)){
                    CombinedRecommendersCell cell = new CombinedRecommendersCell(userIndex, itemIndex);
                    combinedRecMap.putIfAbsent(userIndex, new HashMap<>());
                    combinedRecMap.get(userIndex).putIfAbsent(itemIndex, cell);
                    cell = combinedRecMap.get(userIndex).get(itemIndex);

                    double rate = rItem.getValue();
                    cell.addRate(rate);

                    orderedCombinedRecByVariance.add(cell);
                }
            }
        }

        Iterator<CombinedRecommendersCell> itr = orderedCombinedRecByVariance.iterator(); 
        for (int counter = 0; itr.hasNext() && counter < (imputedTrainMatrix.size()*0.1); ++counter) { 
            CombinedRecommendersCell cell = itr.next();
            imputedTrainMatrix.set(cell.getUserIndex(), cell.getItemIndex(), cell.getMean());
        }
        SparseMatrix.reshape(imputedTrainMatrix);

            // Merge predicted test data with the train data for imputing
            // SparseMatrix newTrainData = merge(sparseTrainData, // == (SparseMatrix) dataModel.getTrainDataSet()
            //         recommender.getRecommendedList(), recommender.getDataModel().getUserMappingData(),
            //         recommender.getDataModel().getItemMappingData());

        // imputed data model
        MatrixDataModel imputedDataModel = new MatrixDataModel(imputedTrainMatrix,
                (SparseMatrix) sparsDataModel.getTestDataSet(), sparsDataModel.getUserMappingData(),
                sparsDataModel.getItemMappingData());
        imputedDataModel.buildDataModel();

        return imputedDataModel;
    }

    // static private SparseMatrix merge(SparseMatrix data, List<RecommendedItem> recommendedList,
    //         BiMap<String, Integer> userMappingDate, BiMap<String, Integer> itemMappingData) {

    //     for (RecommendedItem rItem : recommendedList) {
    //         int row = userMappingDate.get(rItem.getUserId());
    //         int column = itemMappingData.get(rItem.getItemId());

    //         // int row = Integer.parseInt(rItem.getUserId());
    //         // int column = Integer.parseInt(rItem.getItemId());

    //         double value = rItem.getValue();
    //         // data.set(row, column, value);
    //         if (!data.contains(row, column))
    //             data.set(row, column, value);
    //     }

    //     SparseMatrix.reshape(data);

    //     return data;
    // }

    public void evaluate(Recommender recBySpars, Recommender recByImputed) throws LibrecException {

        RecommenderEvaluator evaluator1 = new RMSEEvaluator();
        double sparsRmse = recBySpars.evaluate(evaluator1);

        RecommenderEvaluator evaluator2 = new RMSEEvaluator();
        double imputedRmse = recByImputed.evaluate(evaluator2);

        LOG.info("RMSE obtained by sparse train set:  " + sparsRmse);
        LOG.info("RMSE obtained by imputed train set: " + imputedRmse);
    }
}
