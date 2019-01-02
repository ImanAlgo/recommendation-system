package iman.research;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

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
    Configuration cfg = new Configuration();

    public App(String configFile) {
        init(configFile);
    }

    public static void main(String[] args) throws LibrecException {

        App app = new App(args==null||args.length==0?null:args[0]);

        DataModel sparsDataModel = app.setupData();

        DataModel imputedDataModel = app.impute(sparsDataModel);

        LOG.info("== Execution of MFALSRecommender base on sparse train data(is not imputed)");
        Recommender recBySpars = app.recommendByFFM(sparsDataModel);
        LOG.info("== Execution of MFALSRecommender base on imputed train data");
        Recommender recByImputed = app.recommendByFFM(imputedDataModel);

        LOG.info("== Comparing the result of recommendations base on being imputed and not imputed train data");
        app.evaluate(recBySpars, recByImputed);

        System.out.println("THE END!");

    }

    private void init(String configFile) {

        Path path = Paths.get("main.properties");
        Configuration.Resource resource = new Configuration.Resource(path);
        cfg.addResource(resource);
    }

    public DataModel setupData() throws LibrecException {
        DataModel sparsDataModel = new TextDataModel(cfg);
        sparsDataModel.buildDataModel();

        return sparsDataModel;
    }

    public MatrixDataModel impute(DataModel sparsDataModel) throws LibrecException {

        LOG.info("================== IMPUTING PHASE ==================");

        SparseMatrix sparseTrainData = (SparseMatrix) sparsDataModel.getTrainDataSet();

        Map<Integer, Map<Integer, CombinedRecommendersCell> > combinedRecMap = new HashMap<>();
        TreeSet<CombinedRecommendersCell> orderedCombinedRecByVariance = new TreeSet<>();

        final String[] SPLITE_BY = cfg.getStrings("impute.splitter.ratio"); // rating,user,userfixed,item
        final int NUMBER_OF_RATIO_TYPES = SPLITE_BY.length;
        final int IMPUTE_ITERATION_NUMBER = cfg.getInt("impute.rounds", 10);

        LOG.info("== Starting ("+NUMBER_OF_RATIO_TYPES * IMPUTE_ITERATION_NUMBER+") random rounds");
        for (int i = 0; i < NUMBER_OF_RATIO_TYPES * IMPUTE_ITERATION_NUMBER; ++i) {
            
            Configuration dmConf = new Configuration();
            dmConf.set("data.model.splitter", "ratio");
            dmConf.set("data.splitter.ratio", SPLITE_BY[i%NUMBER_OF_RATIO_TYPES]);
            dmConf.set("data.splitter.trainset.ratio", "1");

            LOG.info("=== New round("+(i%NUMBER_OF_RATIO_TYPES)+") with splitting data based on " + dmConf.get("data.splitter.ratio"));

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

            LOG.info("------ Create recommender context");
            // build recommender context
            RecommenderContext context = new RecommenderContext(svdConf, dataModel);
            LOG.info("------ Build PCC similarity");
            // build similarity
            RecommenderSimilarity similarity = new PCCSimilarity();
            similarity.buildSimilarityMatrix(dataModel);
            context.setSimilarity(similarity);
            // build recommender
            Recommender recommender = new SVDPlusPlusRecommender();
            recommender.setContext(context);
            LOG.info("------ Execute SVD++ recommender");
            // run recommender algorithm
            recommender.recommend(context);

            LOG.info("------ Accumulate recommendations");
            List<RecommendedItem> recommendedList = recommender.getRecommendedList();
            
            for (RecommendedItem rItem : recommendedList) {

                int userIndex = sparsDataModel.getUserMappingData().get(rItem.getUserId());
                int itemIndex = sparsDataModel.getItemMappingData().get(rItem.getItemId());
                
                if (!sparseTrainData.contains(userIndex, itemIndex)) {
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

        LOG.info("== End of imputing random rounds");

        float imputeRatio = cfg.getFloat("impute.ratio", 0.1f);
        final float IMPUTE_RATIO = imputeRatio > 1 ? 1 : imputeRatio < 0 ? 0 : imputeRatio;
        LOG.info("== Imputing "+IMPUTE_RATIO*100+"% of train data plus its own real present data");
        SparseMatrix imputedTrainMatrix = new SparseMatrix(sparseTrainData);

        Iterator<CombinedRecommendersCell> itr = orderedCombinedRecByVariance.iterator(); 
        for (int counter = 0; itr.hasNext() && counter < (imputedTrainMatrix.size()*IMPUTE_RATIO); ++counter) {
            CombinedRecommendersCell cell = itr.next();
            imputedTrainMatrix.set(cell.getUserIndex(), cell.getItemIndex(), cell.getMean());
        }
        SparseMatrix.reshape(imputedTrainMatrix);

            // Merge predicted test data with the train data for imputing
            // SparseMatrix newTrainData = merge(sparseTrainData, // == (SparseMatrix) dataModel.getTrainDataSet()
            //         recommender.getRecommendedList(), recommender.getDataModel().getUserMappingData(),
            //         recommender.getDataModel().getItemMappingData());

        LOG.info("=== Creating the new data model base on the imputed train matrix");

        // Create imputed data model
        MatrixDataModel imputedDataModel = new MatrixDataModel(imputedTrainMatrix,
                (SparseMatrix) sparsDataModel.getTestDataSet(), sparsDataModel.getUserMappingData(),
                sparsDataModel.getItemMappingData());
        imputedDataModel.buildDataModel();

        LOG.info("================== IMPUTING PHASE DONE! ==================");

        return imputedDataModel;
    }

    public Recommender recommendByFFM(DataModel dataModel) throws LibrecException {

        Configuration svdConf = new Configuration();
        svdConf.set("rec.recommender.similarity.key", "user");
        svdConf.set("rec.recommender.class", "svdpp");
        svdConf.set("rec.iterator.learnrate", "0.01");
        svdConf.set("rec.iterator.learnrate.maximum", "0.01");
        svdConf.set("rec.iterator.maximum", "1");
        svdConf.set("rec.user.regularization", "0.01");
        svdConf.set("rec.item.regularization", "0.01");
        svdConf.set("rec.impItem.regularization", "0.001");
        svdConf.set("rec.factor.number", "20");
        svdConf.set("rec.learnrate.bolddriver", "false");
        svdConf.set("rec.learnrate.decay", "1.0");

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
        //Recommender recommender = new SVDPlusPlusRecommender();
        recommender.setContext(context);
        // run recommender algorithm
        recommender.recommend(context);

        return recommender;
    }

    public void evaluate(Recommender recBySpars, Recommender recByImputed) throws LibrecException {

        RecommenderEvaluator evaluator1 = new RMSEEvaluator();
        double sparsRmse = recBySpars.evaluate(evaluator1);

        RecommenderEvaluator evaluator2 = new RMSEEvaluator();
        double imputedRmse = recByImputed.evaluate(evaluator2);

        LOG.info("== RMSE obtained by sparse train set:  " + sparsRmse);
        LOG.info("== RMSE obtained by imputed train set: " + imputedRmse);
    }
}
