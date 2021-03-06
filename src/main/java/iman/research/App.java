package iman.research;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.Normalizer;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;
import net.librec.eval.rating.MAEEvaluator;
import net.librec.eval.rating.MPEEvaluator;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.MatrixEntry;
import net.librec.recommender.cf.ItemKNNRecommender;
import net.librec.recommender.cf.UserKNNRecommender;
import net.librec.recommender.cf.rating.NMFRecommender;
import net.librec.recommender.cf.rating.PMFRecommender;
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
 */
public class App {
    /**
     * Log
     */
    private static final Log LOG = LogFactory.getLog(App.class);
    Configuration cfg = new Configuration();
    double lastVariance = -1;
    List<Double> differentVariance = new ArrayList<>();
    long actualNumberOfImputedCells = 0;
    long maximumNumberOfToBeImputedCells = 0;
    long numberOfWholeCells = 0;

    private Instant start;

    public App(String configFile) {
        init(configFile);
    }

    public static void main(String[] args) throws LibrecException, IOException {
        App app = new App(args == null || args.length == 0 ? null : args[0]);

        app.start = Instant.now();

        DataModel sparsDataModel = app.setupData();

        DataModel imputedDataModel = app.impute(sparsDataModel);

        LOG.info("== Execution of MFALSRecommender base on sparse train data(is not imputed)");
        Recommender recBySpars = app.recommendByFFM(sparsDataModel);
        LOG.info("== Execution of MFALSRecommender base on imputed train data");
        Recommender recByImputed = app.recommendByFFM(imputedDataModel);

        LOG.info("== Comparing the result of recommendations base on being imputed and not imputed train data");
        EvaluationResult result = app.evaluate(recBySpars, recByImputed);
        app.printResult(result);

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

        SparseMatrix sparseTrainMatrix = (SparseMatrix) sparsDataModel.getTrainDataSet();

        Map<Integer, Map<Integer, CombinedRecommendersCell>> combinedRecMap = new HashMap<>();
        List<CombinedRecommendersCell> orderedCombinedRecByVariance = new ArrayList<>(sparseTrainMatrix.numRows()*sparseTrainMatrix.numColumns());

        final String SPLITE_RATIO = cfg.get("impute.splitter.trainset.ratio", "0.5");
        final String[] SPLITE_BY = cfg.getStrings("impute.splitter.ratio"); // rating,user,userfixed,item
        final int NUMBER_OF_RATIO_TYPES = SPLITE_BY.length;
        final int IMPUTE_ITERATION_NUMBER = cfg.getInt("impute.rounds", 10);

        LOG.info("== Starting (" + NUMBER_OF_RATIO_TYPES * IMPUTE_ITERATION_NUMBER + ") random rounds");
        for (int i = 0; i < NUMBER_OF_RATIO_TYPES * IMPUTE_ITERATION_NUMBER; ++i) {

            Configuration dmConf = new Configuration();
            dmConf.set("data.model.splitter", "ratio");
            dmConf.set("data.splitter.ratio", SPLITE_BY[i % NUMBER_OF_RATIO_TYPES]);
            dmConf.set("data.splitter.trainset.ratio", SPLITE_RATIO);
            dmConf.set("rec.random.seed", String.valueOf(System.currentTimeMillis()));

            LOG.info("=== New round " + (i) + " out of " + NUMBER_OF_RATIO_TYPES * IMPUTE_ITERATION_NUMBER + " rounds with splitting data based on " + dmConf.get("data.splitter.ratio"));

            MatrixDataModel imputingDataModel = new MatrixDataModel(dmConf, sparseTrainMatrix,
                    sparsDataModel.getUserMappingData(), sparsDataModel.getItemMappingData());
            imputingDataModel.buildDataModel();

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
            RecommenderContext context = new RecommenderContext(svdConf, imputingDataModel);
            LOG.info("------ Build PCC similarity");
            // build similarity
            RecommenderSimilarity similarity = new PCCSimilarity();
            similarity.buildSimilarityMatrix(imputingDataModel);
            context.setSimilarity(similarity);
            // build recommender
            CustomizableRecommender recommender = (CustomizableRecommender) CustomizableRecommender.create(new SVDPlusPlusRecommender());
            recommender.setContext(context);
            LOG.info("------ Train SVD++ recommender");
            // run recommender algorithm
            recommender.recommend(context);

            LOG.info("------ Accumulate sparse values with predicted recommendations");

            SparseMatrix recommenderTrainMatrix = (SparseMatrix) imputingDataModel.getTrainDataSet();
            SparseMatrix recommenderTestMatrix = (SparseMatrix) imputingDataModel.getTestDataSet();
            int counter = 0;
            double predictRating;

            for (int rowIdx = 0; rowIdx < recommenderTrainMatrix.numRows(); ++rowIdx) {
                for (int ColumnIdx = 0; ColumnIdx < recommenderTrainMatrix.numColumns(); ++ColumnIdx) {

                    int userIndex = recommenderTrainMatrix.rowInd[rowIdx];
                    int itemIndex = recommenderTrainMatrix.colInd[ColumnIdx];

                    if (recommenderTrainMatrix.contains(userIndex, itemIndex)
                            || recommenderTestMatrix.contains(userIndex, itemIndex)) {
                        continue;
                    }

                    predictRating = recommender.predict(userIndex, itemIndex);

                    CombinedRecommendersCell cell = new CombinedRecommendersCell(userIndex, itemIndex);
                    combinedRecMap.putIfAbsent(userIndex, new HashMap<>());
                    combinedRecMap.get(userIndex).putIfAbsent(itemIndex, cell);
                    cell = combinedRecMap.get(userIndex).get(itemIndex);
                    cell.addRate(predictRating);
                    orderedCombinedRecByVariance.add(cell);

                    ++counter;
                }
            }
        }

        orderedCombinedRecByVariance.sort(CombinedRecommendersCell::compareTo);

        LOG.info("== End of imputing random rounds");

        float imputeRatio = cfg.getFloat("impute.ratio", 0.1f);
        final float IMPUTE_RATIO = imputeRatio > 1 ? 1 : imputeRatio < 0 ? 0 : imputeRatio;
        LOG.info("== Imputing " + IMPUTE_RATIO * 100f + "% of train data plus its own real present data");

        Table<Integer, Integer, Double> dataTable = HashBasedTable.create();
        Multimap<Integer, Integer> colMap = HashMultimap.create();

        int counter = 0;

        for (Iterator<CombinedRecommendersCell> itr = orderedCombinedRecByVariance.iterator();
             itr.hasNext() && counter < ((float)sparseTrainMatrix.numRows * (float)sparseTrainMatrix.numColumns * IMPUTE_RATIO);
             ++counter) {
            CombinedRecommendersCell cell = itr.next();
            if(cell.getVariance()>2)
                break;
            dataTable.put(cell.getUserIndex(), cell.getItemIndex(), cell.getMean());
            colMap.put(cell.getItemIndex(), cell.getUserIndex());
            lastVariance = cell.getVariance();
            int absVariance = (int) Math.abs(cell.getVariance());
            //differentVariance.add(cell.getVariance()-absVariance < 0.5 ? (double)absVariance : 0.5d + (double)absVariance);
            differentVariance.add(cell.getVariance());
        }
        actualNumberOfImputedCells = --counter;
        maximumNumberOfToBeImputedCells = (long)((float)sparseTrainMatrix.numRows * (float)sparseTrainMatrix.numColumns * IMPUTE_RATIO);
        numberOfWholeCells = (long)(sparseTrainMatrix.numRows * sparseTrainMatrix.numColumns);
        LOG.info(String.format("=== Imputed %s items from %s", actualNumberOfImputedCells, maximumNumberOfToBeImputedCells));
        orderedCombinedRecByVariance = null;
        for(MatrixEntry entry : sparseTrainMatrix){
            if(dataTable.contains(entry.row(), entry.column()))
                LOG.error(String.format("@@@@@@@@ Recommend value already is exist(%s , %s)",entry.row(), entry.column()));

            dataTable.put(entry.row(), entry.column(), entry.get());
            colMap.put(entry.column(), entry.row());
        }
        SparseMatrix imputedTrainMatrix = new SparseMatrix(sparseTrainMatrix.numRows, sparseTrainMatrix.numColumns, dataTable, colMap);

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

//        Configuration svdConf = new Configuration();
//        svdConf.set("rec.recommender.similarity.key", "user");
//        svdConf.set("rec.recommender.class", "svdpp");
//        svdConf.set("rec.iterator.learnrate", "0.01");
//        svdConf.set("rec.iterator.learnrate.maximum", "0.01");
//        svdConf.set("rec.iterator.maximum", "1");
//        svdConf.set("rec.user.regularization", "0.01");
//        svdConf.set("rec.item.regularization", "0.01");
//        svdConf.set("rec.impItem.regularization", "0.001");
//        svdConf.set("rec.factor.number", "20");
//        svdConf.set("rec.learnrate.bolddriver", "false");
//        svdConf.set("rec.learnrate.decay", "1.0");
//
//        Configuration fmalsConf = new Configuration();
//        fmalsConf.set("rec.recommender.similarity.key", "user");
//        fmalsConf.set("rec.iterator.learnRate", "0.001");
//        fmalsConf.set("rec.iterator.maximum", "100");
//        fmalsConf.set("rec.recommender.class", "fmals");
//        fmalsConf.set("rec.recommender.maxrate", "12.0");
//        fmalsConf.set("rec.recommender.minrate", "0.0");
//        fmalsConf.set("rec.factor.number", "10");
//        fmalsConf.set("rec.fm.regw0", "0.01");
//        fmalsConf.set("reg.fm.regW", "0.01");
//        fmalsConf.set("reg.fm.regF", "10");

//        Configuration knnConf = new Configuration();
//        knnConf.set("rec.neighbors.knn.number", "50");

        // build recommender context

        RecommenderContext context = new RecommenderContext(new Configuration(), dataModel);
        // build similarity
        RecommenderSimilarity similarity = new PCCSimilarity();
        similarity.buildSimilarityMatrix(dataModel);
        context.setSimilarity(similarity);
        // build recommender
        //Recommender recommender = new MFALSRecommender();
        Recommender recommender = new SVDPlusPlusRecommender();
        //Recommender recommender = new ItemKNNRecommender();
        //Recommender recommender = new UserKNNRecommender();
        //Recommender recommender = new NMFRecommender();
        //Recommender recommender = new PMFRecommender();

        recommender.setContext(context);
        // run recommender algorithm
        recommender.recommend(context);

        return recommender;
    }

    public EvaluationResult evaluate(Recommender recBySpars, Recommender recByImputed) throws LibrecException {

        EvaluationResult er = new EvaluationResult();

        double maxError = -1000;
        String maxErrorAlg = "";

        RecommenderEvaluator evaluator = new RMSEEvaluator();
        double sparsResult = recBySpars.evaluate(evaluator);

        evaluator = new RMSEEvaluator();
        double imputedResult = recByImputed.evaluate(evaluator);

        er.setSparsRmse(sparsResult);
        er.setImputedRmse(imputedResult);

        LOG.info("== RMSE obtained by sparse train set:  " + sparsResult);
        LOG.info("== RMSE obtained by imputed train set: " + imputedResult);
        LOG.info("== RMSE Differentiation:  " + (sparsResult-imputedResult));
        if ((sparsResult-imputedResult) > maxError){
            maxError = (sparsResult-imputedResult);
            maxErrorAlg = "RMSE";
        }


        evaluator = new MAEEvaluator();
        sparsResult = recBySpars.evaluate(evaluator);

        evaluator = new MAEEvaluator();
        imputedResult = recByImputed.evaluate(evaluator);

        er.setSparsMae(sparsResult);
        er.setImputedMae(imputedResult);
        LOG.info("== MAE obtained by sparse train set:  " + sparsResult);
        LOG.info("== MAE obtained by imputed train set: " + imputedResult);
        LOG.info("== MAE Differentiation:  " + (sparsResult-imputedResult));
        if ((sparsResult-imputedResult) > maxError){
            maxError = (sparsResult-imputedResult);
            maxErrorAlg = "MAE";
        }


        evaluator = new MPEEvaluator();
        sparsResult = recBySpars.evaluate(evaluator);

        evaluator = new MPEEvaluator();
        imputedResult = recByImputed.evaluate(evaluator);

        er.setSparsMpe(sparsResult);
        er.setImputedMpe(imputedResult);
        LOG.info("== MPE obtained by sparse train set:  " + sparsResult);
        LOG.info("== MPE obtained by imputed train set: " + imputedResult);
        LOG.info("== MPE Differentiation:  " + (sparsResult-imputedResult));
        if ((sparsResult-imputedResult) > maxError){
            maxError = (sparsResult-imputedResult);
            maxErrorAlg = "MPE";
        }

        er.setMinDiffError(maxError);
        er.setMinDiffErrorAlg(maxErrorAlg);

        return er;
    }

    private void printResult(EvaluationResult result) throws IOException {
        String finalRecommenderName = "SVDPP";
        StringBuilder builder = new StringBuilder();
        builder.append(String.format("This test has taken %s minutes to complete on %s"
                , Duration.between(start, Instant.now()).getSeconds()/60, LocalDateTime.now()));
        builder.append(String.format("\nNumber of whole cells is: %s", numberOfWholeCells));
        builder.append(String.format("\nNumber of to be imputed cells is: %s", maximumNumberOfToBeImputedCells));
        builder.append(String.format("\nActual number of imputed cells is: %s", actualNumberOfImputedCells));
        builder.append(String.format("\n\nThe evaluation result for '%s' is as the follow:", finalRecommenderName));
        builder.append(String.format("\n\n\tThe best result is for algorithm '%s' with %f differentiation", result.getMinDiffErrorAlg(), result.getMinDiffError()));
        builder.append(String.format("\n\n\tRMSE obtained by sparse train set:\t%f", result.getSparsRmse()));
        builder.append(String.format("\n\tRMSE obtained by imputed train set:\t%f", result.getImputedRmse()));
        builder.append(String.format("\n\n\tMAE obtained by sparse train set:\t%f", result.getSparsMae()));
        builder.append(String.format("\n\tMAE obtained by imputed train set:\t%f", result.getImputedMae()));
        builder.append(String.format("\n\n\tMPE obtained by sparse train set:\t%f", result.getSparsMpe()));
        builder.append(String.format("\n\tMPE obtained by imputed train set:\t%f", result.getImputedMpe()));
        builder.append("\n\nHere is the initial used configuration: ");
        for (Map.Entry<String, String> entry : cfg) {
            builder.append(String.format("\n%s\t=\t%s", entry.getKey(), entry.getValue()));
        }
        builder.append("\n\nHere is the list of variances: \n");
        for(Double v : differentVariance) {
            builder.append(String.format("%+10.4f\n", v));
        }

        String filename = String.format("%s_rounds-%s_impute-%s_%s_%s%f", finalRecommenderName
            , cfg.get("impute.rounds"), cfg.get("impute.ratio")
            , LocalDateTime.now().format(DateTimeFormatter.ofPattern("YYYY-MM-dd-HH-mm-ss"))
            , result.getMinDiffErrorAlg(), result.getMinDiffError());
        String resultPath = cfg.get("dfs.result.dir", ".");
        FileWriter writer = new FileWriter(resultPath + "/" + filename);
        writer.write(builder.toString());
        writer.close();
    }
}
