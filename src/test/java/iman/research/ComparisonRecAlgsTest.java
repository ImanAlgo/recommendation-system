package iman.research;

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.model.TextDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.rating.RMSEEvaluator;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.UserKNNRecommender;
import net.librec.recommender.cf.rating.MFALSRecommender;
import net.librec.recommender.cf.rating.SVDPlusPlusRecommender;
import net.librec.recommender.ext.SlopeOneRecommender;
import net.librec.recommender.item.RecommendedItemList;
import net.librec.recommender.item.RecommendedList;
import net.librec.recommender.item.UserItemRatingEntry;
import net.librec.similarity.PCCSimilarity;
import net.librec.similarity.RecommenderSimilarity;

public class ComparisonRecAlgsTest {
    // Algorithms
    final int KNN = 0;
    final int ALS = 1;
    final int SVD = 2;
    final int ONE = 3;
    final int nAlgorithms = 4;
    Configuration conf = null;
    TextDataModel MotherDataModel = null;
    OutputData data = new OutputData();
    final String dataFolder = "1M";
    final String dataPath = "/Users/victorwegeborn/Documents" + "/KTH/vt17/kexdata/movielens/20M";

    class OutputData {
        double[] rmse = new double[nAlgorithms];
        double[] interp_rmse = new double[2];
        double rmseSum = 0;
        int numberOfUsers = 0;
        HashMap<Integer, RecommendedList> output = new HashMap<Integer, RecommendedList>();
    };

    public static void main(String[] args) throws Exception {
        ComparisonRecAlgsTest t = new ComparisonRecAlgsTest();
        t.setupData();
        t.SlopeOne();
        t.KNN();
        t.SVD();
        t.MFALS();
        t.EvaluteAlgorithms();
        t.OutputResults();
    }

    void setupData() throws LibrecException {
        conf = new Configuration();
        conf.set("dfs.data.dir", dataPath);
        conf.set("dfs.result.dir", "/Users/victorwegeborn/Documents/KTH/vt17/kexdata/libec/result");
        conf.set("data.input.path", dataFolder);
        conf.set("data.column.format", "UIR");
        conf.set("data.model.splitter", "ratio");
        conf.set("data.splitter.ratio", "rating");
        conf.set("data.model.format", "text");
        conf.set("data.splitter.trainset.ratio", "0.8");
        conf.set("rec.random.seed", "1");
        conf.set("data.convert.binarize.threshold", "-1.0");
        MotherDataModel = new TextDataModel(conf);
        MotherDataModel.buildDataModel();
        data.numberOfUsers = MotherDataModel.getDataSplitter().getTrainData().numRows;
    }

    void SlopeOne() throws Exception {
        Configuration ONEconf = new Configuration();
        ONEconf.set("rec.recommender.isranking", "false");
        ONEconf.set("rec.recommender.class", "slopeone");
        ONEconf.set("rec.eval.enable", "true");
        ONEconf.set("rec.iterator.maximum", "50");
        ONEconf.set("rec.factory.number", "30");
        ONEconf.set("rec.iterator.learn.rate", "0.001");
        ONEconf.set("rec.recommender.lambda.user", "0.05");
        ONEconf.set("rec.recommender.lambda.item", "0.05");
        // build recommender context
        TextDataModel dataModel = MotherDataModel;
        RecommenderContext context = new RecommenderContext(ONEconf, dataModel);
        // build recommender
        Recommender recommender = new SlopeOneRecommender();
        recommender.setContext(context);
        // run recommender algorithm
        recommender.recommend(context);
        // filter the recommended result
        RecommenderEvaluator evaluator = new RMSEEvaluator();
        data.rmse[ONE] = recommender.evaluate(evaluator);
        data.output.put(ONE, recommender.getRawRecommendedList());
    }

    void MFALS() throws Exception {
        Configuration ALSconf = new Configuration();
        ALSconf.set("rec.recommender.similarity.key", "user");
        ALSconf.set("rec.recommender.class", "mfals");
        ALSconf.set("rec.iterator.learnrate", "0.01");
        ALSconf.set("rec.iterator.learnrate.maximum", "0.01");
        ALSconf.set("rec.iterator.maximum", "100");
        ALSconf.set("rec.user.regularization", "0.01");
        ALSconf.set("rec.item.regularization", "0.01");
        ALSconf.set("rec.factor.number", "10");
        ALSconf.set("rec.learnrate.bolddriver", "false");
        ALSconf.set("rec.learnrate.decay", "1.0");
        // set recommender context
        TextDataModel dataModel = MotherDataModel;
        RecommenderContext context = new RecommenderContext(ALSconf, dataModel);
        // build similarity
        RecommenderSimilarity similarity = new PCCSimilarity();
        similarity.buildSimilarityMatrix(dataModel);
        context.setSimilarity(similarity);
        // build recommender
        Recommender recommender = new MFALSRecommender();
        recommender.setContext(context);
        // run recommender algorithm
        recommender.recommend(context);
        // evaluate the recommended result
        RecommenderEvaluator evaluator = new RMSEEvaluator();
        data.rmse[ALS] = recommender.evaluate(evaluator);
        data.output.put(ALS, recommender.getRawRecommendedList());
    }

    void SVD() throws Exception {
        Configuration SVDconf = new Configuration();
        SVDconf.set("rec.recommender.similarity.key", "user");
        SVDconf.set("rec.recommender.class", "svdpp");
        SVDconf.set("rec.iterator.learnrate", "0.01");
        SVDconf.set("rec.iterator.learnrate.maximum", "0.01");
        SVDconf.set("rec.iterator.maximum", "13");
        SVDconf.set("rec.user.regularization", "0.01");
        SVDconf.set("rec.item.regularization", "0.01");
        SVDconf.set("rec.impItem.regularization", "0.001");
        SVDconf.set("rec.factor.number", "10");
        SVDconf.set("rec.learnrate.bolddriver", "false");
        SVDconf.set("rec.learnrate.decay", "1.0");
        // build recommender context
        TextDataModel dataModel = MotherDataModel;
        RecommenderContext context = new RecommenderContext(SVDconf, dataModel);
        // build similarity
        RecommenderSimilarity similarity = new PCCSimilarity();
        similarity.buildSimilarityMatrix(dataModel);
        context.setSimilarity(similarity);
        // build recommender
        Recommender recommender = new SVDPlusPlusRecommender();
        recommender.setContext(context);
        // run recommender algorithm
        recommender.recommend(context);
        // evaluate the recommended result
        RecommenderEvaluator evaluator = new RMSEEvaluator();
        data.rmse[SVD] = recommender.evaluate(evaluator);
        data.output.put(SVD, recommender.getRawRecommendedList());
    }

    void KNN() throws Exception {
        Configuration KNNconf = new Configuration();
        KNNconf.set("rec.similarity.class", "pcc");
        KNNconf.set("rec.neighbors.KNN.number", "80");
        KNNconf.set("rec.recommender.class", "userKNN");
        KNNconf.set("rec.recommender.similarities", "user");
        KNNconf.set("rec.recommender.isranking", "false");
        KNNconf.set("rec.recommender.ranking.topn", "10");
        KNNconf.set("rec.filter.class", "generic");
        KNNconf.set("rec.similarity.shrinkage", "25");
        KNNconf.set("rec.recommender.verbose", "true");
        // build recommender context
        TextDataModel dataModel = MotherDataModel;
        RecommenderContext context = new RecommenderContext(KNNconf, dataModel);
        // build similarity
        RecommenderSimilarity similarity = new PCCSimilarity();
        similarity.buildSimilarityMatrix(dataModel);
        context.setSimilarity(similarity);
        // build recommender
        Recommender recommender = new UserKNNRecommender();
        recommender.setContext(context);
        // run recommender algorithm
        recommender.recommend(context);
        // evaluate the recommended result
        RecommenderEvaluator evaluator = new RMSEEvaluator();
        data.rmse[KNN] = recommender.evaluate(evaluator);
        data.output.put(KNN, recommender.getRawRecommendedList());
    }

    void EvaluteAlgorithms() throws Exception {
        System.err.println("============= RESULTS =============");
        System.out.println("RMSE VALUES");
        System.out.println("SLOPE ONE :: RMSE =" + data.rmse[ONE]);
        System.out.println("KNN :: RMSE =" + data.rmse[KNN]);
        System.out.println("SVD ++ :: RMSE =" + data.rmse[SVD]);
        System.out.println("ALS :: RMSE =" + data.rmse[ALS]);
        data.rmseSum = data.rmse[ONE] + data.rmse[ALS] + data.rmse[KNN] + data.rmse[SVD];
        int ALSsize = data.output.get(ALS).size();
        int SVDsize = data.output.get(SVD).size();
        int KNNsize = data.output.get(KNN).size();
        int ONEsize = data.output.get(ONE).size();
        if (!(ALSsize == SVDsize && SVDsize == KNNsize && KNNsize == ONEsize))
            throw new Exception("OUTPUT DATA LISTS ARE NOT OF SAME LENGHT");
        Iterator<UserItemRatingEntry> ALSiterator = data.output.get(ALS).entryIterator();
        Iterator<UserItemRatingEntry> KNNiterator = data.output.get(KNN).entryIterator();
        Iterator<UserItemRatingEntry> SVDiterator = data.output.get(SVD).entryIterator();
        Iterator<UserItemRatingEntry> ONEiterator = data.output.get(ONE).entryIterator();
        RecommendedList AMrating = new RecommendedItemList(data.numberOfUsers);
        RecommendedList WAMrating = new RecommendedItemList(data.numberOfUsers);
        ;
        double[] factors = { data.rmse[ALS] / data.rmseSum, data.rmse[KNN] / data.rmseSum,
                data.rmse[SVD] / data.rmseSum, data.rmse[ONE] / data.rmseSum };
        while (ALSiterator.hasNext() && KNNiterator.hasNext() && SVDiterator.hasNext() && ONEiterator.hasNext()) {
            LinkedList<UserItemRatingEntry> entries = new LinkedList<UserItemRatingEntry>();
            entries.add(ALSiterator.next());
            entries.add(KNNiterator.next());
            entries.add(SVDiterator.next());
            entries.add(ONEiterator.next());
            int userID = entries.get(0).getUserIdx();
            int itemIdx = entries.get(0).getItemIdx();
            LinkedList<Double> ratings = new LinkedList<Double>();
            for (UserItemRatingEntry e : entries) {
                if (e.getUserIdx() == userID && e.getItemIdx() == itemIdx) {
                    ratings.add(e.getValue());
                } else
                    throw new Exception("Iteration through recommendation lists are non uniform");
            }
            AMrating.addUserItemIdx(userID, itemIdx, AM(ratings));
            WAMrating.addUserItemIdx(userID, itemIdx, WAM(ratings, factors));
        }
        // Make sure no more users are left in the lists
        if (KNNiterator.hasNext())
            throw new Exception("User/item pairs not processed");
        if (SVDiterator.hasNext())
            throw new Exception("User/item pairs not processed");
        if (ONEiterator.hasNext())
            throw new Exception("User/item pairs not processed");
        RMSEEvaluator evaluator = new RMSEEvaluator();
        data.interp_rmse[0] = evaluator.evaluate(MotherDataModel.getDataSplitter().getTestData(), AMrating);
        data.interp_rmse[1] = evaluator.evaluate(MotherDataModel.getDataSplitter().getTestData(), WAMrating);
        System.out.println("~ AM :: RMSE =" + data.interp_rmse[0]);
        System.out.println("~ WAM :: RMSE =" + data.interp_rmse[1]);
    }

    double AM(LinkedList<Double> values) {
        double sum = 0;
        for (Double value : values)
            sum += value;
        return sum / values.size();
    }

    double WAM(LinkedList<Double> values, double[] factors) {
        double sum = 0;
        for (int i = 0; i < values.size(); i++) {
            sum += ((1 - factors[i]) * values.get(i));
        }
        return sum / (values.size());
    }

    void OutputResults() throws UnsupportedEncodingException, FileNotFoundException, IOException {
        try {
            FileWriter writer = new FileWriter(dataPath + "/results/results.txt", true)
            StringBuilder sb = new StringBuilder();
            sb.append("================= BEGIN ===================\n");
            sb.append("RESULTS ::" + dataFolder + "\n");
            sb.append("SLOPE ONE ::" + data.rmse[ONE] + "\n");
            sb.append("KNN ::" + data.rmse[KNN] + "\n");
            sb.append("SVD ::" + data.rmse[SVD] + "\n");
            sb.append("MFALS ::" + data.rmse[ALS] + "\n");
            sb.append("AM ::" + data.interp_rmse[0] + "\n");
            sb.append("WAM ::" + data.interp_rmse[1] + "\n");
            sb.append("================= END =====================\n\n");
            writer.append(sb.toString());
        } catch (Exception e){
            //IGNORE
        }
    }
}