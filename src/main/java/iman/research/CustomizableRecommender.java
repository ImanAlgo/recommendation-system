package iman.research;

import net.librec.common.LibrecException;
import net.librec.data.DataModel;
import net.librec.eval.Measure;
import net.librec.eval.RecommenderEvaluator;
import net.librec.recommender.AbstractRecommender;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.item.RecommendedItem;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;

public class CustomizableRecommender implements Recommender {

    private static final Log LOG = LogFactory.getLog(CustomizableRecommender.class);

    private AbstractRecommender recommender;
    private Method predictMethod;

    private CustomizableRecommender(AbstractRecommender recommender) {
        this.recommender = recommender;

        Class<?> c = recommender.getClass();
        while (c.getSuperclass() != null) {
            try {
                this.predictMethod = c.getDeclaredMethod("predict", int.class, int.class, boolean.class);
                this.predictMethod.setAccessible(true);
            } catch (NoSuchMethodException e) {
                //IGNORE - LOG.fatal(e.getMessage());
            }
            c = c.getSuperclass();
        }

    }

    public static Recommender create(AbstractRecommender recommender) {
        if (recommender == null)
            return null;
        return new CustomizableRecommender(recommender);
    }

    public double predict(int userIdx, int itemIdx) throws LibrecException {

        try {
            return (double)predictMethod.invoke(recommender, userIdx, itemIdx, true);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            throw (LibrecException)e.getTargetException();
        }
        return 0;
    }


    /**
     * recommend
     *
     * @param context recommender context
     * @throws LibrecException if error occurs during recommending
     */
    @Override
    public void recommend(RecommenderContext context) throws LibrecException {
        recommender.recommend(context);
    }

    /**
     * evaluate
     *
     * @param evaluator recommender evaluator
     * @return evaluate result
     * @throws LibrecException if error occurs during evaluating
     */
    @Override
    public double evaluate(RecommenderEvaluator evaluator) throws LibrecException {
        return recommender.evaluate(evaluator);
    }

    /**
     * evaluate Map
     *
     * @return evaluate map
     * @throws LibrecException if error occurs during constructing evaluate map
     */
    @Override
    public Map<Measure.MeasureValue, Double> evaluateMap() throws LibrecException {
        return recommender.evaluateMap();
    }

    /**
     * get DataModel
     *
     * @return data model
     */
    @Override
    public DataModel getDataModel() {
        return recommender.getDataModel();
    }

    /**
     * load Model
     *
     * @param filePath file path
     */
    @Override
    public void loadModel(String filePath) {
        recommender.loadModel(filePath);
    }

    /**
     * save Model
     *
     * @param filePath file path
     */
    @Override
    public void saveModel(String filePath) {
        recommender.saveModel(filePath);
    }

    /**
     * get Recommended List
     *
     * @return recommended list
     */
    @Override
    public List<RecommendedItem> getRecommendedList() {
        return recommender.getRecommendedList();
    }

    /**
     * set Context
     *
     * @param context recommender context
     */
    @Override
    public void setContext(RecommenderContext context) {
        recommender.setContext(context);
    }
}
