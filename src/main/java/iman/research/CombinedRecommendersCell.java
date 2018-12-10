package iman.research;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import net.librec.math.algorithm.Maths;

class CombinedRecommendersCell implements Comparable<CombinedRecommendersCell> {
    private int userIndex;
    private int itemIndex;
    private double varianceRate;
    private double meanRate;
    private List<Double> rates;

    public CombinedRecommendersCell(int userIndex, int itemIndex) {
        this.userIndex = userIndex;
        this.itemIndex = itemIndex;
        this.varianceRate = -1;
        this.rates = new ArrayList<Double>();
    }

    /**
     * @return the userIndex
     */
    public int getUserIndex() {
        return userIndex;
    }

    /**
     * @return the itemIndex
     */
    public int getItemIndex() {
        return itemIndex;
    }

    public void addRate(double rate){
        rates.add(rate);
        this.varianceRate = -1;
    }

    public double getMean(){
        if(varianceRate < 0)
            meanRate = Maths.mean(this.rates);
        return meanRate;
    }
    public double getVariance() {
        if(varianceRate < 0){
            meanRate = Maths.mean(this.rates);
            varianceRate = 0;
            varianceRate = rates.stream()
                            .collect(Collectors.averagingDouble(r->Math.pow(r-meanRate, 2)));
        }

        return varianceRate;
    }

    @Override
    public boolean equals(Object obj) {
        return super.equals(obj);
    }

    @Override
    public int compareTo(CombinedRecommendersCell o) {
        return (this.getVariance() == o.getVariance() ? 0 : (this.getVariance() > o.getVariance() ? 1 : -1));
    }
    
}