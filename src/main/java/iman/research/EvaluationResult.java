package iman.research;

public class EvaluationResult {
    double sparsRmse;
    double imputedRmse;

    public EvaluationResult(double sparsRmse, double imputedRmse) {
        this.sparsRmse = sparsRmse;
        this.imputedRmse = imputedRmse;
    }

    public double getSparsRmse() {
        return sparsRmse;
    }

    public double getImputedRmse() {
        return imputedRmse;
    }
}
