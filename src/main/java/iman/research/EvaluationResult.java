package iman.research;

public class EvaluationResult {
    private double sparsRmse;
    private double imputedRmse;

    private double sparsMae;
    private double imputedMae;

    private double sparsMpe;
    private double imputedMpe;

    public EvaluationResult() {
    }

    public double getSparsRmse() {
        return sparsRmse;
    }

    public void setSparsRmse(double sparsRmse) {
        this.sparsRmse = sparsRmse;
    }

    public double getImputedRmse() {
        return imputedRmse;
    }

    public void setImputedRmse(double imputedRmse) {
        this.imputedRmse = imputedRmse;
    }

    public double getSparsMae() {
        return sparsMae;
    }

    public void setSparsMae(double sparsMae) {
        this.sparsMae = sparsMae;
    }

    public double getImputedMae() {
        return imputedMae;
    }

    public void setImputedMae(double imputedMae) {
        this.imputedMae = imputedMae;
    }

    public double getSparsMpe() {
        return sparsMpe;
    }

    public void setSparsMpe(double sparsMpe) {
        this.sparsMpe = sparsMpe;
    }

    public double getImputedMpe() {
        return imputedMpe;
    }

    public void setImputedMpe(double imputedMpe) {
        this.imputedMpe = imputedMpe;
    }
}
