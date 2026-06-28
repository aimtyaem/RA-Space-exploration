// ============================================================
// VotingEnsemble.java
// A reusable class for building and comparing voting classifiers
// using WEKA. Supports Logistic Regression, J48, and k-NN.
// ============================================================

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

import java.io.File;
import java.util.Random;

/**
 * A builder/trainer for an ensemble of three base classifiers:
 * Logistic Regression, Decision Tree (J48), and k-NN.
 * Implements hard voting, soft voting (average), and weighted soft voting.
 * The best method (single or ensemble) is selected based on validation accuracy.
 */
public class VotingEnsemble {

    // -------------------- Configuration --------------------
    private String datasetPath;           // path to CSV file
    private double trainRatio = 0.6;      // fraction for training
    private double valRatio = 0.2;        // fraction for validation
    private double testRatio = 0.2;       // fraction for test (computed if not set)
    private int seed = 42;                // random seed for reproducibility
    private int[] kValues = {3, 5, 7};    // k values to tune for k-NN

    // -------------------- Internal state --------------------
    private Instances train, val, test;
    private Instances trainStd, valStd, testStd;
    private Classifier logistic, j48, knn;
    private double accLogistic, accJ48, accKnn;
    private int bestK;
    private String bestSingleModelName;
    private double bestSingleAcc;
    private int selectedMethodType;       // 0=single, 1=hard, 2=soft, 3=weighted
    private String selectedMethodName;
    private double selectedValAcc;

    // -------------------- Constructors --------------------
    public VotingEnsemble(String datasetPath) {
        this.datasetPath = datasetPath;
    }

    // -------------------- Configuration setters --------------------
    public VotingEnsemble setDatasetPath(String path) {
        this.datasetPath = path;
        return this;
    }

    public VotingEnsemble setSplitRatios(double train, double val, double test) {
        if (Math.abs(train + val + test - 1.0) > 1e-6) {
            throw new IllegalArgumentException("Ratios must sum to 1.0");
        }
        this.trainRatio = train;
        this.valRatio = val;
        this.testRatio = test;
        return this;
    }

    public VotingEnsemble setSeed(int seed) {
        this.seed = seed;
        return this;
    }

    public VotingEnsemble setKValues(int... kValues) {
        this.kValues = kValues;
        return this;
    }

    // -------------------- Main workflow --------------------
    /**
     * Executes the entire pipeline:
     * 1. Load and split data
     * 2. Standardize features
     * 3. Train base models (with k-NN tuning)
     * 4. Evaluate ensembles and select best method
     * 5. Report final test results
     *
     * @return EnsembleResult containing metrics and best model info
     * @throws Exception if any WEKA operation fails
     */
    public EnsembleResult run() throws Exception {
        loadAndSplitData();
        standardizeFeatures();
        trainBaseModels();
        tuneKnn();
        evaluateEnsemblesAndSelectBest();
        return evaluateOnTestSet();
    }

    // -------------------- Step 1: Load & Split --------------------
    private void loadAndSplitData() throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(datasetPath));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        System.out.println("Dataset loaded: " + data.numInstances() + " instances, "
                + data.numAttributes() + " attributes.");

        // Shuffle and stratify
        data.randomize(new Random(seed));
        if (data.classAttribute().isNominal()) {
            data.stratify(5);
        }

        int n = data.numInstances();
        int nTrain = (int) (n * trainRatio);
        int nVal = (int) (n * valRatio);
        int nTest = n - nTrain - nVal;

        train = new Instances(data, 0, nTrain);
        val = new Instances(data, nTrain, nVal);
        test = new Instances(data, nTrain + nVal, nTest);

        System.out.println("Splits: train=" + train.numInstances()
                + ", val=" + val.numInstances()
                + ", test=" + test.numInstances());
    }

    // -------------------- Step 2: Standardize --------------------
    private void standardizeFeatures() throws Exception {
        Standardize std = new Standardize();
        std.setInputFormat(train);
        trainStd = Filter.useFilter(train, std);
        valStd = Filter.useFilter(val, std);
        testStd = Filter.useFilter(test, std);
        System.out.println("Standardization applied.");
    }

    // -------------------- Step 3: Train base models --------------------
    private void trainBaseModels() throws Exception {
        logistic = new Logistic();
        logistic.buildClassifier(trainStd);

        j48 = new J48();
        j48.buildClassifier(trainStd);

        knn = new IBk(5); // will be tuned later
        knn.buildClassifier(trainStd);

        // Evaluate initial performance
        accLogistic = evaluateAccuracy(logistic, valStd);
        accJ48 = evaluateAccuracy(j48, valStd);
        accKnn = evaluateAccuracy(knn, valStd);

        System.out.println("\n=== Base Model Validation Accuracies ===");
        System.out.printf("Logistic: %.3f\n", accLogistic);
        System.out.printf("J48:      %.3f\n", accJ48);
        System.out.printf("k-NN (k=5): %.3f\n", accKnn);

        // Determine best single model
        bestSingleModelName = "Logistic";
        bestSingleAcc = accLogistic;
        if (accJ48 > bestSingleAcc) {
            bestSingleAcc = accJ48;
            bestSingleModelName = "J48";
        }
        if (accKnn > bestSingleAcc) {
            bestSingleAcc = accKnn;
            bestSingleModelName = "k-NN";
        }
        System.out.println("Best single model: " + bestSingleModelName
                + " with accuracy " + String.format("%.3f", bestSingleAcc));
    }

    // -------------------- Step 4: Tune k-NN --------------------
    private void tuneKnn() throws Exception {
        System.out.println("\n=== k-NN Tuning ===");
        int bestK = 5;
        double bestKAcc = accKnn;
        for (int k : kValues) {
            IBk temp = new IBk(k);
            temp.buildClassifier(trainStd);
            double acc = evaluateAccuracy(temp, valStd);
            System.out.printf("k=%d: accuracy=%.3f\n", k, acc);
            if (acc > bestKAcc) {
                bestKAcc = acc;
                bestK = k;
            }
        }
        if (bestK != 5) {
            knn = new IBk(bestK);
            knn.buildClassifier(trainStd);
            accKnn = bestKAcc;
        }
        this.bestK = bestK;
        System.out.println("Using k=" + bestK);
    }

    // -------------------- Step 5: Evaluate ensembles --------------------
    private void evaluateEnsemblesAndSelectBest() throws Exception {
        double[] weights = computeWeights(accLogistic, accJ48, accKnn);

        int correctHard = 0, correctSoft = 0, correctWeighted = 0;
        for (int i = 0; i < valStd.numInstances(); i++) {
            Instance x = valStd.instance(i);
            int actual = (int) x.classValue();

            if (predictHard(x, logistic, j48, knn) == actual) correctHard++;
            if (predictSoftAvg(x, logistic, j48, knn) == actual) correctSoft++;
            if (predictSoftWeighted(x, logistic, j48, knn, weights[0], weights[1], weights[2]) == actual)
                correctWeighted++;
        }

        double accHard = correctHard / (double) valStd.numInstances();
        double accSoft = correctSoft / (double) valStd.numInstances();
        double accWeighted = correctWeighted / (double) valStd.numInstances();

        System.out.println("\n=== Ensemble Validation Accuracies ===");
        System.out.printf("Hard Voting:      %.3f\n", accHard);
        System.out.printf("Soft (average):   %.3f\n", accSoft);
        System.out.printf("Soft (weighted):  %.3f\n", accWeighted);
        System.out.printf("Best single:      %.3f\n", bestSingleAcc);

        // Select the best method
        selectedMethodType = 0; // single
        selectedMethodName = bestSingleModelName + " (single)";
        selectedValAcc = bestSingleAcc;

        if (accHard > selectedValAcc) {
            selectedValAcc = accHard;
            selectedMethodName = "Hard Voting";
            selectedMethodType = 1;
        }
        if (accSoft > selectedValAcc) {
            selectedValAcc = accSoft;
            selectedMethodName = "Soft Voting (Average)";
            selectedMethodType = 2;
        }
        if (accWeighted > selectedValAcc) {
            selectedValAcc = accWeighted;
            selectedMethodName = "Soft Voting (Weighted)";
            selectedMethodType = 3;
        }

        System.out.println("\nBest method: " + selectedMethodName
                + " with validation accuracy " + String.format("%.3f", selectedValAcc));
    }

    // -------------------- Step 6: Final test evaluation --------------------
    private EnsembleResult evaluateOnTestSet() throws Exception {
        int[] predictions = new int[testStd.numInstances()];
        int[] actuals = new int[testStd.numInstances()];
        int correct = 0;

        for (int i = 0; i < testStd.numInstances(); i++) {
            Instance x = testStd.instance(i);
            int actual = (int) x.classValue();
            actuals[i] = actual;
            int predicted = predictWithSelectedMethod(x);
            predictions[i] = predicted;
            if (predicted == actual) correct++;
        }

        double testAcc = correct / (double) testStd.numInstances();

        // Build confusion matrix (binary)
        int[][] cm = new int[2][2];
        for (int i = 0; i < testStd.numInstances(); i++) {
            cm[actuals[i]][predictions[i]]++;
        }

        double tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
        double precision = (tp + fp == 0) ? 0 : tp / (tp + fp);
        double recall = (tp + fn == 0) ? 0 : tp / (tp + fn);
        double f1 = (precision + recall == 0) ? 0 : 2 * (precision * recall) / (precision + recall);

        System.out.println("\n=== FINAL TEST RESULTS ===");
        System.out.println("Method: " + selectedMethodName);
        System.out.printf("Test Accuracy: %.3f\n", testAcc);
        System.out.println("Confusion Matrix (rows=actual, cols=predicted):");
        System.out.println("       Pred_0  Pred_1");
        System.out.printf("Act_0: %6d  %6d\n", cm[0][0], cm[0][1]);
        System.out.printf("Act_1: %6d  %6d\n", cm[1][0], cm[1][1]);
        System.out.printf("\nPrecision: %.3f\n", precision);
        System.out.printf("Recall:    %.3f\n", recall);
        System.out.printf("F1-Score:  %.3f\n", f1);

        return new EnsembleResult(selectedMethodName, testAcc, cm, precision, recall, f1,
                logistic, j48, knn, selectedMethodType);
    }

    // -------------------- Prediction helpers --------------------
    private int predictHard(Instance x, Classifier lr, Classifier dt, Classifier knn) throws Exception {
        int[] votes = new int[x.classAttribute().numValues()];
        votes[(int) lr.classifyInstance(x)]++;
        votes[(int) dt.classifyInstance(x)]++;
        votes[(int) knn.classifyInstance(x)]++;
        int max = 0, best = 0;
        for (int i = 0; i < votes.length; i++) {
            if (votes[i] > max) {
                max = votes[i];
                best = i;
            }
        }
        return best;
    }

    private int predictSoftAvg(Instance x, Classifier lr, Classifier dt, Classifier knn) throws Exception {
        double[] pLr = lr.distributionForInstance(x);
        double[] pDt = dt.distributionForInstance(x);
        double[] pKnn = knn.distributionForInstance(x);
        double[] avg = new double[pLr.length];
        for (int i = 0; i < avg.length; i++) {
            avg[i] = (pLr[i] + pDt[i] + pKnn[i]) / 3.0;
        }
        return argmax(avg);
    }

    private int predictSoftWeighted(Instance x, Classifier lr, Classifier dt, Classifier knn,
                                    double wLr, double wDt, double wKnn) throws Exception {
        double[] pLr = lr.distributionForInstance(x);
        double[] pDt = dt.distributionForInstance(x);
        double[] pKnn = knn.distributionForInstance(x);
        double[] weighted = new double[pLr.length];
        for (int i = 0; i < weighted.length; i++) {
            weighted[i] = wLr * pLr[i] + wDt * pDt[i] + wKnn * pKnn[i];
        }
        return argmax(weighted);
    }

    private int predictWithSelectedMethod(Instance x) throws Exception {
        switch (selectedMethodType) {
            case 0: // single
                if (bestSingleModelName.equals("Logistic")) return (int) logistic.classifyInstance(x);
                else if (bestSingleModelName.equals("J48")) return (int) j48.classifyInstance(x);
                else return (int) knn.classifyInstance(x);
            case 1: return predictHard(x, logistic, j48, knn);
            case 2: return predictSoftAvg(x, logistic, j48, knn);
            case 3:
                double[] w = computeWeights(accLogistic, accJ48, accKnn);
                return predictSoftWeighted(x, logistic, j48, knn, w[0], w[1], w[2]);
            default: throw new IllegalStateException("Unknown method type");
        }
    }

    // -------------------- Utility methods --------------------
    private double evaluateAccuracy(Classifier model, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(model, data);
        return eval.pctCorrect() / 100.0;
    }

    private double[] computeWeights(double a, double b, double c) {
        double sum = a + b + c;
        return new double[]{a / sum, b / sum, c / sum};
    }

    private int argmax(double[] probs) {
        int best = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[best]) best = i;
        }
        return best;
    }

    // -------------------- Result container --------------------
    public static class EnsembleResult {
        public final String methodName;
        public final double testAccuracy;
        public final int[][] confusionMatrix;
        public final double precision, recall, f1;
        // You may store the trained classifiers for later use
        public final Classifier logistic, j48, knn;
        public final int methodType;

        public EnsembleResult(String methodName, double testAccuracy, int[][] cm,
                              double precision, double recall, double f1,
                              Classifier logistic, Classifier j48, Classifier knn,
                              int methodType) {
            this.methodName = methodName;
            this.testAccuracy = testAccuracy;
            this.confusionMatrix = cm;
            this.precision = precision;
            this.recall = recall;
            this.f1 = f1;
            this.logistic = logistic;
            this.j48 = j48;
            this.knn = knn;
            this.methodType = methodType;
        }

        @Override
        public String toString() {
            return String.format("EnsembleResult{method='%s', testAcc=%.3f, precision=%.3f, recall=%.3f, f1=%.3f}",
                    methodName, testAccuracy, precision, recall, f1);
        }
    }

    // -------------------- Example usage (main) --------------------
    public static void main(String[] args) {
        try {
            // Adjust the path to your CSV file
            String path = "online_shoppers_intention.csv";
            VotingEnsemble builder = new VotingEnsemble(path)
                    .setSplitRatios(0.6, 0.2, 0.2)
                    .setSeed(42)
                    .setKValues(3, 5, 7);

            EnsembleResult result = builder.run();
            System.out.println("\n=== SUMMARY ===");
            System.out.println(result);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}