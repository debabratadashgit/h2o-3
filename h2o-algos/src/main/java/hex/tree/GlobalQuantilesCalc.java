package hex.tree;

import hex.quantile.Quantile;
import hex.quantile.QuantileModel;
import water.DKV;
import water.Job;
import water.Key;
import water.fvec.Frame;
import water.util.ArrayUtils;

public class GlobalQuantilesCalc {

    static double[][] globalQuantiles(Frame fr, String weightsColumn, final int N, int nbins_top_level) {
        QuantileModel.QuantileParameters p = new QuantileModel.QuantileParameters();
        Key rndKey = Key.make();
        if (DKV.get(rndKey)==null) DKV.put(rndKey, fr);
        p._train = rndKey;
        p._weights_column = weightsColumn;
        p._combine_method = QuantileModel.CombineMethod.INTERPOLATE;
        p._probs = new double[N];
        for (int i = 0; i < N; ++i) //compute quantiles such that they span from (inclusive) min...maxEx (exclusive)
            p._probs[i] = i * 1./N;
        Job<QuantileModel> job = new Quantile(p).trainModel();
        QuantileModel qm = job.get();
        job.remove();
        double[][] origQuantiles = qm._output._quantiles;
        //pad the quantiles until we have nbins_top_level bins
        double[][] splitPoints = new double[origQuantiles.length][];
        for (int i=0;i<origQuantiles.length;++i) {
            if (!fr.vec(i).isNumeric() || fr.vec(i).isCategorical() || fr.vec(i).isBinary() || origQuantiles[i].length <= 1) {
                continue;
            }
            // make the quantiles split points unique
            splitPoints[i] = ArrayUtils.makeUniqueAndLimitToRange(origQuantiles[i], fr.vec(i).min(), fr.vec(i).max());
            if (splitPoints[i].length <= 1) //not enough split points left - fall back to regular binning
                splitPoints[i] = null;
            else
                splitPoints[i] = ArrayUtils.padUniformly(splitPoints[i], nbins_top_level);
            assert splitPoints[i] == null || splitPoints[i].length > 1;
        }
        qm.delete();
        DKV.remove(rndKey);
        return splitPoints;
    }

}
